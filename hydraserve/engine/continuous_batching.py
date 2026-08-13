from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock

from hydraserve.cache.block_manager import KVBlockManager
from hydraserve.engine.scheduler import Request, RequestState


@dataclass(frozen=True, slots=True)
class PrefillBatchItem:
    request_id: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class DecodeBatch:
    request_ids: tuple[int, ...]
    token_ids: tuple[int, ...]


class ContinuousBatchScheduler:
    """Iteration-level scheduler: each active sequence contributes one decode token."""

    def __init__(
        self,
        block_manager: KVBlockManager,
        *,
        max_decode_sequences: int = 64,
        max_prefill_tokens: int = 8192,
        prefill_chunk_size: int = 4096,
    ) -> None:
        if min(max_decode_sequences, max_prefill_tokens, prefill_chunk_size) <= 0:
            raise ValueError("scheduler limits must be positive")
        self.block_manager = block_manager
        self.max_decode_sequences = max_decode_sequences
        self.max_prefill_tokens = max_prefill_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self._requests: dict[int, Request] = {}
        self._waiting: deque[int] = deque()
        self._prefill_offsets: dict[int, int] = {}
        self._ready: deque[int] = deque()
        self._running: dict[int, Request] = {}
        self._preempted: deque[int] = deque()
        self._inflight: DecodeBatch | None = None
        self._lock = RLock()

    def add(self, request: Request) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(f"duplicate request {request.request_id}")
            if request.state is not RequestState.WAITING:
                raise ValueError("only waiting requests can enter continuous batching")
            self._requests[request.request_id] = request
            self._waiting.append(request.request_id)
            self._prefill_offsets[request.request_id] = 0

    def next_prefill_batch(self) -> tuple[PrefillBatchItem, ...]:
        """FCFS chunked-prefill selection under a global token budget."""
        with self._lock:
            budget = self.max_prefill_tokens
            items: list[PrefillBatchItem] = []
            visits = len(self._waiting)
            for _ in range(visits):
                if budget == 0:
                    break
                request_id = self._waiting.popleft()
                request = self._requests[request_id]
                start = self._prefill_offsets[request_id]
                remaining = len(request.token_ids) - start
                size = min(remaining, self.prefill_chunk_size, budget)
                if size <= 0:
                    continue
                if request.state is RequestState.WAITING:
                    request.transition(RequestState.PREFILL_RUNNING)
                end = start + size
                self._prefill_offsets[request_id] = end
                items.append(PrefillBatchItem(request_id, start, end))
                budget -= size
                if end < len(request.token_ids):
                    self._waiting.append(request_id)
            return tuple(items)

    def mark_prefill_complete(self, request_id: int, *, transferred: bool = False) -> None:
        with self._lock:
            request = self._request(request_id)
            if self._prefill_offsets[request_id] != len(request.token_ids):
                raise RuntimeError("cannot complete a partially processed prompt")
            if request.state is not RequestState.PREFILL_RUNNING:
                raise RuntimeError("request is not in prefill")
            if transferred:
                request.transition(RequestState.TRANSFER_PENDING)
            else:
                request.transition(RequestState.READY)
                self._ready.append(request_id)

    def mark_transfer_complete(self, request_id: int) -> None:
        with self._lock:
            request = self._request(request_id)
            if request.state is not RequestState.TRANSFER_PENDING:
                raise RuntimeError("request is not awaiting transfer")
            request.transition(RequestState.READY)
            self._ready.append(request_id)

    def next_decode_batch(self) -> DecodeBatch:
        with self._lock:
            if self._inflight is not None:
                raise RuntimeError("the previous decode batch has not been committed")
            while self._ready and len(self._running) < self.max_decode_sequences:
                request_id = self._ready.popleft()
                request = self._requests[request_id]
                request.transition(RequestState.RUNNING)
                self._running[request_id] = request
            selected = tuple(self._running.values())[: self.max_decode_sequences]
            if not selected:
                return DecodeBatch((), ())
            token_ids = tuple(
                request.generated_token_ids[-1]
                if request.generated_token_ids
                else request.token_ids[-1]
                for request in selected
            )
            grown: list[int] = []
            try:
                for request in selected:
                    self.block_manager.grow(request.request_id, 1)
                    grown.append(request.request_id)
            except Exception:
                # A full rollback would need shrinking allocations. Preempt the
                # requests already grown so capacity accounting remains correct.
                for request_id in grown:
                    request = self._running.pop(request_id)
                    request.transition(RequestState.PREEMPTED)
                    self.block_manager.free(request_id)
                    self._preempted.append(request_id)
                raise
            self._inflight = DecodeBatch(tuple(r.request_id for r in selected), token_ids)
            return self._inflight

    def commit_decode_tokens(self, batch: DecodeBatch, token_ids: tuple[int, ...]) -> None:
        if len(batch.request_ids) != len(token_ids):
            raise ValueError("decode output count does not match the scheduled batch")
        with self._lock:
            if batch != self._inflight:
                raise RuntimeError("decode result does not match the in-flight batch")
            for request_id, token_id in zip(batch.request_ids, token_ids, strict=True):
                request = self._running.get(request_id)
                if request is None:
                    raise RuntimeError(f"request {request_id} is no longer running")
                request.generated_token_ids.append(int(token_id))
                if len(request.generated_token_ids) >= request.max_new_tokens:
                    request.transition(RequestState.FINISHED)
                    self._running.pop(request_id)
                    self.block_manager.free(request_id)
            self._inflight = None

    def fail_decode_batch(self, batch: DecodeBatch) -> None:
        """Abort an in-flight iteration and release affected request storage."""
        with self._lock:
            if batch != self._inflight:
                raise RuntimeError("decode failure does not match the in-flight batch")
            for request_id in batch.request_ids:
                request = self._running.pop(request_id)
                request.transition(RequestState.FAILED)
                self.block_manager.free(request_id)
            self._inflight = None

    def preempt(self, request_id: int) -> None:
        """Release KV blocks; caller is responsible for recompute on resume."""
        with self._lock:
            if self._inflight is not None and request_id in self._inflight.request_ids:
                raise RuntimeError("cannot preempt a request while its decode batch is in flight")
            request = self._running.pop(request_id, None)
            if request is None:
                raise RuntimeError("only a running request can be preempted")
            request.transition(RequestState.PREEMPTED)
            self.block_manager.free(request_id)
            self._preempted.append(request_id)

    def resume(self, request_id: int) -> None:
        with self._lock:
            request = self._request(request_id)
            if request.state is not RequestState.PREEMPTED:
                raise RuntimeError("request is not preempted")
            required_tokens = len(request.token_ids) + len(request.generated_token_ids)
            self.block_manager.allocate(request_id, required_tokens)
            request.transition(RequestState.READY)
            try:
                self._preempted.remove(request_id)
            except ValueError:
                pass
            self._ready.append(request_id)

    def cancel(self, request_id: int) -> None:
        with self._lock:
            request = self._request(request_id)
            if request.state not in {RequestState.FINISHED, RequestState.FAILED}:
                request.transition(RequestState.FAILED)
            self._running.pop(request_id, None)
            self.block_manager.free(request_id)
            for queue in (self._waiting, self._ready, self._preempted):
                try:
                    queue.remove(request_id)
                except ValueError:
                    pass

    @property
    def active_count(self) -> int:
        return sum(
            request.state not in {RequestState.FINISHED, RequestState.FAILED}
            for request in self._requests.values()
        )

    def _request(self, request_id: int) -> Request:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request {request_id}") from exc
