"""Persistent continuous-batching generation loop.

The loop is transport/API agnostic. A runtime backend owns model state and
physical KV pages; the coordinator owns admission, streaming, cancellation,
and batch lifecycle.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import count
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Protocol


class OverloadedError(RuntimeError):
    """The bounded admission queue cannot accept more work."""


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    retryable: bool = False
    reason: str | None = None

    @classmethod
    def accept(cls) -> "AdmissionDecision":
        return cls(True)

    @classmethod
    def defer(cls, reason: str) -> "AdmissionDecision":
        return cls(False, retryable=True, reason=reason)

    @classmethod
    def reject(cls, reason: str) -> "AdmissionDecision":
        return cls(False, retryable=False, reason=reason)


@dataclass(frozen=True, slots=True)
class BackendCapacity:
    kv_total_blocks: int
    kv_free_blocks: int
    state_total_slots: int
    state_free_slots: int

    def __post_init__(self) -> None:
        if min(
            self.kv_total_blocks,
            self.kv_free_blocks,
            self.state_total_slots,
            self.state_free_slots,
        ) < 0:
            raise ValueError("capacity values cannot be negative")
        if self.kv_free_blocks > self.kv_total_blocks:
            raise ValueError("free KV blocks exceed total blocks")
        if self.state_free_slots > self.state_total_slots:
            raise ValueError("free state slots exceed total slots")

    @property
    def decode_load(self) -> float:
        kv_load = (
            1.0 - self.kv_free_blocks / self.kv_total_blocks
            if self.kv_total_blocks
            else 1.0
        )
        state_load = (
            1.0 - self.state_free_slots / self.state_total_slots
            if self.state_total_slots
            else 1.0
        )
        return max(kv_load, state_load)

    @property
    def has_request_slot(self) -> bool:
        return self.kv_free_blocks > 0 and self.state_free_slots > 0


@dataclass(slots=True)
class ServingRequest:
    request_id: int
    token_ids: tuple[int, ...]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    cancelled: Event = field(default_factory=Event)


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    request_id: int
    token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None


class GenerationBackend(Protocol):
    def prefill(self, request: ServingRequest) -> int: ...

    def decode(self, requests: tuple[ServingRequest, ...]) -> tuple[int, ...]: ...

    def release(self, request_id: int) -> None: ...


class GenerationHandle:
    """Thread-safe event stream returned to API and benchmark callers."""

    def __init__(self, request: ServingRequest, wake: Event) -> None:
        self.request = request
        self._wake = wake
        self._events: Queue[GenerationEvent] = Queue()

    @property
    def request_id(self) -> int:
        return self.request.request_id

    def cancel(self) -> None:
        self.request.cancelled.set()
        self._wake.set()

    def get(self, timeout: float | None = None) -> GenerationEvent:
        return self._events.get(timeout=timeout)

    def __iter__(self):
        while True:
            event = self.get()
            yield event
            if event.finished:
                return

    def _put(self, event: GenerationEvent) -> None:
        self._events.put(event)


class ContinuousGenerationLoop:
    """Long-lived prefill/decode coordinator with iteration-level admission."""

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        max_batch_size: int = 64,
        max_queue_size: int = 1024,
        max_queue_tokens: int = 1_048_576,
        eos_token_id: int | None = None,
        idle_wait_s: float = 0.01,
    ) -> None:
        if min(max_batch_size, max_queue_size, max_queue_tokens) <= 0 or idle_wait_s <= 0:
            raise ValueError("invalid serving-loop limits")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.max_queue_tokens = max_queue_tokens
        self.eos_token_id = eos_token_id
        self.idle_wait_s = idle_wait_s
        self._ids = count()
        self._incoming: Queue[tuple[ServingRequest, GenerationHandle]] = Queue()
        self._deferred: deque[tuple[ServingRequest, GenerationHandle]] = deque()
        self._pending_count = 0
        self._pending_tokens = 0
        self._pending_lock = Lock()
        self._handles: dict[int, GenerationHandle] = {}
        self._handles_lock = Lock()
        self._lifecycle_lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop.is_set():
                raise RuntimeError("a stopped serving loop cannot be restarted")
            self._thread = Thread(target=self._run, name="hydraserve-generation", daemon=True)
            self._thread.start()

    def submit(
        self, token_ids: list[int] | tuple[int, ...], max_new_tokens: int
    ) -> GenerationHandle:
        if not token_ids or max_new_tokens <= 0:
            raise ValueError("request needs a prompt and positive max_new_tokens")
        if self._stop.is_set():
            raise RuntimeError("serving loop is stopping")
        request = ServingRequest(next(self._ids), tuple(token_ids), max_new_tokens)
        handle = GenerationHandle(request, self._wake)
        demand = len(request.token_ids) + request.max_new_tokens
        with self._pending_lock:
            if self._pending_count >= self.max_queue_size:
                raise OverloadedError("admission queue request limit reached")
            if self._pending_tokens + demand > self.max_queue_tokens:
                raise OverloadedError("admission queue token limit reached")
            self._pending_count += 1
            self._pending_tokens += demand
        with self._handles_lock:
            self._handles[request.request_id] = handle
        self._incoming.put((request, handle))
        self._wake.set()
        self.start()
        return handle

    def cancel(self, request_id: int) -> None:
        with self._handles_lock:
            try:
                handle = self._handles[request_id]
            except KeyError as exc:
                raise KeyError(f"unknown or completed request {request_id}") from exc
        handle.cancel()

    def close(self, timeout: float | None = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("generation loop did not stop")
        close_backend = getattr(self.backend, "close", None)
        if close_backend is not None:
            close_backend()

    def _run(self) -> None:
        active: OrderedDict[int, ServingRequest] = OrderedDict()
        try:
            if getattr(self.backend, "supports_async_prefill", False):
                self._run_disaggregated(active)
            else:
                while not self._stop.is_set():
                    did_work = self._admit(active)
                    did_work = self._remove_cancelled(active) or did_work
                    if active:
                        self._decode_once(active)
                        did_work = True
                    if not did_work:
                        self._wake.wait(self.idle_wait_s)
                        self._wake.clear()
        finally:
            self._cancel_incoming()
            for request in tuple(active.values()):
                self._finish(request, "cancelled", active=active, release=True)

    def _run_disaggregated(
        self, active: OrderedDict[int, ServingRequest]
    ) -> None:
        pending: OrderedDict[int, tuple[ServingRequest, Future]] = OrderedDict()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydraserve-prefill") as executor:
            while not self._stop.is_set():
                did_work = self._submit_async_prefill(active, pending, executor)
                did_work = self._collect_async_prefill(active, pending) or did_work
                did_work = self._remove_cancelled(active) or did_work
                if active:
                    self._decode_once(active)
                    did_work = True
                if not did_work:
                    self._wake.wait(self.idle_wait_s)
                    self._wake.clear()
            for request, _ in pending.values():
                request.cancelled.set()
            while pending:
                self._collect_async_prefill(active, pending, wait=True)

    def _submit_async_prefill(self, active, pending, executor) -> bool:
        did_work = False
        available_slots = self.max_batch_size - len(active) - len(pending)
        for _ in range(available_slots):
            waiting = self._next_waiting()
            if waiting is None:
                break
            request, handle = waiting
            if request.cancelled.is_set():
                self._pending_done(request)
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            decision = self._admission_decision(request)
            if not decision.admitted:
                if decision.retryable:
                    self._deferred.appendleft((request, handle))
                    break
                self._pending_done(request)
                self._fail(
                    request,
                    MemoryError(decision.reason or "request cannot be admitted"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            self._pending_done(request)
            pending[request.request_id] = (
                request,
                executor.submit(self.backend.prefill, request),
            )
            did_work = True
        return did_work

    def _collect_async_prefill(self, active, pending, *, wait: bool = False) -> bool:
        completed = []
        for request_id, (_, future) in pending.items():
            if wait or future.done():
                completed.append(request_id)
                if wait:
                    # Resolve in admission order during shutdown.
                    break
        for request_id in completed:
            request, future = pending.pop(request_id)
            try:
                token_id = int(future.result())
                if request.cancelled.is_set() or self._stop.is_set():
                    self._finish(request, "cancelled", active=active, release=True)
                    continue
                request.generated_token_ids.append(token_id)
                self._emit(request, token_id)
                reason = self._finish_reason(request, token_id)
                if reason is not None:
                    self._finish(request, reason, active=active, release=True)
                else:
                    active[request.request_id] = request
            except Exception as exc:
                self._fail(request, exc, active=active, release=True)
        return bool(completed)

    def _admit(self, active: OrderedDict[int, ServingRequest]) -> bool:
        did_work = False
        available_slots = self.max_batch_size - len(active)
        for _ in range(available_slots):
            waiting = self._next_waiting()
            if waiting is None:
                return did_work
            request, handle = waiting
            if request.cancelled.is_set():
                self._pending_done(request)
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            decision = self._admission_decision(request)
            if not decision.admitted:
                if decision.retryable:
                    self._deferred.appendleft((request, handle))
                    return did_work
                self._pending_done(request)
                self._fail(
                    request,
                    MemoryError(decision.reason or "request cannot be admitted"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            self._pending_done(request)
            did_work = True
            try:
                token_id = int(self.backend.prefill(request))
            except Exception as exc:
                self._fail(request, exc, active=active, release=True)
                continue
            request.generated_token_ids.append(token_id)
            self._emit(request, token_id)
            reason = self._finish_reason(request, token_id)
            if reason is not None:
                self._finish(request, reason, active=active, release=True)
            else:
                active[request.request_id] = request
        return did_work

    def _decode_once(self, active: OrderedDict[int, ServingRequest]) -> None:
        batch = tuple(active.values())[: self.max_batch_size]
        try:
            token_ids = self.backend.decode(batch)
            if len(token_ids) != len(batch):
                raise RuntimeError("decode output count does not match the batch")
        except Exception as exc:
            for request in batch:
                self._fail(request, exc, active=active, release=True)
            return
        for request, token_id in zip(batch, token_ids, strict=True):
            token_id = int(token_id)
            request.generated_token_ids.append(token_id)
            self._emit(request, token_id)
            reason = self._finish_reason(request, token_id)
            if reason is not None:
                self._finish(request, reason, active=active, release=True)

    def _remove_cancelled(self, active: OrderedDict[int, ServingRequest]) -> bool:
        cancelled = [request for request in active.values() if request.cancelled.is_set()]
        for request in cancelled:
            self._finish(request, "cancelled", active=active, release=True)
        return bool(cancelled)

    def _finish_reason(self, request: ServingRequest, token_id: int) -> str | None:
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return "stop"
        if len(request.generated_token_ids) >= request.max_new_tokens:
            return "length"
        return None

    def _emit(self, request: ServingRequest, token_id: int) -> None:
        self._handle(request.request_id)._put(
            GenerationEvent(request.request_id, token_id=token_id)
        )

    def _finish(
        self,
        request: ServingRequest,
        reason: str,
        *,
        active: OrderedDict[int, ServingRequest],
        release: bool,
    ) -> None:
        active.pop(request.request_id, None)
        error = None
        if release:
            try:
                self.backend.release(request.request_id)
            except Exception as exc:
                error = f"release failed: {exc}"
        handle = self._pop_handle(request.request_id)
        if handle is not None:
            handle._put(
                GenerationEvent(
                    request.request_id,
                    finished=True,
                    finish_reason="error" if error else reason,
                    error=error,
                )
            )

    def _fail(
        self,
        request: ServingRequest,
        exc: Exception,
        *,
        active: OrderedDict[int, ServingRequest],
        release: bool,
    ) -> None:
        active.pop(request.request_id, None)
        if release:
            try:
                self.backend.release(request.request_id)
            except Exception:
                pass
        handle = self._pop_handle(request.request_id)
        if handle is not None:
            handle._put(
                GenerationEvent(
                    request.request_id,
                    finished=True,
                    finish_reason="error",
                    error=str(exc),
                )
            )

    def _cancel_incoming(self) -> None:
        empty: OrderedDict[int, ServingRequest] = OrderedDict()
        while self._deferred:
            request, _ = self._deferred.popleft()
            self._pending_done(request)
            self._finish(request, "cancelled", active=empty, release=False)
        while True:
            try:
                request, _ = self._incoming.get_nowait()
            except Empty:
                return
            self._pending_done(request)
            self._finish(request, "cancelled", active=empty, release=False)

    def _next_waiting(self):
        if self._deferred:
            return self._deferred.popleft()
        try:
            return self._incoming.get_nowait()
        except Empty:
            return None

    def _admission_decision(self, request: ServingRequest) -> AdmissionDecision:
        admit = getattr(self.backend, "admit", None)
        if admit is None:
            return AdmissionDecision.accept()
        try:
            decision = admit(request)
        except Exception as exc:
            return AdmissionDecision.reject(f"admission failed: {exc}")
        if isinstance(decision, AdmissionDecision):
            return decision
        if decision is True:
            return AdmissionDecision.accept()
        if decision is False:
            return AdmissionDecision.defer("backend capacity is temporarily exhausted")
        return AdmissionDecision.reject(
            "backend admit() must return AdmissionDecision or bool"
        )

    def _pending_done(self, request: ServingRequest) -> None:
        demand = len(request.token_ids) + request.max_new_tokens
        with self._pending_lock:
            self._pending_count -= 1
            self._pending_tokens -= demand
            if self._pending_count < 0 or self._pending_tokens < 0:
                raise RuntimeError("admission queue accounting underflow")

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return self._pending_count

    @property
    def pending_tokens(self) -> int:
        with self._pending_lock:
            return self._pending_tokens

    def _handle(self, request_id: int) -> GenerationHandle:
        with self._handles_lock:
            return self._handles[request_id]

    def _pop_handle(self, request_id: int) -> GenerationHandle | None:
        with self._handles_lock:
            return self._handles.pop(request_id, None)


class RuntimeGenerationBackend:
    """HydraServe runtime adapter; no external model-execution backend is used."""

    def __init__(
        self,
        runtime,
        paged_cache,
        *,
        prefill_chunk_size: int = 4096,
        max_state_slots: int = 64,
    ) -> None:
        if min(prefill_chunk_size, max_state_slots) <= 0:
            raise ValueError("prefill chunk size and state slots must be positive")
        from hydraserve.cache import RequestStateSlotManager

        self.runtime = runtime
        self.paged_cache = paged_cache
        self.prefill_chunk_size = prefill_chunk_size
        self.state_slots = RequestStateSlotManager(max_state_slots)
        self.states: dict[int, object] = {}
        self._admission_lock = Lock()

    @staticmethod
    def _total_kv_tokens(request: ServingRequest) -> int:
        # The first output token is sampled from prompt logits. Each remaining
        # output token requires appending the preceding generated token to KV.
        return len(request.token_ids) + max(0, request.max_new_tokens - 1)

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        manager = self.paged_cache.block_manager
        total_tokens = self._total_kv_tokens(request)
        required = manager.blocks_required(total_tokens)
        if required > manager.num_blocks:
            return AdmissionDecision.reject(
                f"request needs {required} KV blocks, worker capacity is {manager.num_blocks}"
            )
        with self._admission_lock:
            try:
                allocation = manager.get(request.request_id)
            except KeyError:
                allocation = None
            try:
                self.state_slots.get(request.request_id)
                owns_state_slot = True
            except KeyError:
                owns_state_slot = False
            if allocation is not None and owns_state_slot:
                if (
                    allocation.num_tokens != len(request.token_ids)
                    or allocation.reserved_tokens < total_tokens
                ):
                    return AdmissionDecision.reject(
                        "request id is already reserved with incompatible KV capacity"
                    )
                return AdmissionDecision.accept()
            if allocation is not None or owns_state_slot:
                manager.free(request.request_id)
                self.state_slots.free(request.request_id)
                self.states.pop(request.request_id, None)
            try:
                self.state_slots.allocate(request.request_id)
            except MemoryError:
                return AdmissionDecision.defer("recurrent-state slots are exhausted")
            try:
                self.paged_cache.allocate(
                    request.request_id,
                    len(request.token_ids),
                    reserve_tokens=total_tokens,
                )
            except MemoryError:
                self.state_slots.free(request.request_id)
                return AdmissionDecision.defer(
                    f"request needs {required} KV blocks, only {manager.num_free_blocks} are free"
                )
            except Exception:
                self.state_slots.free(request.request_id)
                raise
            return AdmissionDecision.accept()

    def prefill(self, request: ServingRequest) -> int:
        import torch

        try:
            self.paged_cache.block_manager.get(request.request_id)
        except KeyError:
            decision = self.admit(request)
            if not decision.admitted:
                raise MemoryError(decision.reason or "request cannot be admitted")
        try:
            input_ids = torch.tensor(
                [request.token_ids], device=self.runtime.device, dtype=torch.long
            )
            with torch.inference_mode():
                logits, state = self.runtime.prefill(
                    input_ids,
                    chunk_size=self.prefill_chunk_size,
                    paged_cache=self.paged_cache,
                    request_id=request.request_id,
                )
            self.states[request.request_id] = state
            return int(logits[0, -1].argmax())
        except Exception:
            self.release(request.request_id)
            raise

    def decode(self, requests: tuple[ServingRequest, ...]) -> tuple[int, ...]:
        import torch

        for request in requests:
            self.paged_cache.reserve_append(request.request_id)
        input_ids = torch.tensor(
            [request.generated_token_ids[-1] for request in requests],
            device=self.runtime.device,
            dtype=torch.long,
        ).unsqueeze(1)
        states = [self.states[request.request_id] for request in requests]
        with torch.inference_mode():
            logits, _ = self.runtime.decode_batch(
                input_ids,
                states,
                self.paged_cache,
                tuple(request.request_id for request in requests),
            )
        return tuple(int(token) for token in logits[:, -1].argmax(dim=-1).tolist())

    def release(self, request_id: int) -> None:
        with self._admission_lock:
            self.states.pop(request_id, None)
            self.paged_cache.free(request_id)
            self.state_slots.free(request_id)

    def capacity(self) -> BackendCapacity:
        kv = self.paged_cache.block_manager.capacity()
        state = self.state_slots.capacity()
        return BackendCapacity(
            kv_total_blocks=kv.total_blocks,
            kv_free_blocks=kv.free_blocks,
            state_total_slots=state.total_slots,
            state_free_slots=state.free_slots,
        )
