"""Persistent continuous-batching generation loop.

The loop is transport/API agnostic. A runtime backend owns model state and
physical KV pages; the coordinator owns admission, streaming, cancellation,
and batch lifecycle.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Protocol


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
        eos_token_id: int | None = None,
        idle_wait_s: float = 0.01,
    ) -> None:
        if max_batch_size <= 0 or idle_wait_s <= 0:
            raise ValueError("invalid serving-loop limits")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id
        self.idle_wait_s = idle_wait_s
        self._ids = count()
        self._incoming: Queue[tuple[ServingRequest, GenerationHandle]] = Queue()
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

    def _admit(self, active: OrderedDict[int, ServingRequest]) -> bool:
        did_work = False
        available_slots = self.max_batch_size - len(active)
        for _ in range(available_slots):
            try:
                request, _ = self._incoming.get_nowait()
            except Empty:
                return did_work
            did_work = True
            if request.cancelled.is_set():
                self._finish(request, "cancelled", active=active, release=False)
                continue
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
        while True:
            try:
                request, _ = self._incoming.get_nowait()
            except Empty:
                return
            self._finish(request, "cancelled", active=empty, release=False)

    def _handle(self, request_id: int) -> GenerationHandle:
        with self._handles_lock:
            return self._handles[request_id]

    def _pop_handle(self, request_id: int) -> GenerationHandle | None:
        with self._handles_lock:
            return self._handles.pop(request_id, None)


class RuntimeGenerationBackend:
    """HydraServe runtime adapter; no external model-execution backend is used."""

    def __init__(self, runtime, paged_cache, *, prefill_chunk_size: int = 4096) -> None:
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        self.runtime = runtime
        self.paged_cache = paged_cache
        self.prefill_chunk_size = prefill_chunk_size
        self.states: dict[int, object] = {}

    def prefill(self, request: ServingRequest) -> int:
        import torch

        self.paged_cache.allocate(request.request_id, len(request.token_ids))
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
            self.paged_cache.free(request.request_id)
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
        self.states.pop(request_id, None)
        self.paged_cache.free(request_id)
