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
from time import monotonic
from typing import Protocol
from secrets import randbits

from hydraserve.engine.fair_scheduler import FairDecodeScheduler
from hydraserve.engine.sampling import SamplingParams, TokenSample


class OverloadedError(RuntimeError):
    """The bounded admission queue cannot accept more work."""


class PartialDecodeError(RuntimeError):
    """A decode iteration succeeded for some requests and failed for others."""

    def __init__(
        self,
        token_ids: dict[int, int | TokenSample],
        errors: dict[int, BaseException],
    ) -> None:
        overlap = set(token_ids) & set(errors)
        if overlap:
            raise ValueError(f"partial decode outcomes overlap for requests {sorted(overlap)}")
        if not errors:
            raise ValueError("partial decode error requires at least one failed request")
        self.samples = {
            int(key): value if isinstance(value, TokenSample) else TokenSample(int(value))
            for key, value in token_ids.items()
        }
        self.token_ids = {key: sample.token_id for key, sample in self.samples.items()}
        self.errors = {int(key): value for key, value in errors.items()}
        failed = ", ".join(str(request_id) for request_id in sorted(self.errors))
        super().__init__(f"decode failed for request(s): {failed}")


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
    route: str | None = None
    route_reason: str | None = None
    route_collocated_cost_ms: float | None = None
    route_pd_cost_ms: float | None = None
    route_estimated_savings_ms: float | None = None
    route_cost_confidence: float | None = None
    route_decode_load: float | None = None
    route_prefill_queue_ahead_ms: float = 0.0
    route_observed_prefill_service_ms: float | None = None
    prefill_started_at: float | None = None
    submitted_at: float = field(default_factory=monotonic)
    admitted_at: float | None = None
    admission_wait_ms: float | None = None
    observed_prefill_queue_wait_ms: float | None = None
    worker_id: int | None = None
    priority: int = 0
    admission_age: int = 0
    sampling_params: SamplingParams = SamplingParams()


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    request_id: int
    token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, float], ...] = ()


class GenerationBackend(Protocol):
    def prefill(self, request: ServingRequest) -> int | TokenSample: ...

    def decode(
        self, requests: tuple[ServingRequest, ...]
    ) -> tuple[int | TokenSample, ...]: ...

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
        max_active_requests: int | None = None,
        max_queue_size: int = 1024,
        max_queue_tokens: int = 1_048_576,
        eos_token_id: int | None = None,
        idle_wait_s: float = 0.01,
    ) -> None:
        active_limit = max_batch_size if max_active_requests is None else max_active_requests
        if (
            min(max_batch_size, active_limit, max_queue_size, max_queue_tokens) <= 0
            or idle_wait_s <= 0
            or active_limit < max_batch_size
        ):
            raise ValueError("invalid serving-loop limits")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_active_requests = active_limit
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
        self._decode_scheduler = FairDecodeScheduler()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop.is_set():
                raise RuntimeError("a stopped serving loop cannot be restarted")
            self._thread = Thread(target=self._run, name="hydraserve-generation", daemon=True)
            self._thread.start()

    def submit(
        self,
        token_ids: list[int] | tuple[int, ...],
        max_new_tokens: int,
        *,
        priority: int = 0,
        sampling_params: SamplingParams | None = None,
    ) -> GenerationHandle:
        if not token_ids or max_new_tokens <= 0:
            raise ValueError("request needs a prompt and positive max_new_tokens")
        if self._stop.is_set():
            raise RuntimeError("serving loop is stopping")
        if not 0 <= priority <= self._decode_scheduler.config.max_priority:
            raise ValueError(
                f"priority must be in [0, {self._decode_scheduler.config.max_priority}]"
            )
        sampling = sampling_params or SamplingParams()
        if sampling.seed is None:
            sampling = sampling.with_seed(randbits(63))
        request = ServingRequest(
            next(self._ids),
            tuple(token_ids),
            max_new_tokens,
            priority=priority,
            sampling_params=sampling,
        )
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
        available_slots = self.max_active_requests - len(active) - len(pending)
        candidates = self._waiting_candidates()
        for index, waiting in enumerate(candidates):
            if available_slots <= 0:
                self._defer_remaining(candidates[index:])
                break
            request, handle = waiting
            if request.cancelled.is_set():
                self._pending_done(request)
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            request.route_prefill_queue_ahead_ms = self._prefill_queue_ahead_ms(
                pending
            )
            decision = self._admission_decision(request)
            if not decision.admitted:
                if decision.retryable:
                    request.admission_age += 1
                    self._deferred.append((request, handle))
                    continue
                self._pending_done(request)
                self._fail(
                    request,
                    MemoryError(decision.reason or "request cannot be admitted"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            self._record_admission(request)
            self._pending_done(request)
            pending[request.request_id] = (
                request,
                executor.submit(self._execute_prefill, request),
            )
            available_slots -= 1
            did_work = True
        return did_work

    @staticmethod
    def _prefill_queue_ahead_ms(pending) -> float:
        now = monotonic()
        total = 0.0
        for request, future in pending.values():
            if future.done():
                continue
            if request.route == "pd_disaggregated":
                predicted = request.route_pd_cost_ms
            else:
                predicted = request.route_collocated_cost_ms
            if predicted is not None:
                service_ms = max(
                    0.0,
                    predicted - request.route_prefill_queue_ahead_ms,
                )
                if future.running() and request.prefill_started_at is not None:
                    service_ms = max(
                        0.0,
                        service_ms - (now - request.prefill_started_at) * 1000.0,
                    )
                total += service_ms
        return total

    def _execute_prefill(self, request: ServingRequest):
        request.prefill_started_at = monotonic()
        if request.admitted_at is not None:
            request.observed_prefill_queue_wait_ms = max(
                0.0, (request.prefill_started_at - request.admitted_at) * 1000.0
            )
        return self.backend.prefill(request)

    @staticmethod
    def _record_admission(request: ServingRequest) -> None:
        if request.admission_wait_ms is None:
            request.admitted_at = monotonic()
            request.admission_wait_ms = max(
                0.0, (request.admitted_at - request.submitted_at) * 1000.0
            )

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
                sample = self._normalize_sample(future.result())
                if request.cancelled.is_set() or self._stop.is_set():
                    self._finish(request, "cancelled", active=active, release=True)
                    continue
                request.generated_token_ids.append(sample.token_id)
                self._emit(request, sample)
                reason = self._finish_reason(request, sample.token_id)
                if reason is not None:
                    self._finish(request, reason, active=active, release=True)
                else:
                    active[request.request_id] = request
            except Exception as exc:
                self._fail(request, exc, active=active, release=True)
        return bool(completed)

    def _admit(self, active: OrderedDict[int, ServingRequest]) -> bool:
        did_work = False
        available_slots = self.max_active_requests - len(active)
        candidates = self._waiting_candidates()
        for index, waiting in enumerate(candidates):
            if available_slots <= 0:
                self._defer_remaining(candidates[index:])
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
                    request.admission_age += 1
                    self._deferred.append((request, handle))
                    continue
                self._pending_done(request)
                self._fail(
                    request,
                    MemoryError(decision.reason or "request cannot be admitted"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            self._record_admission(request)
            self._pending_done(request)
            available_slots -= 1
            did_work = True
            try:
                sample = self._normalize_sample(self.backend.prefill(request))
            except Exception as exc:
                self._fail(request, exc, active=active, release=True)
                continue
            request.generated_token_ids.append(sample.token_id)
            self._emit(request, sample)
            reason = self._finish_reason(request, sample.token_id)
            if reason is not None:
                self._finish(request, reason, active=active, release=True)
            else:
                active[request.request_id] = request
        return did_work

    def _decode_once(self, active: OrderedDict[int, ServingRequest]) -> None:
        batch = self._decode_scheduler.select(active.values(), self.max_batch_size)
        try:
            token_ids = self.backend.decode(batch)
            if len(token_ids) != len(batch):
                raise RuntimeError("decode output count does not match the batch")
        except PartialDecodeError as exc:
            expected = {request.request_id for request in batch}
            actual = set(exc.token_ids) | set(exc.errors)
            if actual != expected:
                malformed = RuntimeError(
                    "partial decode outcome does not cover the scheduled batch"
                )
                for request in batch:
                    self._fail(request, malformed, active=active, release=True)
                return
            for request in batch:
                error = exc.errors.get(request.request_id)
                if error is not None:
                    self._fail(request, error, active=active, release=True)
                else:
                    self._accept_decode_sample(
                        request, exc.samples[request.request_id], active
                    )
            return
        except Exception as exc:
            for request in batch:
                self._fail(request, exc, active=active, release=True)
            return
        for request, token_id in zip(batch, token_ids, strict=True):
            self._accept_decode_sample(request, token_id, active)

    def _accept_decode_sample(
        self,
        request: ServingRequest,
        value: int | TokenSample,
        active: OrderedDict[int, ServingRequest],
    ) -> None:
        sample = self._normalize_sample(value)
        request.generated_token_ids.append(sample.token_id)
        self._emit(request, sample)
        reason = self._finish_reason(request, sample.token_id)
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
        generated = request.generated_token_ids
        if any(
            len(generated) >= len(sequence)
            and tuple(generated[-len(sequence) :]) == sequence
            for sequence in request.sampling_params.stop_token_sequences
        ):
            return "stop"
        if len(request.generated_token_ids) >= request.max_new_tokens:
            return "length"
        return None

    def _emit(self, request: ServingRequest, sample: TokenSample) -> None:
        self._handle(request.request_id)._put(
            GenerationEvent(
                request.request_id,
                token_id=sample.token_id,
                logprob=sample.logprob,
                top_logprobs=sample.top_logprobs,
            )
        )

    @staticmethod
    def _normalize_sample(value: int | TokenSample) -> TokenSample:
        return value if isinstance(value, TokenSample) else TokenSample(int(value))

    def _finish(
        self,
        request: ServingRequest,
        reason: str,
        *,
        active: OrderedDict[int, ServingRequest],
        release: bool,
    ) -> None:
        active.pop(request.request_id, None)
        self._decode_scheduler.forget(request.request_id)
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
        self._decode_scheduler.forget(request.request_id)
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

    def _waiting_candidates(self):
        candidates = list(self._deferred)
        self._deferred.clear()
        while True:
            try:
                candidates.append(self._incoming.get_nowait())
            except Empty:
                break
        candidates.sort(
            key=lambda item: (
                -(item[0].priority * 8 + item[0].admission_age),
                item[0].request_id,
            )
        )
        return candidates

    def _defer_remaining(self, waiting) -> None:
        for request, handle in waiting:
            request.admission_age += 1
            self._deferred.append((request, handle))

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
        from hydraserve.cache import GpuLinearStatePool, RequestStateSlotManager

        self.runtime = runtime
        self.paged_cache = paged_cache
        self.prefill_chunk_size = prefill_chunk_size
        config = getattr(runtime, "config", None)
        self.state_pool = (
            GpuLinearStatePool(max_state_slots, config, device=runtime.device)
            if config is not None and config.linear_layer_indices
            else None
        )
        self.state_slots = (
            self.state_pool.slots
            if self.state_pool is not None
            else RequestStateSlotManager(max_state_slots)
        )
        self.states: dict[int, object] = {}
        self._admission_lock = Lock()

    @staticmethod
    def _total_kv_tokens(request: ServingRequest) -> int:
        # The first output token is sampled from prompt logits. Each remaining
        # output token requires appending the preceding generated token to KV.
        return len(request.token_ids) + max(0, request.max_new_tokens - 1)

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        manager = self.paged_cache.block_manager
        initial_capacity = self.capacity()
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
                if self.state_pool is not None:
                    self.state_pool.free(request.request_id)
                else:
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
                    token_ids=request.token_ids,
                )
            except MemoryError:
                self.state_slots.free(request.request_id)
                return AdmissionDecision.defer(
                    f"request needs {required} KV blocks, only {manager.num_free_blocks} are free"
                )
            except Exception:
                self.state_slots.free(request.request_id)
                raise
            if request.route is None:
                request.route = "collocated"
                request.route_reason = "fixed_collocated"
                request.worker_id = 0
                request.route_decode_load = initial_capacity.decode_load
            return AdmissionDecision.accept()

    def prefill(self, request: ServingRequest) -> TokenSample:
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
            self.paged_cache.publish_prefix(request.request_id, request.token_ids)
            if self.state_pool is not None:
                state = self.state_pool.install(request.request_id, state)
            self.states[request.request_id] = state
            from hydraserve.engine.sampling import sample_logits

            return sample_logits(
                logits[:, -1],
                (request.token_ids,),
                (request.sampling_params,),
                steps=(0,),
            )[0]
        except Exception:
            self.release(request.request_id)
            raise

    def decode(self, requests: tuple[ServingRequest, ...]) -> tuple[TokenSample, ...]:
        """Decode transactionally, bisecting a failed batch to isolate requests."""
        import torch

        if not requests:
            return ()
        request_by_id = {request.request_id: request for request in requests}
        if len(request_by_id) != len(requests):
            raise ValueError("decode requests must have unique request ids")
        manager = self.paged_cache.block_manager
        base_lengths = {
            request.request_id: manager.get(request.request_id).num_tokens
            for request in requests
        }
        checkpoints = {
            request.request_id: self._checkpoint_state(self.states[request.request_id])
            for request in requests
        }

        def restore(request_ids: tuple[int, ...]) -> None:
            for request_id in request_ids:
                manager.truncate(request_id, base_lengths[request_id])
                self.states[request_id] = self._checkpoint_state(
                    checkpoints[request_id]
                )

        def attempt(request_ids: tuple[int, ...]) -> tuple[TokenSample, ...]:
            manager.grow_many(request_ids, additional_tokens=1)
            input_ids = torch.tensor(
                [request_by_id[item].generated_token_ids[-1] for item in request_ids],
                device=self.runtime.device,
                dtype=torch.long,
            ).unsqueeze(1)
            states = [self.states[item] for item in request_ids]
            try:
                with torch.inference_mode():
                    logits, _ = self.runtime.decode_batch(
                        input_ids,
                        states,
                        self.paged_cache,
                        request_ids,
                    )
                from hydraserve.engine.sampling import sample_logits

                token_ids = sample_logits(
                    logits[:, -1],
                    (
                        request_by_id[item].token_ids
                        + tuple(request_by_id[item].generated_token_ids)
                        for item in request_ids
                    ),
                    (request_by_id[item].sampling_params for item in request_ids),
                    steps=(
                        len(request_by_id[item].generated_token_ids)
                        for item in request_ids
                    ),
                )
                if len(token_ids) != len(request_ids):
                    raise RuntimeError(
                        "runtime decode output count does not match its request group"
                    )
                return token_ids
            except Exception:
                restore(request_ids)
                raise

        all_ids = tuple(request_by_id)
        try:
            return attempt(all_ids)
        except Exception:
            pass

        successes: dict[int, TokenSample] = {}
        failures: dict[int, Exception] = {}

        def isolate(request_ids: tuple[int, ...]) -> None:
            try:
                token_ids = attempt(request_ids)
            except Exception as exc:
                if len(request_ids) == 1:
                    failures[request_ids[0]] = exc
                    return
                middle = len(request_ids) // 2
                isolate(request_ids[:middle])
                isolate(request_ids[middle:])
                return
            successes.update(zip(request_ids, token_ids, strict=True))

        isolate(all_ids)
        if failures:
            raise PartialDecodeError(successes, failures)
        return tuple(successes[request.request_id] for request in requests)

    @staticmethod
    def _checkpoint_state(state):
        """Take a cheap decode rollback point without copying immutable tensors."""
        from copy import copy

        checkpoint = copy(state)
        for name in ("recurrent", "convolution", "keys", "values"):
            values = getattr(state, name, None)
            if isinstance(values, dict):
                setattr(checkpoint, name, values.copy())
        return checkpoint

    def release(self, request_id: int) -> None:
        with self._admission_lock:
            self.states.pop(request_id, None)
            self.paged_cache.free(request_id)
            if self.state_pool is not None:
                self.state_pool.free(request_id)
            else:
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
