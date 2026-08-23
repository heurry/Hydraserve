"""Persistent continuous-batching generation loop.

The loop is transport/API agnostic. A runtime backend owns model state and
physical KV pages; the coordinator owns admission, streaming, cancellation,
and batch lifecycle.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
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
            raise ValueError(
                f"partial decode outcomes overlap for requests {sorted(overlap)}"
            )
        if not errors:
            raise ValueError(
                "partial decode error requires at least one failed request"
            )
        self.samples = {
            int(key): (
                value if isinstance(value, TokenSample) else TokenSample(int(value))
            )
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
        if (
            min(
                self.kv_total_blocks,
                self.kv_free_blocks,
                self.state_total_slots,
                self.state_free_slots,
            )
            < 0
        ):
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
    ignore_eos: bool = False
    generated_token_ids: list[int] = field(default_factory=list)
    cancelled: Event = field(default_factory=Event)
    route: str | None = None
    route_reason: str | None = None
    route_collocated_cost_ms: float | None = None
    route_pd_cost_ms: float | None = None
    route_estimated_savings_ms: float | None = None
    route_cost_confidence: float | None = None
    route_decode_load: float | None = None
    route_prefill_load: float = 0.0
    route_prefill_queue_ahead_ms: float = 0.0
    route_observed_prefill_service_ms: float | None = None
    prefill_started_at: float | None = None
    submitted_at: float = field(default_factory=monotonic)
    admitted_at: float | None = None
    admission_wait_ms: float | None = None
    observed_prefill_queue_wait_ms: float | None = None
    deadline_at: float | None = None
    worker_id: int | None = None
    worker_pool: str | None = None
    priority: int = 0
    admission_age: int = 0
    preemption_count: int = 0
    recovery_count: int = 0
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
    # Present for decode tokens so benchmarks can distinguish sparse-batch
    # latency from transport/release overhead. Prefill seed tokens use ``None``.
    decode_batch_size: int | None = None


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
        max_preemptions_per_request: int = 2,
        max_step_tokens: int | None = None,
        dp_graph_sync: bool = False,
        dp_process_group=None,
    ) -> None:
        active_limit = (
            max_batch_size if max_active_requests is None else max_active_requests
        )
        if (
            min(max_batch_size, active_limit, max_queue_size, max_queue_tokens) <= 0
            or idle_wait_s <= 0
            or active_limit < max_batch_size
            or max_preemptions_per_request < 0
            or (max_step_tokens is not None and max_step_tokens <= 0)
        ):
            raise ValueError("invalid serving-loop limits")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_active_requests = active_limit
        self.max_queue_size = max_queue_size
        self.max_queue_tokens = max_queue_tokens
        self.eos_token_id = eos_token_id
        self.idle_wait_s = idle_wait_s
        self.max_preemptions_per_request = max_preemptions_per_request
        self.max_step_tokens = (
            max_queue_tokens if max_step_tokens is None else max_step_tokens
        )
        self.dp_graph_sync = dp_graph_sync
        self.dp_process_group = dp_process_group
        self._ids = count()
        self._incoming: Queue[tuple[ServingRequest, GenerationHandle]] = Queue()
        self._deferred: deque[tuple[ServingRequest, GenerationHandle]] = deque()
        self._preempted: deque[ServingRequest] = deque()
        self._pending_count = 0
        self._pending_tokens = 0
        self._pending_lock = Lock()
        self._stats_lock = Lock()
        self._active_count = 0
        self._prefill_pending_count = 0
        self._preempted_count = 0
        self._preemptions_total = 0
        self._preemption_failures_total = 0
        self._recoveries_total = 0
        self._recovery_failures_total = 0
        self._fault_suspensions_total = 0
        self._prefill_slot_deferrals_total = 0
        release_parallelism = max(
            1, int(getattr(backend, "release_parallelism", min(4, active_limit)))
        )
        self._release_executor = ThreadPoolExecutor(
            max_workers=release_parallelism,
            thread_name_prefix="hydraserve-release",
        )
        self._release_pending_count = 0
        self._release_total = 0
        self._release_failures_total = 0
        self._handles: dict[int, GenerationHandle] = {}
        self._handles_lock = Lock()
        self._lifecycle_lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self._decode_scheduler = FairDecodeScheduler()
        self._decode_inflight_ids: set[int] = set()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop.is_set():
                raise RuntimeError("a stopped serving loop cannot be restarted")
            self._thread = Thread(
                target=self._run, name="hydraserve-generation", daemon=True
            )
            self._thread.start()

    def submit(
        self,
        token_ids: list[int] | tuple[int, ...],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        priority: int = 0,
        sampling_params: SamplingParams | None = None,
        timeout_ms: float | None = None,
    ) -> GenerationHandle:
        if not token_ids or max_new_tokens <= 0:
            raise ValueError("request needs a prompt and positive max_new_tokens")
        if self._stop.is_set():
            raise RuntimeError("serving loop is stopping")
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
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
            ignore_eos=ignore_eos,
            priority=priority,
            sampling_params=sampling,
            deadline_at=(
                None if timeout_ms is None else monotonic() + timeout_ms / 1000.0
            ),
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
        # The generation thread may have queued device-local cleanup while
        # resolving its final active requests. Drain it before closing workers.
        self._release_executor.shutdown(wait=True)
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
                    did_work = self._admit(
                        active,
                        token_budget=max(0, self.max_step_tokens - len(active)),
                    )
                    did_work = self._remove_cancelled_or_expired(active) or did_work
                    self._publish_scheduler_depth(len(active), 0)
                    if active or self.dp_graph_sync:
                        self._decode_once(active)
                        did_work = True
                    if not did_work:
                        self._wake.wait(self.idle_wait_s)
                        self._wake.clear()
        finally:
            self._cancel_incoming()
            for request in tuple(active.values()):
                self._finish(request, "cancelled", active=active, release=True)
            self._publish_scheduler_depth(0, 0)

    def _run_disaggregated(self, active: OrderedDict[int, ServingRequest]) -> None:
        pending: OrderedDict[int, tuple[ServingRequest, Future]] = OrderedDict()
        recovering: set[int] = set()
        raw_limits = getattr(self.backend, "prefill_executor_limits", None)
        if raw_limits is None:
            raw_limits = {
                "default": max(1, int(getattr(self.backend, "prefill_parallelism", 1)))
            }
        executor_limits = {
            str(group): max(1, int(limit)) for group, limit in dict(raw_limits).items()
        }
        if not executor_limits:
            executor_limits = {"default": 1}
        group_for = getattr(self.backend, "prefill_executor_group", None)
        group_hint_for = getattr(self.backend, "prefill_executor_group_hint", None)

        with ExitStack() as stack:
            executors = {
                group: stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=limit,
                        thread_name_prefix=f"hydraserve-prefill-{group}",
                    )
                )
                for group, limit in executor_limits.items()
            }
            default_executor = next(iter(executors.values()))
            independent_decode = (
                bool(getattr(self.backend, "supports_independent_decode", False))
                and not self.dp_graph_sync
            )
            # Keep the original synchronous ``_decode_once`` path below for
            # simple/legacy backends. Multi-worker Hybrid PD opts into this
            # executor so a P-side chunk boundary never becomes a barrier for
            # unrelated permanent-D workers.
            decode_executor = (
                stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=max(
                            1,
                            int(
                                getattr(
                                    self.backend,
                                    "decode_executor_parallelism",
                                    self.max_batch_size,
                                )
                            ),
                        ),
                        thread_name_prefix="hydraserve-decode-worker",
                    )
                )
                if independent_decode
                else None
            )
            pending_decode: dict[object, tuple[tuple[ServingRequest, ...], Future]] = {}

            def executor_for(request: ServingRequest) -> ThreadPoolExecutor:
                if not callable(group_for):
                    return default_executor
                return executors.get(str(group_for(request)), default_executor)

            def can_submit_prefill(request: ServingRequest) -> bool:
                """Bound admission by the physical pool, not its host queue.

                In particular, do not reserve decode-side KV for every queued
                long request while only one or two P workers can make progress.
                Backends return ``None`` when routing is not deterministic yet.
                """

                hinted = group_hint_for(request) if callable(group_hint_for) else None
                if hinted is None:
                    # A backend with one executor pool has an unambiguous
                    # physical target even without a route hint (for example
                    # N-way collocated DP). Do not let its executor's unbounded
                    # host queue reserve KV for work that cannot run yet.
                    if len(executor_limits) != 1:
                        return True
                    group = next(iter(executor_limits))
                else:
                    group = str(hinted)
                limit = executor_limits.get(group)
                if limit is None:
                    return True
                occupied = 0
                for pending_request, future in pending.values():
                    if future.done():
                        continue
                    pending_group = (
                        str(group_for(pending_request))
                        if callable(group_for)
                        else next(iter(executor_limits))
                    )
                    if pending_group == group:
                        occupied += 1
                return occupied < limit

            prefill_cost_for = getattr(self.backend, "prefill_admission_tokens", None)

            def admission_tokens(request: ServingRequest) -> int:
                if callable(prefill_cost_for):
                    return min(
                        self.max_step_tokens,
                        max(1, int(prefill_cost_for(request))),
                    )
                # Async execution owns a physical slot until the complete RPC
                # returns. Charge at most one scheduling quantum here; charging
                # the whole prompt makes 32K requests impossible whenever a
                # decode request is active, despite the independent executor.
                return min(len(request.token_ids), self.max_step_tokens)

            while not self._stop.is_set():
                # Async prefill runs in independently bounded physical pools.
                # Its admission budget is therefore separate from the active
                # decode width; worker-local serialization/preemption handles
                # actual same-GPU contention.
                prefill_budget = self.max_step_tokens
                did_work = self._submit_async_prefill(
                    active,
                    pending,
                    executor_for,
                    can_submit_prefill=can_submit_prefill,
                    admission_tokens=admission_tokens,
                    token_budget=prefill_budget,
                )
                did_work = self._remove_cancelled_or_expired(active) or did_work
                did_work = (
                    self._submit_async_recovery(
                        active, pending, recovering, executor_for
                    )
                    or did_work
                )
                self._publish_scheduler_depth(len(active), len(pending))
                if independent_decode:
                    did_work = (
                        self._collect_independent_decode(active, pending_decode)
                        or did_work
                    )
                    did_work = (
                        self._submit_independent_decode(
                            active, pending_decode, decode_executor
                        )
                        or did_work
                    )
                elif active or self.dp_graph_sync:
                    self._decode_once(active)
                    did_work = True
                # Resolve CPU futures after launching the active decode step;
                # this keeps response handling off the decode critical path.
                did_work = (
                    self._collect_async_prefill(active, pending, recovering=recovering)
                    or did_work
                )
                if not did_work:
                    self._wake.wait(self.idle_wait_s)
                    self._wake.clear()
            while pending_decode:
                self._collect_independent_decode(active, pending_decode, wait=True)
            for request, _ in pending.values():
                request.cancelled.set()
            while pending:
                self._collect_async_prefill(
                    active, pending, recovering=recovering, wait=True
                )

    def _submit_async_prefill(
        self,
        active,
        pending,
        executor_for,
        *,
        can_submit_prefill=None,
        admission_tokens=None,
        token_budget: int | None = None,
    ) -> bool:
        did_work = False
        available_slots = self.max_active_requests - len(active) - len(pending)
        candidates = self._waiting_candidates()
        for index, waiting in enumerate(candidates):
            if available_slots <= 0:
                if self._try_preempt_for(waiting[0], active):
                    available_slots += 1
                else:
                    self._defer_remaining(candidates[index:])
                    break
            request, handle = waiting
            demand = (
                max(1, int(admission_tokens(request)))
                if callable(admission_tokens)
                else len(request.token_ids)
            )
            if token_budget is not None and demand > token_budget:
                request.admission_age += 1
                self._deferred.append((request, handle))
                continue
            if request.cancelled.is_set():
                self._pending_done(request)
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            if self._deadline_expired(request):
                self._pending_done(request)
                self._fail(
                    request,
                    TimeoutError("request deadline expired before admission"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            if callable(can_submit_prefill) and not can_submit_prefill(request):
                request.admission_age += 1
                self._deferred.append((request, handle))
                with self._stats_lock:
                    self._prefill_slot_deferrals_total += 1
                continue
            request.route_prefill_queue_ahead_ms = self._prefill_queue_ahead_ms(pending)
            decision = self._admission_decision(request)
            if not decision.admitted:
                if decision.retryable and self._try_preempt_for(request, active):
                    available_slots += 1
                    decision = self._admission_decision(request)
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
            if token_budget is not None:
                token_budget = max(0, token_budget - demand)
            self._record_admission(request)
            self._pending_done(request)
            pending[request.request_id] = (
                request,
                executor_for(request).submit(self._execute_prefill, request),
            )
            available_slots -= 1
            did_work = True
        return did_work

    def _submit_async_recovery(
        self, active, pending, recovering: set[int], executor_for
    ) -> bool:
        recover = getattr(self.backend, "recover", None)
        if not callable(recover) or not self._preempted:
            return False
        did_work = False
        visits = len(self._preempted)
        for _ in range(visits):
            request = self._preempted.popleft()
            if request.cancelled.is_set():
                self._finish(request, "cancelled", active=active, release=False)
                with self._stats_lock:
                    self._preempted_count -= 1
                did_work = True
                continue
            if self._deadline_expired(request):
                self._fail(
                    request,
                    TimeoutError("request deadline expired while preempted"),
                    active=active,
                    release=False,
                )
                with self._stats_lock:
                    self._preempted_count -= 1
                did_work = True
                continue
            if len(active) + len(pending) >= self.max_active_requests:
                self._preempted.append(request)
                continue
            pending[request.request_id] = (
                request,
                executor_for(request).submit(recover, request),
            )
            recovering.add(request.request_id)
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
        # Backends without an explicit router are collocated, but assign that
        # label only after admission succeeds. Rejected requests must remain
        # ``unknown`` instead of being misreported as collocated failures.
        if request.route is None:
            request.route = "collocated"
            request.route_reason = request.route_reason or "implicit_collocated"
        if request.admission_wait_ms is None:
            request.admitted_at = monotonic()
            request.admission_wait_ms = max(
                0.0, (request.admitted_at - request.submitted_at) * 1000.0
            )

    def _collect_async_prefill(
        self,
        active,
        pending,
        *,
        recovering: set[int] | None = None,
        wait: bool = False,
    ) -> bool:
        recovering = set() if recovering is None else recovering
        completed = []
        for request_id, (_, future) in pending.items():
            if wait or future.done():
                completed.append(request_id)
                if wait:
                    # Resolve in admission order during shutdown.
                    break
        for request_id in completed:
            request, future = pending.pop(request_id)
            is_recovery = request_id in recovering
            recovering.discard(request_id)
            try:
                result = future.result()
                if is_recovery:
                    if request.cancelled.is_set() or self._stop.is_set():
                        self._finish(request, "cancelled", active=active, release=True)
                    elif self._deadline_expired(request):
                        self._fail(
                            request,
                            TimeoutError("request deadline expired during recovery"),
                            active=active,
                            release=True,
                        )
                    elif isinstance(result, AdmissionDecision) and not result.admitted:
                        if result.retryable:
                            self._preempted.append(request)
                            continue
                        raise MemoryError(
                            result.reason or "preempted request cannot recover"
                        )
                    else:
                        request.recovery_count += 1
                        active[request.request_id] = request
                        with self._stats_lock:
                            self._recoveries_total += 1
                    with self._stats_lock:
                        self._preempted_count -= 1
                    continue
                sample = self._normalize_sample(result)
                if request.cancelled.is_set() or self._stop.is_set():
                    self._finish(request, "cancelled", active=active, release=True)
                    continue
                if self._deadline_expired(request):
                    self._fail(
                        request,
                        TimeoutError("request deadline expired during prefill"),
                        active=active,
                        release=True,
                    )
                    continue
                request.generated_token_ids.append(sample.token_id)
                self._emit(request, sample)
                reason = self._finish_reason(request, sample.token_id)
                if reason is not None:
                    self._finish(request, reason, active=active, release=True)
                else:
                    active[request.request_id] = request
            except Exception as exc:
                if is_recovery:
                    with self._stats_lock:
                        self._recovery_failures_total += 1
                        self._preempted_count -= 1
                self._fail(request, exc, active=active, release=True)
        return bool(completed)

    def _admit(
        self,
        active: OrderedDict[int, ServingRequest],
        *,
        token_budget: int | None = None,
    ) -> bool:
        did_work = False
        available_slots = self.max_active_requests - len(active)
        candidates = self._waiting_candidates()
        for index, waiting in enumerate(candidates):
            if available_slots <= 0:
                if self._try_preempt_for(waiting[0], active):
                    available_slots += 1
                else:
                    self._defer_remaining(candidates[index:])
                    break
            request, handle = waiting
            demand = len(request.token_ids)
            if token_budget is not None and demand > token_budget:
                if active or token_budget != self.max_step_tokens:
                    self._defer_remaining(candidates[index:])
                    break
            if token_budget is not None:
                token_budget = max(0, token_budget - demand)
            if request.cancelled.is_set():
                self._pending_done(request)
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            if self._deadline_expired(request):
                self._pending_done(request)
                self._fail(
                    request,
                    TimeoutError("request deadline expired before admission"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            decision = self._admission_decision(request)
            if not decision.admitted:
                if decision.retryable and self._try_preempt_for(request, active):
                    available_slots += 1
                    decision = self._admission_decision(request)
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
            if self._deadline_expired(request):
                self._fail(
                    request,
                    TimeoutError("request deadline expired during prefill"),
                    active=active,
                    release=True,
                )
                continue
            request.generated_token_ids.append(sample.token_id)
            self._emit(request, sample)
            reason = self._finish_reason(request, sample.token_id)
            if reason is not None:
                self._finish(request, reason, active=active, release=True)
            else:
                active[request.request_id] = request
        return self._recover_preempted(active) or did_work

    def _try_preempt_for(
        self,
        candidate: ServingRequest,
        active: OrderedDict[int, ServingRequest],
    ) -> bool:
        """Evict one strictly less urgent request at an iteration boundary."""
        preempt = getattr(self.backend, "preempt", None)
        recover = getattr(self.backend, "recover", None)
        if (
            self.max_preemptions_per_request == 0
            or not callable(preempt)
            or not callable(recover)
        ):
            return False
        victims = [
            request
            for request in active.values()
            if request.request_id not in self._decode_inflight_ids
            if request.preemption_count < self.max_preemptions_per_request
            and self._strictly_more_urgent(candidate, request)
        ]
        if not victims:
            return False
        victim = min(victims, key=self._preemption_victim_key)
        try:
            preempt(victim.request_id)
        except Exception as exc:
            with self._stats_lock:
                self._preemption_failures_total += 1
            self._fail(
                victim,
                RuntimeError(f"preemption failed: {exc}"),
                active=active,
                release=True,
            )
            return True
        active.pop(victim.request_id, None)
        self._decode_scheduler.forget(victim.request_id)
        victim.preemption_count += 1
        self._preempted.append(victim)
        with self._stats_lock:
            self._preemptions_total += 1
            self._preempted_count = len(self._preempted)
        return True

    @staticmethod
    def _strictly_more_urgent(
        candidate: ServingRequest, victim: ServingRequest
    ) -> bool:
        if candidate.priority != victim.priority:
            return candidate.priority > victim.priority
        if candidate.deadline_at is None:
            return False
        if victim.deadline_at is None:
            return True
        return candidate.deadline_at < victim.deadline_at

    @staticmethod
    def _preemption_victim_key(request: ServingRequest):
        # Prefer the least urgent request, then the cheapest exact replay.
        has_deadline = request.deadline_at is not None
        deadline = 0.0 if request.deadline_at is None else -request.deadline_at
        replay_tokens = len(request.token_ids) + max(
            0, len(request.generated_token_ids) - 1
        )
        return (
            request.priority,
            has_deadline,
            deadline,
            replay_tokens,
            request.request_id,
        )

    def _recover_preempted(self, active: OrderedDict[int, ServingRequest]) -> bool:
        recover = getattr(self.backend, "recover", None)
        if not callable(recover) or not self._preempted:
            return False
        did_work = False
        visits = len(self._preempted)
        for _ in range(visits):
            request = self._preempted.popleft()
            if request.cancelled.is_set():
                self._finish(request, "cancelled", active=active, release=False)
                did_work = True
                continue
            if self._deadline_expired(request):
                self._fail(
                    request,
                    TimeoutError("request deadline expired while preempted"),
                    active=active,
                    release=False,
                )
                did_work = True
                continue
            if len(active) >= self.max_active_requests:
                self._preempted.append(request)
                continue
            try:
                decision = recover(request)
            except Exception as exc:
                with self._stats_lock:
                    self._recovery_failures_total += 1
                self._fail(request, exc, active=active, release=True)
                did_work = True
                continue
            if isinstance(decision, AdmissionDecision) and not decision.admitted:
                if decision.retryable:
                    self._preempted.append(request)
                    continue
                with self._stats_lock:
                    self._recovery_failures_total += 1
                self._fail(
                    request,
                    MemoryError(decision.reason or "preempted request cannot recover"),
                    active=active,
                    release=True,
                )
                did_work = True
                continue
            request.recovery_count += 1
            active[request.request_id] = request
            with self._stats_lock:
                self._recoveries_total += 1
            did_work = True
        with self._stats_lock:
            self._preempted_count = len(self._preempted)
        return did_work

    def _submit_independent_decode(
        self,
        active: OrderedDict[int, ServingRequest],
        pending: dict[object, tuple[tuple[ServingRequest, ...], Future]],
        executor: ThreadPoolExecutor,
    ) -> bool:
        """Launch one decode batch per ready physical worker.

        The legacy backend-wide ``decode`` call is still used inside each
        future, but requests are grouped before submission. This deliberately
        removes only the cross-worker completion barrier; device-local batching,
        error recovery, and the old synchronous path remain intact.
        """

        group_for = getattr(self.backend, "decode_executor_group", None)
        if not callable(group_for) or not active:
            return False
        eligible = []
        groups_by_request = {}
        group_failures = []
        for request in tuple(active.values()):
            if request.request_id in self._decode_inflight_ids:
                continue
            try:
                group = group_for(request)
            except Exception as exc:
                group_failures.append((request, exc))
                continue
            if group in pending:
                continue
            eligible.append(request)
            groups_by_request[request.request_id] = group
        for request, exc in group_failures:
            self._fail(request, exc, active=active, release=True)
        if not eligible:
            return bool(group_failures)
        selected = self._decode_scheduler.select(
            eligible, min(self.max_batch_size, self.max_step_tokens)
        )
        grouped: OrderedDict[object, list[ServingRequest]] = OrderedDict()
        for request in selected:
            grouped.setdefault(groups_by_request[request.request_id], []).append(
                request
            )
        submitted = False
        for group, requests in grouped.items():
            if group in pending:
                continue
            batch = tuple(requests)
            future = executor.submit(self.backend.decode, batch)
            future.add_done_callback(lambda _future: self._wake.set())
            pending[group] = (batch, future)
            self._decode_inflight_ids.update(request.request_id for request in batch)
            submitted = True
        return submitted or bool(group_failures)

    def _collect_independent_decode(
        self,
        active: OrderedDict[int, ServingRequest],
        pending: dict[object, tuple[tuple[ServingRequest, ...], Future]],
        *,
        wait: bool = False,
    ) -> bool:
        completed = []
        for group, (_, future) in pending.items():
            if wait or future.done():
                completed.append(group)
                if wait:
                    break
        for group in completed:
            batch, future = pending.pop(group)
            for request in batch:
                self._decode_inflight_ids.discard(request.request_id)
            try:
                values = future.result()
                if len(values) != len(batch):
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
                    continue
                for request in batch:
                    if request.cancelled.is_set() or self._stop.is_set():
                        self._finish(request, "cancelled", active=active, release=True)
                        continue
                    error = exc.errors.get(request.request_id)
                    if error is not None:
                        if not self._suspend_recoverable_failure(
                            request, error, active
                        ):
                            self._fail(request, error, active=active, release=True)
                    else:
                        self._accept_decode_sample(
                            request,
                            exc.samples[request.request_id],
                            active,
                            decode_batch_size=len(batch),
                        )
                continue
            except Exception as exc:
                for request in batch:
                    if request.cancelled.is_set() or self._stop.is_set():
                        self._finish(request, "cancelled", active=active, release=True)
                    elif not self._suspend_recoverable_failure(request, exc, active):
                        self._fail(request, exc, active=active, release=True)
                continue
            for request, value in zip(batch, values, strict=True):
                if request.cancelled.is_set() or self._stop.is_set():
                    self._finish(request, "cancelled", active=active, release=True)
                    continue
                self._accept_decode_sample(
                    request,
                    value,
                    active,
                    decode_batch_size=len(batch),
                )
        return bool(completed)

    def _decode_once(self, active: OrderedDict[int, ServingRequest]) -> None:
        batch = self._decode_scheduler.select(
            active.values(), min(self.max_batch_size, self.max_step_tokens)
        )
        physical_batch_sizes = {request.request_id: len(batch) for request in batch}
        batch_size_resolver = getattr(self.backend, "decode_batch_sizes", None)
        if batch and callable(batch_size_resolver):
            try:
                resolved = batch_size_resolver(batch)
                physical_batch_sizes.update(
                    {
                        int(request_id): max(1, int(size))
                        for request_id, size in dict(resolved).items()
                    }
                )
            except Exception:
                # Observability must never fail an otherwise valid decode.
                pass
        try:
            if self.dp_graph_sync:
                from hydraserve.engine.dp_graph_sync import synchronize_dp_token_count

                plan = synchronize_dp_token_count(
                    len(batch), process_group=self.dp_process_group
                )
                if plan.padding_tokens:
                    decode_padded = getattr(self.backend, "decode_padded", None)
                    if not callable(decode_padded):
                        raise RuntimeError(
                            "DP CUDA Graph synchronization requires backend.decode_padded"
                        )
                    token_ids = decode_padded(batch, plan.synchronized_tokens)
                elif batch:
                    token_ids = self.backend.decode(batch)
                else:
                    return
            else:
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
                    if not self._suspend_recoverable_failure(request, error, active):
                        self._fail(request, error, active=active, release=True)
                else:
                    self._accept_decode_sample(
                        request,
                        exc.samples[request.request_id],
                        active,
                        decode_batch_size=physical_batch_sizes[request.request_id],
                    )
            return
        except Exception as exc:
            for request in batch:
                if not self._suspend_recoverable_failure(request, exc, active):
                    self._fail(request, exc, active=active, release=True)
            return
        for request, token_id in zip(batch, token_ids, strict=True):
            self._accept_decode_sample(
                request,
                token_id,
                active,
                decode_batch_size=physical_batch_sizes[request.request_id],
            )

    def _suspend_recoverable_failure(
        self,
        request: ServingRequest,
        error: BaseException,
        active: OrderedDict[int, ServingRequest],
    ) -> bool:
        checker = getattr(self.backend, "is_recoverable_decode_error", None)
        abandon = getattr(self.backend, "abandon", None)
        recover = getattr(self.backend, "recover", None)
        if (
            not callable(checker)
            or not callable(abandon)
            or not callable(recover)
            or not checker(request.request_id, error)
        ):
            return False
        try:
            abandon(request.request_id)
        except Exception:
            return False
        active.pop(request.request_id, None)
        self._decode_scheduler.forget(request.request_id)
        self._preempted.append(request)
        with self._stats_lock:
            self._fault_suspensions_total += 1
            self._preempted_count += 1
        return True

    def _accept_decode_sample(
        self,
        request: ServingRequest,
        value: int | TokenSample,
        active: OrderedDict[int, ServingRequest],
        *,
        decode_batch_size: int,
    ) -> None:
        if self._deadline_expired(request):
            self._fail(
                request,
                TimeoutError("request deadline expired during decode"),
                active=active,
                release=True,
            )
            return
        sample = self._normalize_sample(value)
        request.generated_token_ids.append(sample.token_id)
        self._emit(request, sample, decode_batch_size=decode_batch_size)
        reason = self._finish_reason(request, sample.token_id)
        if reason is not None:
            self._finish(request, reason, active=active, release=True)

    def _remove_cancelled_or_expired(
        self, active: OrderedDict[int, ServingRequest]
    ) -> bool:
        cancelled = [
            request
            for request in active.values()
            if request.request_id not in self._decode_inflight_ids
            and request.cancelled.is_set()
        ]
        for request in cancelled:
            self._finish(request, "cancelled", active=active, release=True)
        expired = [
            request
            for request in active.values()
            if request.request_id not in self._decode_inflight_ids
            and self._deadline_expired(request)
        ]
        for request in expired:
            self._fail(
                request,
                TimeoutError("request deadline expired during decode"),
                active=active,
                release=True,
            )
        return bool(cancelled or expired)

    @staticmethod
    def _deadline_expired(request: ServingRequest) -> bool:
        return request.deadline_at is not None and monotonic() >= request.deadline_at

    def _finish_reason(self, request: ServingRequest, token_id: int) -> str | None:
        if (
            not request.ignore_eos
            and self.eos_token_id is not None
            and token_id == self.eos_token_id
        ):
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

    def _emit(
        self,
        request: ServingRequest,
        sample: TokenSample,
        *,
        decode_batch_size: int | None = None,
    ) -> None:
        self._handle(request.request_id)._put(
            GenerationEvent(
                request.request_id,
                token_id=sample.token_id,
                logprob=sample.logprob,
                top_logprobs=sample.top_logprobs,
                decode_batch_size=decode_batch_size,
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
        handle = self._pop_handle(request.request_id)
        if release:
            self._schedule_release(
                request.request_id,
                handle,
                finish_reason=reason,
                request_error=None,
            )
        elif handle is not None:
            handle._put(
                GenerationEvent(request.request_id, finished=True, finish_reason=reason)
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
        handle = self._pop_handle(request.request_id)
        if release:
            self._schedule_release(
                request.request_id,
                handle,
                finish_reason="error",
                request_error=str(exc),
            )
        elif handle is not None:
            handle._put(
                GenerationEvent(
                    request.request_id,
                    finished=True,
                    finish_reason="error",
                    error=str(exc),
                )
            )

    def _schedule_release(
        self,
        request_id: int,
        handle: GenerationHandle | None,
        *,
        finish_reason: str,
        request_error: str | None,
    ) -> None:
        """Release device state off the decode loop, then emit the terminal event."""

        with self._stats_lock:
            self._release_pending_count += 1

        def release_and_finish() -> None:
            release_error = None
            try:
                self.backend.release(request_id)
            except Exception as exc:
                release_error = f"release failed: {exc}"
            finally:
                with self._stats_lock:
                    self._release_pending_count = max(
                        0, self._release_pending_count - 1
                    )
                    self._release_total += 1
                    if release_error is not None:
                        self._release_failures_total += 1

            if handle is None:
                return
            error = request_error or release_error
            handle._put(
                GenerationEvent(
                    request_id,
                    finished=True,
                    finish_reason="error" if error else finish_reason,
                    error=error,
                )
            )

        self._release_executor.submit(release_and_finish)

    def _cancel_incoming(self) -> None:
        empty: OrderedDict[int, ServingRequest] = OrderedDict()
        while self._preempted:
            request = self._preempted.popleft()
            self._finish(request, "cancelled", active=empty, release=False)
        with self._stats_lock:
            self._preempted_count = 0
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
                float("inf") if item[0].deadline_at is None else item[0].deadline_at,
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

    @property
    def active_count(self) -> int:
        with self._stats_lock:
            return self._active_count

    @property
    def prefill_pending_count(self) -> int:
        with self._stats_lock:
            return self._prefill_pending_count

    @property
    def preempted_count(self) -> int:
        with self._stats_lock:
            return self._preempted_count

    @property
    def preemptions_total(self) -> int:
        with self._stats_lock:
            return self._preemptions_total

    @property
    def preemption_failures_total(self) -> int:
        with self._stats_lock:
            return self._preemption_failures_total

    @property
    def recoveries_total(self) -> int:
        with self._stats_lock:
            return self._recoveries_total

    @property
    def recovery_failures_total(self) -> int:
        with self._stats_lock:
            return self._recovery_failures_total

    @property
    def fault_suspensions_total(self) -> int:
        with self._stats_lock:
            return self._fault_suspensions_total

    @property
    def release_pending_count(self) -> int:
        with self._stats_lock:
            return self._release_pending_count

    @property
    def release_total(self) -> int:
        with self._stats_lock:
            return self._release_total

    @property
    def release_failures_total(self) -> int:
        with self._stats_lock:
            return self._release_failures_total

    @property
    def prefill_slot_deferrals_total(self) -> int:
        with self._stats_lock:
            return self._prefill_slot_deferrals_total

    def _publish_scheduler_depth(self, active: int, prefill_pending: int) -> None:
        with self._stats_lock:
            self._active_count = active
            self._prefill_pending_count = prefill_pending

    def _handle(self, request_id: int) -> GenerationHandle:
        with self._handles_lock:
            return self._handles[request_id]

    def _pop_handle(self, request_id: int) -> GenerationHandle | None:
        with self._handles_lock:
            return self._handles.pop(request_id, None)


class RuntimeGenerationBackend:
    """HydraServe runtime adapter; no external model-execution backend is used."""

    release_parallelism = 1

    def __init__(
        self,
        runtime,
        paged_cache,
        *,
        prefill_chunk_size: int = 4096,
        max_state_slots: int = 64,
        max_decode_batch_size: int | None = None,
    ) -> None:
        max_decode_batch_size = (
            max_state_slots if max_decode_batch_size is None else max_decode_batch_size
        )
        if min(prefill_chunk_size, max_state_slots, max_decode_batch_size) <= 0:
            raise ValueError("prefill chunk size and state slots must be positive")
        from hydraserve.cache import GpuLinearStatePool, RequestStateSlotManager

        self.runtime = runtime
        self.paged_cache = paged_cache
        self.prefill_chunk_size = prefill_chunk_size
        config = getattr(runtime, "config", None)
        self.state_pool = (
            GpuLinearStatePool(
                max_state_slots,
                config,
                device=runtime.device,
                workspace_capacity=min(max_state_slots, max_decode_batch_size),
            )
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
        return self._reserve(request, request.token_ids)

    def _reserve(
        self, request: ServingRequest, initial_token_ids: tuple[int, ...]
    ) -> AdmissionDecision:
        manager = self.paged_cache.block_manager
        initial_capacity = self.capacity()
        total_tokens = self._total_kv_tokens(request)
        required = manager.blocks_required(total_tokens)
        if required > manager.usable_blocks:
            return AdmissionDecision.reject(
                f"request needs {required} KV blocks, worker capacity is {manager.usable_blocks}"
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
                    allocation.num_tokens != len(initial_token_ids)
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
                    len(initial_token_ids),
                    reserve_tokens=total_tokens,
                    token_ids=initial_token_ids,
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
                request.worker_pool = "collocated"
                request.route_decode_load = initial_capacity.decode_load
            return AdmissionDecision.accept()

    def preempt(self, request_id: int) -> None:
        """Release runtime state and KV so another request can be admitted."""
        self.release(request_id)

    def recover(self, request: ServingRequest) -> AdmissionDecision:
        """Recompute exact state without sampling or re-emitting prior tokens."""
        import torch

        replay_token_ids = request.token_ids + tuple(request.generated_token_ids[:-1])
        decision = self._reserve(request, replay_token_ids)
        if not decision.admitted:
            return decision
        try:
            input_ids = torch.tensor(
                [replay_token_ids],
                device=getattr(self.runtime, "input_device", self.runtime.device),
                dtype=torch.long,
            )
            with torch.inference_mode():
                _, state = self.runtime.prefill(
                    input_ids,
                    chunk_size=self.prefill_chunk_size,
                    paged_cache=self.paged_cache,
                    request_id=request.request_id,
                )
            if self.state_pool is not None:
                state = self.state_pool.install(request.request_id, state)
            self.states[request.request_id] = state
            return AdmissionDecision.accept()
        except Exception:
            try:
                self.release(request.request_id)
            except Exception:
                pass
            raise

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
                [request.token_ids],
                device=getattr(self.runtime, "input_device", self.runtime.device),
                dtype=torch.long,
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
                state = self.states.get(request_id)
                # Pooled recurrent states are committed atomically at the end of
                # the transaction, so a failed decode leaves them untouched; the
                # state-pool identity check requires the object be preserved.
                if (
                    state is not None
                    and getattr(state, "_state_pool_ref", None) is None
                ):
                    self.states[request_id] = self._checkpoint_state(
                        checkpoints[request_id]
                    )

        def attempt(request_ids: tuple[int, ...]) -> tuple[TokenSample, ...]:
            manager.grow_many(request_ids, additional_tokens=1)
            input_ids = torch.tensor(
                [request_by_id[item].generated_token_ids[-1] for item in request_ids],
                device=getattr(self.runtime, "input_device", self.runtime.device),
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

    def cache_stats(self) -> dict[str, int | float]:
        stats = self.paged_cache.stats()
        if self.state_pool is not None:
            stats.update(self.state_pool.stats())
        return stats

    def audit_resources(self) -> dict[str, int | float]:
        stats = self.paged_cache.audit()
        state = self.state_slots.capacity()
        if state.allocated_slots != len(self.states):
            raise RuntimeError("recurrent-state slots and runtime states disagree")
        return {
            **stats,
            **({} if self.state_pool is None else self.state_pool.stats()),
            "state_total_slots": state.total_slots,
            "state_allocated_slots": state.allocated_slots,
            "state_free_slots": state.free_slots,
        }
