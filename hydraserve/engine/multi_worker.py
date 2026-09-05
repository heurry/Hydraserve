"""One-prefill, many-decode worker generation backend."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from itertools import count
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from time import monotonic
from typing import Callable
from uuid import uuid4

from hydraserve.engine.pd_service import (
    PDWorkerConfig,
    RoutingStats,
    _decode_worker,
    _pd_protocol_trace,
    _prefill_worker,
)
from hydraserve.engine.serving_loop import (
    AdmissionDecision,
    BackendCapacity,
    PartialDecodeError,
    ServingRequest,
)
from hydraserve.engine.sampling import TokenSample
from hydraserve.router import (
    AdaptiveRouter,
    CostAwareRouter,
    DecodeWorkerRegistry,
    DecodeWorkerSnapshot,
    Route,
    RouteDecision,
    RouteReason,
    WorkerTopology,
)


def _normalize_device(device: str) -> str:
    """Accept a bare CUDA index (``"2"``) as well as a full ``"cuda:2"`` string."""
    value = device.strip()
    return f"cuda:{value}" if value.isdigit() else value


class HybridRole(str, Enum):
    """Runtime role of a work-conserving prefill worker."""

    DECODE = "decode"
    PREFILL_PENDING = "prefill_pending"
    PREFILL_ACTIVE = "prefill_active"


@dataclass(frozen=True, slots=True)
class PDClusterConfig:
    model_dir: str
    decode_devices: tuple[str, ...]
    prefill_device: str = "cuda:0"
    prefill_devices: tuple[str, ...] = ()
    cache_tokens_per_worker: int = 65_536
    block_size: int = 16
    max_state_slots_per_worker: int = 64
    use_flash_attention: bool = True
    prefill_chunk_size: int = 4096
    prefix_cache_blocks: int = 0
    prefix_cache_min_frequency: int = 2
    kv_headroom_blocks: int = 0
    topologies: tuple[WorkerTopology, ...] = ()
    max_decode_batch_size_per_worker: int = 64
    kv_quant: str | None = None
    host_prefix_cache_bytes: int = 0
    transfer_backend: str = "shm-ring"
    transfer_quant: str | None = None
    transfer_target_bytes: int = 8 << 20
    max_inflight_transfer_chunks: int = 2
    max_concurrent_prepares_per_worker: int = 2
    shm_ring_slots: int = 3
    shm_ring_slot_bytes: int = 64 << 20
    worker_log_dir: str = ""
    pd_schedule: str = "round-robin"
    conditional_pd_tokens: int = 0
    prefill_short_policy: str = "work-conserving"
    prefill_preempt_max_ops: int = 8
    hybrid_prefill_reserve_tokens: int = -1
    hybrid_long_overflow_ms: float = 5000.0
    pd_prefill_token_budget: int = 0
    hybrid_short_max_prefill_backlog_tokens: int = 0
    hybrid_short_max_assigned_work: int = 0
    hybrid_long_pressure_hold_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_dir or not self.decode_devices:
            raise ValueError("cluster requires a model and decode workers")
        object.__setattr__(
            self,
            "decode_devices",
            tuple(_normalize_device(device) for device in self.decode_devices),
        )
        object.__setattr__(
            self, "prefill_device", _normalize_device(self.prefill_device)
        )
        if self.prefill_devices:
            object.__setattr__(
                self,
                "prefill_devices",
                tuple(_normalize_device(device) for device in self.prefill_devices),
            )
        else:
            object.__setattr__(self, "prefill_devices", (self.prefill_device,))
        if len(set(self.decode_devices)) != len(self.decode_devices):
            raise ValueError("decode devices must be unique")
        if len(set(self.prefill_devices)) != len(self.prefill_devices):
            raise ValueError("prefill devices must be unique")
        if set(self.prefill_devices) & set(self.decode_devices):
            raise ValueError("prefill and decode devices must be distinct")
        if (
            min(
                self.cache_tokens_per_worker,
                self.block_size,
                self.max_state_slots_per_worker,
                self.max_decode_batch_size_per_worker,
                self.prefill_chunk_size,
                self.prefix_cache_min_frequency,
                self.transfer_target_bytes,
                self.max_inflight_transfer_chunks,
                self.max_concurrent_prepares_per_worker,
                self.shm_ring_slots,
                self.shm_ring_slot_bytes,
                self.prefill_preempt_max_ops,
            )
            <= 0
        ):
            raise ValueError("cluster resource limits must be positive")
        if self.prefix_cache_blocks < 0:
            raise ValueError("prefix cache blocks cannot be negative")
        if self.host_prefix_cache_bytes < 0:
            raise ValueError("host prefix cache bytes cannot be negative")
        if self.transfer_backend not in {"shm-ring", "shm"}:
            raise ValueError("transfer_backend must be shm-ring or shm")
        if self.transfer_quant not in {None, "int8"}:
            raise ValueError("transfer_quant must be None or int8")
        total_blocks = (
            self.cache_tokens_per_worker + self.block_size - 1
        ) // self.block_size
        if not 0 <= self.kv_headroom_blocks < total_blocks:
            raise ValueError("KV headroom must be below physical cache blocks")
        if self.topologies and len(self.topologies) != len(self.decode_devices):
            raise ValueError("topologies must match decode devices")
        if self.pd_schedule not in {"round-robin", "kv-aware", "load-aware"}:
            raise ValueError(
                "pd_schedule must be one of round-robin, kv-aware, load-aware"
            )
        if self.conditional_pd_tokens < 0:
            raise ValueError("conditional PD threshold cannot be negative")
        if self.prefill_short_policy not in {"never", "work-conserving"}:
            raise ValueError("prefill_short_policy must be never or work-conserving")
        if self.hybrid_prefill_reserve_tokens < -1:
            raise ValueError("hybrid prefill reserve tokens must be -1 or non-negative")
        if self.hybrid_long_overflow_ms < 0:
            raise ValueError("hybrid long overflow wait must be non-negative")
        if self.pd_prefill_token_budget < 0:
            raise ValueError("PD prefill token budget cannot be negative")
        if self.hybrid_short_max_prefill_backlog_tokens < 0:
            raise ValueError("hybrid short prefill backlog budget cannot be negative")
        if self.hybrid_short_max_assigned_work < 0:
            raise ValueError("hybrid short assigned-work budget cannot be negative")
        if self.hybrid_long_pressure_hold_ms < 0:
            raise ValueError("hybrid long pressure hold must be non-negative")

    @property
    def effective_hybrid_prefill_reserve_tokens(self) -> int:
        """KV capacity kept available for a temporary long-prefill role.

        ``-1`` selects a conservative automatic reserve: half the worker cache,
        capped at 32K tokens.  ``0`` preserves the legacy work-conserving
        behavior without a reserved long-prefill region.
        """

        if self.hybrid_prefill_reserve_tokens >= 0:
            return self.hybrid_prefill_reserve_tokens
        return min(32_768, self.cache_tokens_per_worker // 2)

    def worker_config(self, worker_index: int) -> PDWorkerConfig:
        return PDWorkerConfig(
            self.model_dir,
            prefill_device=self.prefill_devices[0],
            decode_device=self.decode_devices[worker_index],
            cache_tokens=self.cache_tokens_per_worker,
            block_size=self.block_size,
            use_flash_attention=self.use_flash_attention,
            prefill_chunk_size=self.prefill_chunk_size,
            max_state_slots=self.max_state_slots_per_worker,
            max_decode_batch_size=min(
                self.max_state_slots_per_worker,
                self.max_decode_batch_size_per_worker,
            ),
            prefix_cache_blocks=self.prefix_cache_blocks,
            prefix_cache_min_frequency=self.prefix_cache_min_frequency,
            kv_headroom_blocks=self.kv_headroom_blocks,
            kv_quant=self.kv_quant,
            host_prefix_cache_bytes=self.host_prefix_cache_bytes,
            transfer_backend=self.transfer_backend,
            transfer_quant=self.transfer_quant,
            transfer_target_bytes=self.transfer_target_bytes,
            max_inflight_transfer_chunks=self.max_inflight_transfer_chunks,
            max_concurrent_prepares=self.max_concurrent_prepares_per_worker,
            shm_ring_slots=self.shm_ring_slots,
            shm_ring_slot_bytes=self.shm_ring_slot_bytes,
            prefill_preempt_max_ops=self.prefill_preempt_max_ops,
        )

    def prefill_config(self, index: int) -> PDWorkerConfig:
        """Config for the index-th prefill worker.

        The prefill worker also serves collocated short requests (W4), so it
        needs the same state/decoding capacity as a decode worker in addition
        to its PD prefill role.
        """
        return PDWorkerConfig(
            self.model_dir,
            prefill_device=self.prefill_devices[index],
            decode_device=self.decode_devices[0],
            cache_tokens=self.cache_tokens_per_worker,
            block_size=self.block_size,
            use_flash_attention=self.use_flash_attention,
            prefill_chunk_size=self.prefill_chunk_size,
            max_state_slots=self.max_state_slots_per_worker,
            max_decode_batch_size=self.max_decode_batch_size_per_worker,
            prefix_cache_blocks=self.prefix_cache_blocks,
            prefix_cache_min_frequency=self.prefix_cache_min_frequency,
            kv_headroom_blocks=self.kv_headroom_blocks,
            kv_quant=self.kv_quant,
            host_prefix_cache_bytes=self.host_prefix_cache_bytes,
            transfer_backend=self.transfer_backend,
            transfer_quant=self.transfer_quant,
            transfer_target_bytes=self.transfer_target_bytes,
            max_inflight_transfer_chunks=self.max_inflight_transfer_chunks,
            max_concurrent_prepares=self.max_concurrent_prepares_per_worker,
            shm_ring_slots=self.shm_ring_slots,
            shm_ring_slot_bytes=self.shm_ring_slot_bytes,
            prefill_preempt_max_ops=self.prefill_preempt_max_ops,
        )


PrefixAffinity = Callable[[ServingRequest, int], int]


class WorkerUnavailableError(RuntimeError):
    """A decode worker exited before completing its RPC."""


class WorkerStateLostError(RuntimeError):
    """A request lost device-local state when its decode worker failed."""


@dataclass(frozen=True, slots=True)
class WorkerRecoveryStats:
    total_workers: int
    healthy_workers: int
    attempts: int
    successes: int
    failures: int
    recovering_workers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefillRecoveryStats:
    healthy: bool
    attempts: int
    successes: int
    failures: int
    recovering: bool


class MultiWorkerGenerationBackend:
    """Persistent 1P+ND backend with immutable per-request worker binding."""

    supports_async_prefill = True

    @property
    def release_parallelism(self) -> int:
        return len(self.config.prefill_devices) + len(self.config.decode_devices)

    @property
    def prefill_parallelism(self) -> int:
        """Legacy aggregate concurrency for serving-loop compatibility."""
        return len(self.config.prefill_devices) + len(self.config.decode_devices)

    @property
    def supports_independent_decode(self) -> bool:
        """Allow physical workers to advance without a cross-worker barrier."""

        return True

    @property
    def decode_executor_parallelism(self) -> int:
        return len(self.config.prefill_devices) + len(self.config.decode_devices)

    def decode_executor_group(self, request: ServingRequest) -> tuple[str, int]:
        """Return the physical worker that owns a request's decode state."""

        with self._state_lock:
            prefill_index = self._prefill_bound.get(request.request_id)
        if prefill_index is not None:
            return ("prefill", prefill_index)
        return ("decode", self.registry.worker_for(request.request_id))

    def prefill_admission_tokens(self, request: ServingRequest) -> int:
        """Charge one runtime chunk while the physical executor owns the RPC."""

        return min(len(request.token_ids), self.config.prefill_chunk_size)

    @property
    def prefill_executor_limits(self) -> dict[str, int]:
        """Independent host slots for P-prefill and D-collocated prefill.

        A 1P+3D conditional deployment must not collapse all prefill calls into
        one host executor merely because it owns one dedicated P worker.
        Worker-local RPC locks remain the physical-GPU serialization boundary.
        """
        return {
            "prefill": len(self.config.prefill_devices),
            "decode": len(self.config.decode_devices),
        }

    def prefill_executor_group(self, request: ServingRequest) -> str:
        """Return the physical worker pool used by an admitted request."""
        if request.request_id in self._prefill_bound:
            return "prefill"
        if request.route == Route.PD_DISAGGREGATED.value:
            return "prefill"
        return "decode"

    def prefill_executor_group_hint(self, request: ServingRequest) -> str | None:
        """Predict deterministic routing before decode-side KV admission.

        The serving loop uses this only to cap outstanding work per physical
        prefill pool.  Adaptive and work-conserving short routes deliberately
        return ``None`` because their target depends on live load at admission.
        """

        conditional = int(getattr(self.config, "conditional_pd_tokens", 0) or 0)
        force_pd = int(
            getattr(getattr(self.router, "config", None), "force_pd_tokens", 0) or 0
        )
        threshold = conditional or force_pd
        if not threshold:
            return None
        if not self._prefill_available():
            return "decode"
        if len(request.token_ids) >= threshold:
            if (
                self._hybrid_prefill_slot_available()
                and self._pd_prefill_budget_available(request)
            ):
                return "prefill"
            return (
                "decode"
                if self._idle_decode_slot_available(request)
                or self._long_overflow_ready(request)
                else "prefill"
            )
        if (
            conditional
            and getattr(self.config, "prefill_short_policy", "work-conserving")
            == "never"
        ):
            return "decode"
        return None

    def __init__(
        self,
        config: PDClusterConfig,
        *,
        router: AdaptiveRouter | None = None,
        prefix_affinity: PrefixAffinity | None = None,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
        receiver_dispatch_timeout: float = 5.0,
        receiver_arm_timeout: float = 10.0,
        max_worker_restarts: int = 3,
        worker_restart_backoff_s: float = 0.5,
    ) -> None:
        if (
            max_worker_restarts <= 0
            or worker_restart_backoff_s < 0
            or receiver_dispatch_timeout <= 0
            or receiver_arm_timeout <= 0
            or operation_timeout <= 0
        ):
            raise ValueError("invalid decode worker recovery policy")
        self.config = config
        self.router = router or CostAwareRouter()
        self.prefix_affinity = prefix_affinity
        self.operation_timeout = operation_timeout
        self.receiver_dispatch_timeout = min(
            receiver_dispatch_timeout, operation_timeout
        )
        self.receiver_arm_timeout = min(receiver_arm_timeout, operation_timeout)
        self.startup_timeout = startup_timeout
        self.max_worker_restarts = max_worker_restarts
        self.worker_restart_backoff_s = worker_restart_backoff_s
        self.namespace = f"hydraserve-cluster-{uuid4().hex}"
        worker_count = len(config.decode_devices)
        self._namespaces = tuple(
            f"{self.namespace}-decode-{index}" for index in range(worker_count)
        )
        self._context = mp.get_context("spawn")
        self._bootstrap_server = None
        self._bootstrap_address = None
        try:
            from hydraserve.transfer import BootstrapServer

            self._bootstrap_server = BootstrapServer().start()
            self._bootstrap_address = self._bootstrap_server.address
        except PermissionError:
            self._bootstrap_server = None
        if config.worker_log_dir:
            Path(config.worker_log_dir).mkdir(parents=True, exist_ok=True)
        prefill_count = len(config.prefill_devices)
        self._prefill_commands = [self._context.Queue() for _ in range(prefill_count)]
        self._prefill_responses = [self._context.Queue() for _ in range(prefill_count)]
        self._prefill_locks = [Lock() for _ in range(prefill_count)]
        self._prefill_response_locks = [Lock() for _ in range(prefill_count)]
        self._prefill_waiters: list[dict[int, Queue]] = [
            {} for _ in range(prefill_count)
        ]
        self._prefill_rpc_ids = count(1)
        self._decode_commands = [self._context.Queue() for _ in range(worker_count)]
        self._decode_responses = [self._context.Queue() for _ in range(worker_count)]
        self._decode_locks = [Lock() for _ in range(worker_count)]
        self._decode_response_locks = [Lock() for _ in range(worker_count)]
        self._decode_waiters: list[dict[int, Queue]] = [
            {} for _ in range(worker_count)
        ]
        self._decode_rpc_ids = count(1)
        self._decode_processes = [
            self._new_decode_process(index) for index in range(worker_count)
        ]
        self._prefill_processes = [
            self._new_prefill_process(index) for index in range(prefill_count)
        ]
        total_blocks = (
            config.cache_tokens_per_worker + config.block_size - 1
        ) // config.block_size - config.kv_headroom_blocks
        topologies = config.topologies or tuple(
            WorkerTopology() for _ in range(worker_count)
        )
        self.registry = DecodeWorkerRegistry(
            tuple(
                DecodeWorkerSnapshot(
                    index,
                    device,
                    BackendCapacity(
                        total_blocks,
                        total_blocks,
                        config.max_state_slots_per_worker,
                        config.max_state_slots_per_worker,
                    ),
                    topologies[index],
                )
                for index, device in enumerate(config.decode_devices)
            )
        )
        self._reserved_blocks = [dict() for _ in range(worker_count)]
        self._prefill_reserved_blocks = [dict() for _ in range(prefill_count)]
        self._decode_assigned_work = [0] * worker_count
        self._decode_prefill_tokens = [0] * worker_count
        self._decode_request_loads: dict[int, tuple[int, int, int]] = {}
        self._prefill_assigned_work = [0] * prefill_count
        self._prefill_prefill_tokens = [0] * prefill_count
        self._prefill_request_loads: dict[int, tuple[int, int, int]] = {}
        self._prefill_capacities = [
            BackendCapacity(
                total_blocks,
                total_blocks,
                config.max_state_slots_per_worker,
                config.max_state_slots_per_worker,
            )
            for _ in range(prefill_count)
        ]
        self._decode_capacity_versions = [-1] * worker_count
        self._prefill_capacity_versions = [-1] * prefill_count
        self._worker_cache_stats: list[dict[str, int | float]] = [
            {} for _ in range(worker_count)
        ]
        self._route_decisions: dict[int, RouteDecision] = {}
        self._lost_requests: set[int] = set()
        self._host_prefix_tokens: dict[int, int] = {}
        self._state_lock = RLock()
        self._pd_executor = ThreadPoolExecutor(max_workers=max(4, 2 * prefill_count))
        self._prefill_healthy = [True] * prefill_count
        self._prefill_pending = [0] * prefill_count
        self._prefill_long_inflight = [0] * prefill_count
        self._prefill_dispatch_claims = [0] * prefill_count
        self._closed = False
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
        self._prefill_short_count = 0
        self._recovery_stop = Event()
        self._recovering_workers: set[int] = set()
        self._recovery_threads: dict[int, Thread] = {}
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._recovery_failures = 0
        self._prefill_recovering = [False] * prefill_count
        self._prefill_recovery_threads: list[Thread | None] = [None] * prefill_count
        self._prefill_recovery_attempts = [0] * prefill_count
        self._prefill_recovery_successes = [0] * prefill_count
        self._prefill_recovery_failures = [0] * prefill_count
        self._prefill_round_robin = 0
        self._prefill_serve_round_robin = 0
        self._decode_round_robin = 0
        self._short_round_robin = 0
        self._hybrid_roles = [HybridRole.DECODE] * prefill_count
        self._hybrid_long_pressure_until = 0.0
        self._hybrid_short_gate_closures = 0
        # Long requests are bound at admission time so the hybrid worker stops
        # accepting new short requests before the asynchronous prefill RPC is
        # actually submitted.  The legacy unbound selector remains below as a
        # compatibility fallback for old callers and lightweight test doubles.
        self._long_prefill_bound: dict[int, int] = {}
        # request_id -> prefill worker index for collocated requests served on a
        # prefill worker (W4); decode-worker-bound requests stay in the registry.
        self._prefill_bound: dict[int, int] = {}
        self._replay_mismatches = 0
        self._prefill_chunk_preemptions = 0

        for process in self._decode_processes:
            process.start()
        for process in self._prefill_processes:
            process.start()
        try:
            decode_ready = [
                self._get(queue, startup_timeout) for queue in self._decode_responses
            ]
            prefill_ready = [
                self._get(queue, startup_timeout) for queue in self._prefill_responses
            ]
            for index, result in enumerate(decode_ready):
                try:
                    self._check(result, "ready")
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"decode worker {index} failed startup: {exc}"
                    ) from exc
            for index, result in enumerate(prefill_ready):
                try:
                    self._check(result, "ready")
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"prefill worker {index} failed startup: {exc}"
                    ) from exc
            names = {result["model_name"] for result in decode_ready}
            names.update(result["model_name"] for result in prefill_ready)
            if len(names) != 1:
                raise RuntimeError("cluster workers loaded different models")
            self.model_name = names.pop()
            for worker_id, result in enumerate(decode_ready):
                self._update_worker_capacity(worker_id, result)
            for worker_id, result in enumerate(prefill_ready):
                self._update_prefill_capacity(worker_id, result)
        except Exception:
            self.close(force=True)
            raise

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        # ``ContinuousGenerationLoop`` admits before submitting prefill, while
        # ``prefill()`` defensively admits again.  P-worker collocated requests
        # are not registered in the decode registry, so recognize that binding
        # explicitly and avoid issuing a second reserve RPC/counter increment.
        with self._state_lock:
            if request.request_id in self._prefill_bound:
                return AdmissionDecision.accept()
        try:
            self.registry.worker_for(request.request_id)
            return AdmissionDecision.accept()
        except KeyError:
            pass
        required_blocks = self._required_blocks(request)
        prefix_matches = {
            worker.worker_id: self._prefix_match(request, worker.worker_id)
            for worker in self.registry.snapshots()
        }
        candidates = list(
            self.registry.candidates(
                required_blocks=required_blocks,
                prompt_tokens=len(request.token_ids),
                prefix_matches=prefix_matches,
            )
        )

        # Dynamic H1+nD: an idle hybrid worker and the permanent decode workers
        # compete in one load-aware short-request pool.  The old P-first path is
        # intentionally retained through ``_pick_serve_prefill_worker`` as a
        # helper/fallback, but it is no longer allowed to pack every short onto
        # GPU0 before considering the D pool.
        force_pd = int(
            getattr(getattr(self.router, "config", None), "force_pd_tokens", 0) or 0
        )
        short_cutoff = int(getattr(self.config, "conditional_pd_tokens", 0) or force_pd)
        if (
            getattr(self.config, "prefill_short_policy", "work-conserving")
            == "work-conserving"
            and short_cutoff
            and len(request.token_ids) < short_cutoff
            and self._prefill_available()
        ):
            decode_load = min(
                (candidate.decode_load for candidate in candidates), default=1.0
            )
            index = self._pick_serve_prefill_worker(
                required_blocks=required_blocks,
                competing_decode_load=decode_load,
                decode_candidates=len(candidates),
                decode_candidate_ids=tuple(
                    candidate.worker_id for candidate in candidates
                ),
                preferred_index=(
                    request.worker_id if request.worker_pool == "prefill" else None
                ),
            )
            if index is not None:
                try:
                    try:
                        result = self._prefill_serving_rpc(
                            index,
                            self._request_command("reserve", request),
                            "admission",
                            request.request_id,
                        )
                    except (TimeoutError, WorkerUnavailableError):
                        result = {}
                finally:
                    self._release_prefill_dispatch_claim(index)
                if result.get("admitted"):
                    with self._state_lock:
                        self._prefill_bound[request.request_id] = index
                        self._record_prefill_reservation(index, request)
                        self._record_prefill_load(
                            index,
                            request,
                            work=self._request_work(request),
                            prefill_tokens=len(request.token_ids),
                        )
                        self._prefill_short_count = (
                            getattr(self, "_prefill_short_count", 0) + 1
                        )
                    request.route = "collocated"
                    request.route_reason = "prefill_worker_collocated"
                    request.worker_id = index
                    request.worker_pool = "prefill"
                    return AdmissionDecision.accept()
        if self._should_defer_long_for_prefill(request):
            self._note_hybrid_long_pressure()
            return AdmissionDecision.defer(
                "hybrid prefill slot or token budget is busy"
            )
        if not candidates:
            total_blocks = self._total_blocks_per_worker()
            if required_blocks > total_blocks:
                return AdmissionDecision.reject(
                    f"request needs {required_blocks} KV blocks, worker capacity is {total_blocks}"
                )
            return AdmissionDecision.defer("all decode workers are temporarily full")

        candidates = self._order_candidates(candidates, request=request)
        last_retryable = None
        # Health alone is insufficient: a healthy Hybrid already bound to a
        # Long request has no second physical prefill slot.  Treat it as
        # unavailable so another Long can fall back to collocated execution
        # instead of waiting behind the same H worker indefinitely.
        prefill_available = (
            self._hybrid_prefill_slot_available()
            and self._pd_prefill_budget_available(request)
        )
        for candidate in candidates:
            try:
                admitted = self._reserve_on(candidate.worker_id, request)
            except (TimeoutError, WorkerUnavailableError) as exc:
                last_retryable = str(exc)
                continue
            if not admitted.admitted:
                if admitted.retryable:
                    last_retryable = admitted.reason
                    continue
                continue
            self.registry.bind(request.request_id, candidate.worker_id)
            request.worker_id = candidate.worker_id
            request.worker_pool = "decode"
            with self._state_lock:
                conditional_pd_tokens = getattr(self.config, "conditional_pd_tokens", 0)
                if conditional_pd_tokens:
                    if len(request.token_ids) < conditional_pd_tokens:
                        decision = RouteDecision(
                            Route.COLLOCATED,
                            RouteReason.CONDITIONAL_SHORT_COLLOCATED,
                            len(request.token_ids),
                            candidate.decode_load,
                            True,
                        )
                    elif not prefill_available:
                        decision = RouteDecision(
                            Route.COLLOCATED,
                            (
                                RouteReason.HYBRID_QUEUE_OVERFLOW
                                if self._prefill_available()
                                else RouteReason.PREFILL_UNAVAILABLE
                            ),
                            len(request.token_ids),
                            candidate.decode_load,
                            True,
                        )
                    else:
                        decision = RouteDecision(
                            Route.PD_DISAGGREGATED,
                            RouteReason.CONDITIONAL_LONG_PD,
                            len(request.token_ids),
                            candidate.decode_load,
                            True,
                        )
                elif prefill_available:
                    with self._state_lock:
                        busy = sum(
                            1 for pending in self._prefill_pending if pending > 0
                        )
                    prefill_load = busy / len(self._prefill_pending)
                    decision = self.router.decide(
                        len(request.token_ids),
                        candidate.decode_load,
                        True,
                        request.route_prefill_queue_ahead_ms,
                        prefill_load=prefill_load,
                    )
                else:
                    decision = RouteDecision(
                        Route.COLLOCATED,
                        RouteReason.PREFILL_UNAVAILABLE,
                        len(request.token_ids),
                        candidate.decode_load,
                        True,
                    )
                if decision.route is Route.PD_DISAGGREGATED:
                    try:
                        prefill_index = self._bind_long_prefill(request)
                    except WorkerUnavailableError:
                        # Health can change between candidate scoring and the
                        # role claim. Keep the already-reserved decode state and
                        # fall back to the legacy collocated path instead of
                        # leaking the admission or failing the request.
                        decision = RouteDecision(
                            Route.COLLOCATED,
                            (
                                RouteReason.HYBRID_QUEUE_OVERFLOW
                                if self._prefill_available()
                                else RouteReason.PREFILL_UNAVAILABLE
                            ),
                            len(request.token_ids),
                            candidate.decode_load,
                            True,
                        )
                        prefill_index = None
                    else:
                        self._record_prefill_load(
                            prefill_index,
                            request,
                            work=max(1, len(request.token_ids)),
                            prefill_tokens=len(request.token_ids),
                        )
                if decision.route is Route.PD_DISAGGREGATED:
                    decode_work = max(1, request.max_new_tokens)
                    decode_prefill_tokens = 0
                else:
                    decode_work = self._request_work(request)
                    decode_prefill_tokens = len(request.token_ids)
                self._record_decode_load(
                    candidate.worker_id,
                    request,
                    work=decode_work,
                    prefill_tokens=decode_prefill_tokens,
                )
                self._route_decisions[request.request_id] = decision
                request.route = decision.route.value
                request.route_reason = decision.reason.value
                request.route_collocated_cost_ms = decision.collocated_cost_ms
                request.route_pd_cost_ms = decision.pd_cost_ms
                request.route_estimated_savings_ms = decision.estimated_savings_ms
                request.route_cost_confidence = decision.cost_model_confidence
                request.route_decode_load = decision.decode_load
                request.route_prefill_load = decision.prefill_load
                request.route_prefill_queue_ahead_ms = decision.prefill_queue_ahead_ms
            return AdmissionDecision.accept()
        return AdmissionDecision.defer(
            last_retryable or "all decode workers rejected the reservation"
        )

    @staticmethod
    def _request_work(request: ServingRequest) -> int:
        return len(request.token_ids) + max(1, request.max_new_tokens)

    def _order_candidates(self, candidates, *, request: ServingRequest | None = None):
        """Reorder decode-worker candidates per ``--pd-schedule``.

        ``load-aware`` keeps the registry's composite score order (the pre-change
        default). ``kv-aware`` prefers the worker with the fewest occupied KV
        blocks. ``round-robin`` rotates so a different worker leads each admission.
        """
        schedule = self.config.pd_schedule
        if schedule in ("load-aware", ""):
            with self._state_lock:
                self._ensure_hybrid_state()
                start = self._decode_round_robin % max(1, len(self.config.decode_devices))

                def score(candidate):
                    worker_id = candidate.worker_id
                    return (
                        self._decode_prefill_tokens[worker_id] > 0,
                        self._decode_prefill_tokens[worker_id],
                        self._decode_assigned_work[worker_id],
                        candidate.decode_load,
                        candidate.score,
                        (worker_id - start) % max(1, len(self.config.decode_devices)),
                    )

                ordered = sorted(candidates, key=score)
                if request is not None and ordered:
                    self._decode_round_robin = (ordered[0].worker_id + 1) % max(
                        1, len(self.config.decode_devices)
                    )
                return ordered
        if schedule == "kv-aware":
            used = {
                snapshot.worker_id: snapshot.capacity.kv_total_blocks
                - snapshot.capacity.kv_free_blocks
                for snapshot in self.registry.snapshots()
            }
            return sorted(
                candidates,
                key=lambda c: (used.get(c.worker_id, 0), c.decode_load, c.worker_id),
            )
        with self._state_lock:
            offset = self._decode_round_robin % max(1, len(candidates))
            self._decode_round_robin += 1
        if offset == 0:
            return list(candidates)
        return list(candidates[offset:]) + list(candidates[:offset])

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        admitted = self.admit(request)
        if not admitted.admitted:
            raise MemoryError(admitted.reason or "request cannot be admitted")
        started = monotonic()
        if request.request_id in self._prefill_bound:
            index = self._prefill_bound[request.request_id]
            result = self._prefill_serving_rpc(
                index,
                self._request_command("collocated_prepare", request),
                "collocated_prepare",
                request.request_id,
            )
            with self._state_lock:
                self._collocated_count += 1
                self._mark_prefill_load_prefill_complete(request.request_id)
            sample = result.get("sample")
            return (
                sample if isinstance(sample, TokenSample) else int(result["token_id"])
            )
        worker_id = self.registry.worker_for(request.request_id)
        decision = self.route_for(request.request_id)
        if decision.route is Route.COLLOCATED:
            token_id = self._collocated_prefill(worker_id, request)
            self._observe_route_cost(request, decision, started)
            with self._state_lock:
                self._collocated_count += 1
                self._mark_decode_load_prefill_complete(request.request_id)
            return token_id
        command = self._request_command("prefill", request)
        command["worker_index"] = worker_id
        host_prefix_tokens = getattr(self, "_host_prefix_tokens", {}).get(
            request.request_id, 0
        )
        host_cache_hit = host_prefix_tokens == len(request.token_ids)
        command["streamed_transfer"] = (
            not host_cache_hit
            and os.environ.get("HYDRASERVE_CHUNKED_TRANSFER", "1") != "0"
        )
        command["host_cache_hit"] = host_cache_hit
        command["host_prefix_tokens"] = host_prefix_tokens
        # Arm the D-side receiver before the P worker can publish streamed KV.
        # Queue dispatch and receiver readiness are separate phases: only the
        # prepare executor thread emits ``prepare_armed``. The serving loop
        # already runs this method in a bounded P executor, so keep only the
        # receiver in the auxiliary pool and execute the producer in this
        # physical P slot after both handshakes complete.
        receiver_dispatched = Event()
        receiver_armed = Event()
        prepare_future = self._pd_executor.submit(
            self._decode_rpc,
            worker_id,
            {
                **command,
                "op": "prepare",
                "timeout": self.operation_timeout,
            },
            "prepare",
            request.request_id,
            dispatched=receiver_dispatched,
            receiver_armed=receiver_armed,
        )
        _pd_protocol_trace(
            "coordinator_prepare_dispatched",
            request.request_id,
            worker_index=worker_id,
        )
        try:
            dispatch_timeout = getattr(
                self, "receiver_dispatch_timeout", min(5.0, self.operation_timeout)
            )
            if not receiver_dispatched.wait(dispatch_timeout):
                if prepare_future.done():
                    prepare_future.result()
                raise TimeoutError("decode prepare was not dispatched before PD transfer")
            arm_timeout = getattr(
                self, "receiver_arm_timeout", min(10.0, self.operation_timeout)
            )
            if not receiver_armed.wait(arm_timeout):
                if prepare_future.done():
                    prepare_future.result()
                raise TimeoutError("decode receiver was not armed before PD transfer")
            _pd_protocol_trace(
                "coordinator_receiver_armed",
                request.request_id,
                worker_index=worker_id,
            )
            _pd_protocol_trace(
                "coordinator_prefill_started",
                request.request_id,
                worker_index=worker_id,
            )
            prefill_future = self._pd_executor.submit(
                self._prefill_rpc, command, request.request_id
            )
            done, _ = wait(
                (prefill_future, prepare_future),
                return_when=FIRST_COMPLETED,
            )
            if prepare_future in done:
                prepared = prepare_future.result()
            if prefill_future in done:
                result = prefill_future.result()
            result = prefill_future.result()
            _pd_protocol_trace(
                "coordinator_prefill_finished",
                request.request_id,
                worker_index=worker_id,
            )
            prepared = prepare_future.result()
            _pd_protocol_trace(
                "coordinator_prepare_finished",
                request.request_id,
                worker_index=worker_id,
            )
        except Exception:
            if not prepare_future.done():
                try:
                    self._decode_rpc(
                        worker_id,
                        {
                            "op": "cancel_prepare",
                            "request_id": request.request_id,
                        },
                        "cancel_prepare",
                        request.request_id,
                    )
                except Exception:
                    # Preserve the original transfer/prepare failure. Worker
                    # liveness handling in _decode_rpc records cancellation
                    # transport failures independently.
                    pass
            raise
        if result.get("worker_index") != worker_id:
            raise RuntimeError("prefill worker returned a different decode target")
        if result["token_id"] != prepared["token_id"]:
            raise RuntimeError("prefill/decode first-token mismatch")
        if not prepared.get("replay_consistent", True):
            with self._state_lock:
                self._replay_mismatches += 1
        self._observe_route_cost(request, decision, started)
        with self._state_lock:
            self._pd_count += 1
            self._mark_prefill_load_prefill_complete(request.request_id)
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(prepared["token_id"])

    def _observe_route_cost(
        self,
        request: ServingRequest,
        decision: RouteDecision,
        started: float,
    ) -> None:
        elapsed_ms = (monotonic() - started) * 1000.0
        request.route_observed_prefill_service_ms = elapsed_ms
        observe = getattr(self.router, "observe", None)
        if observe is not None:
            observe(
                decision.route,
                decision.prompt_tokens,
                elapsed_ms,
                decision.decode_load,
                decision.prefill_queue_ahead_ms,
            )

    def decode(
        self, requests: tuple[ServingRequest, ...]
    ) -> tuple[int | TokenSample, ...]:
        if not requests:
            return ()
        groups: dict[int, list[tuple[int, ServingRequest]]] = {}
        prefill_groups: dict[int, list[tuple[int, ServingRequest]]] = {}
        failures: dict[int, BaseException] = {}
        for position, request in enumerate(requests):
            prefill_index = self._prefill_bound.get(request.request_id)
            if prefill_index is not None:
                prefill_groups.setdefault(prefill_index, []).append((position, request))
                continue
            try:
                worker_id = self.registry.worker_for(request.request_id)
            except KeyError:
                with self._state_lock:
                    lost = request.request_id in self._lost_requests
                if not lost:
                    raise
                failures[request.request_id] = WorkerStateLostError(
                    f"request {request.request_id} lost decode-worker state"
                )
                continue
            groups.setdefault(worker_id, []).append((position, request))
        output: list[int | TokenSample] = [0] * len(requests)

        def execute(worker_id, indexed_requests):
            request_ids = tuple(item.request_id for _, item in indexed_requests)
            result = self._decode_rpc(
                worker_id,
                {"op": "decode", "request_ids": request_ids},
                "decode",
            )
            if tuple(result["request_ids"]) != request_ids:
                raise RuntimeError("decode worker returned a different request batch")
            samples = result.get("samples")
            if samples is not None:
                return indexed_requests, tuple(samples)
            return indexed_requests, tuple(int(token) for token in result["token_ids"])

        def execute_prefill(index, indexed_requests):
            request_ids = tuple(item.request_id for _, item in indexed_requests)
            result = self._prefill_serving_rpc(
                index,
                {"op": "decode", "request_ids": request_ids},
                "decode",
            )
            if tuple(result["request_ids"]) != request_ids:
                raise RuntimeError("prefill worker returned a different request batch")
            samples = result.get("samples")
            if samples is not None:
                return indexed_requests, tuple(samples)
            return indexed_requests, tuple(int(token) for token in result["token_ids"])

        all_groups = [
            ("decode", worker_id, tuple(indexed))
            for worker_id, indexed in groups.items()
        ] + [
            ("prefill", index, tuple(indexed))
            for index, indexed in prefill_groups.items()
        ]
        successes: dict[int, int | TokenSample] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(all_groups))) as executor:
            futures = {}
            for kind, worker_key, indexed in all_groups:
                fn = execute if kind == "decode" else execute_prefill
                futures[executor.submit(fn, worker_key, indexed)] = indexed
            for future, scheduled in futures.items():
                try:
                    indexed, token_ids = future.result()
                    if len(indexed) != len(token_ids):
                        raise RuntimeError(
                            "decode worker output count does not match its request group"
                        )
                    for (position, request), token_id in zip(
                        indexed, token_ids, strict=True
                    ):
                        output[position] = token_id
                        successes[request.request_id] = token_id
                except Exception as exc:
                    for _, request in scheduled:
                        failures[request.request_id] = exc
        if failures:
            raise PartialDecodeError(successes, failures)
        return tuple(output)

    def decode_batch_sizes(
        self, requests: tuple[ServingRequest, ...]
    ) -> dict[int, int]:
        """Report the batch width seen by each physical P/D worker."""

        groups: dict[tuple[str, int], list[int]] = {}
        with self._state_lock:
            prefill_bound = dict(self._prefill_bound)
        for request in requests:
            prefill_index = prefill_bound.get(request.request_id)
            if prefill_index is not None:
                groups.setdefault(("prefill", prefill_index), []).append(
                    request.request_id
                )
                continue
            try:
                worker_id = self.registry.worker_for(request.request_id)
            except KeyError:
                continue
            groups.setdefault(("decode", worker_id), []).append(request.request_id)
        return {
            request_id: len(request_ids)
            for request_ids in groups.values()
            for request_id in request_ids
        }

    def release(self, request_id: int) -> None:
        prefill_index = self._prefill_bound.get(request_id)
        if prefill_index is not None:
            try:
                self._prefill_serving_rpc(
                    prefill_index,
                    {"op": "release", "request_id": request_id},
                    "release",
                    request_id,
                )
            finally:
                with self._state_lock:
                    self._prefill_bound.pop(request_id, None)
                    self._ensure_hybrid_state()
                    self._prefill_reserved_blocks[prefill_index].pop(request_id, None)
                    self._release_prefill_load(request_id)
                    self._route_decisions.pop(request_id, None)
                    self._lost_requests.discard(request_id)
            return
        try:
            worker_id = self.registry.worker_for(request_id)
        except KeyError:
            with self._state_lock:
                self._lost_requests.discard(request_id)
                self._route_decisions.pop(request_id, None)
            return
        try:
            self._decode_rpc(
                worker_id,
                {"op": "release", "request_id": request_id},
                "release",
                request_id,
            )
        finally:
            self.registry.release(request_id)
            with self._state_lock:
                self._reserved_blocks[worker_id].pop(request_id, None)
                self._release_decode_load(request_id)
                self._release_prefill_load(request_id)
                getattr(self, "_host_prefix_tokens", {}).pop(request_id, None)
                self._route_decisions.pop(request_id, None)
                self._lost_requests.discard(request_id)

    def abandon(self, request_id: int) -> None:
        """Forget host-side ownership after device-local state is known lost."""
        if request_id in self._prefill_bound:
            with self._state_lock:
                prefill_index = self._prefill_bound.pop(request_id, None)
                self._ensure_hybrid_state()
                if prefill_index is not None:
                    self._prefill_reserved_blocks[prefill_index].pop(request_id, None)
                self._release_prefill_load(request_id)
                self._route_decisions.pop(request_id, None)
                self._lost_requests.discard(request_id)
            return
        self._release_long_prefill_binding(request_id)
        worker_id = self.registry.release(request_id)
        with self._state_lock:
            if worker_id is not None:
                self._reserved_blocks[worker_id].pop(request_id, None)
                self._release_decode_load(request_id)
            else:
                for reservations in self._reserved_blocks:
                    reservations.pop(request_id, None)
                self._release_decode_load(request_id)
            self._release_prefill_load(request_id)
            self._route_decisions.pop(request_id, None)
            self._lost_requests.discard(request_id)

    def is_recoverable_decode_error(
        self, request_id: int, error: BaseException
    ) -> bool:
        if isinstance(
            error, (TimeoutError, WorkerUnavailableError, WorkerStateLostError)
        ):
            return True
        with self._state_lock:
            return request_id in self._lost_requests

    def preempt(self, request_id: int) -> None:
        self.release(request_id)

    def recover(self, request: ServingRequest) -> AdmissionDecision:
        decision = self.admit(request)
        if not decision.admitted:
            return decision
        command = {
            **self._request_command("recover", request),
            "generated_token_ids": tuple(request.generated_token_ids),
            "replay_token_ids": request.token_ids
            + tuple(request.generated_token_ids[:-1]),
        }
        try:
            prefill_index = self._prefill_bound.get(request.request_id)
            if prefill_index is not None:
                self._prefill_serving_rpc(
                    prefill_index,
                    command,
                    "recover",
                    request.request_id,
                )
            else:
                worker_id = self.registry.worker_for(request.request_id)
                self._decode_rpc(
                    worker_id,
                    command,
                    "recover",
                    request.request_id,
                )
            return AdmissionDecision.accept()
        except Exception:
            try:
                self.release(request.request_id)
            except Exception:
                pass
            raise

    def capacity(self) -> BackendCapacity:
        snapshots = self.registry.snapshots()
        hybrid = []
        if (
            getattr(self.config, "prefill_short_policy", "work-conserving")
            == "work-conserving"
        ):
            with self._state_lock:
                self._ensure_hybrid_state()
                reserve_blocks = self._hybrid_reserve_blocks()
                hybrid = [
                    BackendCapacity(
                        max(0, capacity.kv_total_blocks - reserve_blocks),
                        max(0, capacity.kv_free_blocks - reserve_blocks),
                        capacity.state_total_slots,
                        capacity.state_free_slots,
                    )
                    for index, capacity in enumerate(self._prefill_capacities)
                    if self._prefill_healthy[index]
                    and self._hybrid_roles[index] is HybridRole.DECODE
                ]
        capacities = [item.capacity for item in snapshots] + hybrid
        return BackendCapacity(
            kv_total_blocks=sum(item.kv_total_blocks for item in capacities),
            kv_free_blocks=sum(item.kv_free_blocks for item in capacities),
            state_total_slots=sum(item.state_total_slots for item in capacities),
            state_free_slots=sum(item.state_free_slots for item in capacities),
        )

    def cache_stats(self) -> dict[str, int | float]:
        with self._state_lock:
            keys = {key for worker in self._worker_cache_stats for key in worker}
            return {
                key: sum(
                    float(worker.get(key, 0)) for worker in self._worker_cache_stats
                )
                for key in keys
            }

    def worker_for(self, request_id: int) -> int:
        return self.registry.worker_for(request_id)

    def route_for(self, request_id: int) -> RouteDecision:
        with self._state_lock:
            try:
                return self._route_decisions[request_id]
            except KeyError as exc:
                raise KeyError(f"request {request_id} has no bound route") from exc

    def routing_stats(self) -> RoutingStats:
        with self._state_lock:
            return RoutingStats(
                self._collocated_count,
                self._pd_count,
                self._pd_failures,
                any(self._prefill_healthy),
                getattr(self, "_prefill_short_count", 0),
                getattr(self, "_prefill_chunk_preemptions", 0),
            )

    def routing_cost_stats(self):
        stats = getattr(self.router, "stats", None)
        return None if stats is None else stats()

    def reset_routing_calibration(self) -> None:
        reset = getattr(self.router, "reset_online_state", None)
        if reset is not None:
            reset()

    def recovery_stats(self) -> WorkerRecoveryStats:
        snapshots = self.registry.snapshots()
        with self._state_lock:
            return WorkerRecoveryStats(
                len(snapshots),
                sum(worker.healthy for worker in snapshots),
                self._recovery_attempts,
                self._recovery_successes,
                self._recovery_failures,
                tuple(sorted(self._recovering_workers)),
            )

    def prefill_recovery_stats(self) -> PrefillRecoveryStats:
        self._prefill_available()
        with self._state_lock:
            return PrefillRecoveryStats(
                all(self._prefill_healthy),
                sum(self._prefill_recovery_attempts),
                sum(self._prefill_recovery_successes),
                sum(self._prefill_recovery_failures),
                any(self._prefill_recovering),
            )

    def transfer_validation_stats(self):
        from hydraserve.engine.pd_service import TransferValidationStats

        with self._state_lock:
            return TransferValidationStats(self._replay_mismatches)

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._recovery_stop.set()
        for commands in self._prefill_commands:
            commands.put({"op": "shutdown"})
        for commands in self._decode_commands:
            commands.put({"op": "shutdown"})
        if not force:
            for responses in self._prefill_responses:
                try:
                    self._get(responses, 10.0)
                except Exception:
                    pass
            for responses in self._decode_responses:
                try:
                    self._get(responses, 10.0)
                except Exception:
                    pass
        for process in (*self._prefill_processes, *self._decode_processes):
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
        if self._bootstrap_server is not None:
            self._bootstrap_server.close()
            self._bootstrap_server = None
        self._pd_executor.shutdown(wait=not force, cancel_futures=force)

    def _reserve_on(self, worker_id: int, request: ServingRequest) -> AdmissionDecision:
        result = self._decode_rpc(
            worker_id,
            self._request_command("reserve", request),
            "admission",
            request.request_id,
        )
        if result.get("admitted"):
            with self._state_lock:
                self._reserved_blocks[worker_id][request.request_id] = (
                    self._required_blocks(request)
                )
                getattr(self, "_host_prefix_tokens", {})[request.request_id] = int(
                    result.get("host_prefix_tokens", 0)
                )
            return AdmissionDecision.accept()
        if result.get("retryable"):
            return AdmissionDecision.defer(
                result.get("reason", "decode worker is temporarily full")
            )
        return AdmissionDecision.reject(
            result.get("reason", "request exceeds decode worker capacity")
        )

    def _collocated_prefill(
        self, worker_id: int, request: ServingRequest
    ) -> int | TokenSample:
        result = self._decode_rpc(
            worker_id,
            self._request_command("collocated_prepare", request),
            "collocated_prepare",
            request.request_id,
        )
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(result["token_id"])

    def _decode_rpc(
        self,
        worker_id: int,
        command: dict,
        expected_op: str,
        request_id: int | None = None,
        *,
        dispatched: Event | None = None,
        receiver_armed: Event | None = None,
    ) -> dict:
        self._ensure_decode_rpc_state()
        rpc_id = next(self._decode_rpc_ids)
        # Prepare is a two-response RPC: the D worker first confirms that its
        # background receive thread is running, then returns the installed
        # state.  Keep this queue unbounded so a fast final response cannot
        # race the coordinator while the intermediate acknowledgement is
        # still waiting to be consumed.
        waiter: Queue = Queue()
        failure = None
        try:
            # Protect process replacement and queue handoff only.  Waiting for
            # a background prepare must not prevent decode/release commands
            # from reaching the same D worker.
            with self._decode_locks[worker_id]:
                if not self._decode_processes[worker_id].is_alive():
                    raise WorkerUnavailableError(
                        f"decode worker {worker_id} is not running"
                    )
                with self._state_lock:
                    self._decode_waiters[worker_id][rpc_id] = waiter
                try:
                    self._decode_commands[worker_id].put(
                        {**command, "rpc_id": rpc_id}
                    )
                except Exception:
                    with self._state_lock:
                        self._decode_waiters[worker_id].pop(rpc_id, None)
                    raise
                if dispatched is not None:
                    dispatched.set()

            deadline = monotonic() + self.operation_timeout
            while True:
                try:
                    result = waiter.get_nowait()
                except Empty:
                    result = None
                if result is not None:
                    if result.get("op") == "prepare_armed":
                        if expected_op != "prepare":
                            raise RuntimeError(
                                "received prepare_armed for a non-prepare RPC"
                            )
                        if (
                            request_id is not None
                            and result.get("request_id") != request_id
                        ):
                            raise RuntimeError(
                                "decode receiver armed a different request"
                            )
                        if receiver_armed is not None:
                            receiver_armed.set()
                        continue
                    break
                if not self._decode_processes[worker_id].is_alive():
                    raise WorkerUnavailableError(
                        f"decode worker {worker_id} exited during RPC"
                    )
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for decode worker")
                acquired = self._decode_response_locks[worker_id].acquire(
                    timeout=min(0.05, remaining)
                )
                if not acquired:
                    continue
                try:
                    try:
                        response = self._decode_responses[worker_id].get(
                            timeout=min(0.05, remaining)
                        )
                    except Empty:
                        continue
                    response_id = response.get("rpc_id")
                    with self._state_lock:
                        target = self._decode_waiters[worker_id].get(response_id)
                        if (
                            response_id is None
                            and len(self._decode_waiters[worker_id]) == 1
                        ):
                            target = waiter
                    if target is not None:
                        target.put_nowait(response)
                finally:
                    self._decode_response_locks[worker_id].release()
        except (TimeoutError, WorkerUnavailableError) as exc:
            failure = exc
        finally:
            with self._state_lock:
                self._decode_waiters[worker_id].pop(rpc_id, None)
        if failure is not None:
            self._invalidate_worker(worker_id)
            self._schedule_decode_recovery(worker_id)
            raise failure
        self._check(result, expected_op, request_id)
        self._update_worker_capacity(worker_id, result)
        return result

    def _ensure_decode_rpc_state(self) -> None:
        """Initialize correlated D-worker RPC state for old test doubles."""

        count_workers = len(self._decode_processes)
        if not hasattr(self, "_decode_response_locks"):
            self._decode_response_locks = [Lock() for _ in range(count_workers)]
        if not hasattr(self, "_decode_waiters"):
            self._decode_waiters = [{} for _ in range(count_workers)]
        if not hasattr(self, "_decode_rpc_ids"):
            self._decode_rpc_ids = count(1)

    def _pick_prefill_worker(self) -> int:
        with self._state_lock:
            count = len(self._prefill_processes)
            claims = getattr(self, "_prefill_dispatch_claims", [0] * count)
            candidates = [
                index for index in range(count) if self._prefill_healthy[index]
            ]
            if not candidates:
                return self._prefill_round_robin % count
            # Least-loaded dispatch: the worker with the fewest in-flight
            # commands gets the next long prefill, keeping the nP pool balanced.
            index = min(
                candidates,
                key=lambda i: (self._prefill_pending[i] + claims[i], i),
            )
            self._prefill_round_robin = (index + 1) % count
            return index

    def _claim_prefill_worker(self) -> int:
        """Atomically select and reserve a P worker for dispatch."""

        with self._state_lock:
            count = len(self._prefill_processes)
            if not hasattr(self, "_prefill_dispatch_claims"):
                self._prefill_dispatch_claims = [0] * count
            candidates = [
                index for index in range(count) if self._prefill_healthy[index]
            ]
            if not candidates:
                candidates = list(range(count))
            index = min(
                candidates,
                key=lambda i: (
                    self._prefill_pending[i] + self._prefill_dispatch_claims[i],
                    i,
                ),
            )
            self._prefill_dispatch_claims[index] += 1
            self._prefill_round_robin = (index + 1) % count
            return index

    def _release_prefill_dispatch_claim(self, index: int) -> None:
        with self._state_lock:
            claims = getattr(self, "_prefill_dispatch_claims", None)
            if claims is not None:
                claims[index] = max(0, claims[index] - 1)

    def _ensure_hybrid_state(self) -> None:
        """Lazily initialize H-worker state for old test doubles/checkpoints."""

        count_workers = len(self._prefill_processes)
        total_blocks = self._total_blocks_per_worker()
        state_slots = int(getattr(self.config, "max_state_slots_per_worker", 1))
        if not hasattr(self, "_prefill_reserved_blocks"):
            self._prefill_reserved_blocks = [dict() for _ in range(count_workers)]
        if not hasattr(self, "_prefill_capacities"):
            self._prefill_capacities = [
                BackendCapacity(total_blocks, total_blocks, state_slots, state_slots)
                for _ in range(count_workers)
            ]
        if not hasattr(self, "_hybrid_roles"):
            self._hybrid_roles = [HybridRole.DECODE] * count_workers
        if not hasattr(self, "_prefill_long_inflight"):
            self._prefill_long_inflight = [0] * count_workers
        if not hasattr(self, "_prefill_dispatch_claims"):
            self._prefill_dispatch_claims = [0] * count_workers
        if not hasattr(self, "_long_prefill_bound"):
            self._long_prefill_bound = {}
        if not hasattr(self, "_short_round_robin"):
            self._short_round_robin = 0
        if not hasattr(self, "_decode_assigned_work"):
            self._decode_assigned_work = [0] * len(self.config.decode_devices)
        if not hasattr(self, "_decode_prefill_tokens"):
            self._decode_prefill_tokens = [0] * len(self.config.decode_devices)
        if not hasattr(self, "_decode_request_loads"):
            self._decode_request_loads = {}
        if not hasattr(self, "_prefill_assigned_work"):
            self._prefill_assigned_work = [0] * count_workers
        if not hasattr(self, "_prefill_prefill_tokens"):
            self._prefill_prefill_tokens = [0] * count_workers
        if not hasattr(self, "_prefill_request_loads"):
            self._prefill_request_loads = {}
        if not hasattr(self, "_hybrid_long_pressure_until"):
            self._hybrid_long_pressure_until = 0.0
        if not hasattr(self, "_hybrid_short_gate_closures"):
            self._hybrid_short_gate_closures = 0

    def _record_decode_load(
        self,
        worker_id: int,
        request: ServingRequest,
        *,
        work: int,
        prefill_tokens: int,
    ) -> None:
        self._ensure_hybrid_state()
        old = self._decode_request_loads.pop(request.request_id, None)
        if old is not None:
            old_worker, old_work, old_prefill = old
            self._decode_assigned_work[old_worker] = max(
                0, self._decode_assigned_work[old_worker] - old_work
            )
            self._decode_prefill_tokens[old_worker] = max(
                0, self._decode_prefill_tokens[old_worker] - old_prefill
            )
        work = max(1, int(work))
        prefill_tokens = max(0, int(prefill_tokens))
        self._decode_request_loads[request.request_id] = (
            worker_id,
            work,
            prefill_tokens,
        )
        self._decode_assigned_work[worker_id] += work
        self._decode_prefill_tokens[worker_id] += prefill_tokens

    def _record_prefill_load(
        self,
        index: int,
        request: ServingRequest,
        *,
        work: int,
        prefill_tokens: int,
    ) -> None:
        self._ensure_hybrid_state()
        old = self._prefill_request_loads.pop(request.request_id, None)
        if old is not None:
            old_index, old_work, old_prefill = old
            self._prefill_assigned_work[old_index] = max(
                0, self._prefill_assigned_work[old_index] - old_work
            )
            self._prefill_prefill_tokens[old_index] = max(
                0, self._prefill_prefill_tokens[old_index] - old_prefill
            )
        work = max(1, int(work))
        prefill_tokens = max(0, int(prefill_tokens))
        self._prefill_request_loads[request.request_id] = (
            index,
            work,
            prefill_tokens,
        )
        self._prefill_assigned_work[index] += work
        self._prefill_prefill_tokens[index] += prefill_tokens

    def _mark_decode_load_prefill_complete(self, request_id: int) -> None:
        load = self._decode_request_loads.get(request_id)
        if load is None:
            return
        worker_id, work, prefill_tokens = load
        if prefill_tokens <= 0:
            return
        self._decode_prefill_tokens[worker_id] = max(
            0, self._decode_prefill_tokens[worker_id] - prefill_tokens
        )
        remaining = max(0, work - prefill_tokens)
        self._decode_assigned_work[worker_id] = max(
            0, self._decode_assigned_work[worker_id] - prefill_tokens
        )
        self._decode_request_loads[request_id] = (worker_id, remaining, 0)

    def _mark_prefill_load_prefill_complete(self, request_id: int) -> None:
        load = self._prefill_request_loads.get(request_id)
        if load is None:
            return
        index, work, prefill_tokens = load
        if prefill_tokens <= 0:
            return
        self._prefill_prefill_tokens[index] = max(
            0, self._prefill_prefill_tokens[index] - prefill_tokens
        )
        remaining = max(0, work - prefill_tokens)
        self._prefill_assigned_work[index] = max(
            0, self._prefill_assigned_work[index] - prefill_tokens
        )
        self._prefill_request_loads[request_id] = (index, remaining, 0)

    def _release_decode_load(self, request_id: int) -> None:
        load = self._decode_request_loads.pop(request_id, None)
        if load is None:
            return
        worker_id, work, prefill_tokens = load
        self._decode_assigned_work[worker_id] = max(
            0, self._decode_assigned_work[worker_id] - work
        )
        self._decode_prefill_tokens[worker_id] = max(
            0, self._decode_prefill_tokens[worker_id] - prefill_tokens
        )

    def _release_prefill_load(self, request_id: int) -> None:
        load = self._prefill_request_loads.pop(request_id, None)
        if load is None:
            return
        index, work, prefill_tokens = load
        self._prefill_assigned_work[index] = max(
            0, self._prefill_assigned_work[index] - work
        )
        self._prefill_prefill_tokens[index] = max(
            0, self._prefill_prefill_tokens[index] - prefill_tokens
        )

    def _hybrid_reserve_blocks(self) -> int:
        configured = getattr(
            self.config, "effective_hybrid_prefill_reserve_tokens", None
        )
        if configured is None:
            configured = getattr(self.config, "hybrid_prefill_reserve_tokens", 0)
        tokens = max(0, int(configured))
        return (tokens + self.config.block_size - 1) // self.config.block_size

    def _record_prefill_reservation(self, index: int, request: ServingRequest) -> None:
        self._ensure_hybrid_state()
        self._prefill_reserved_blocks[index][request.request_id] = (
            self._required_blocks(request)
        )

    def _update_prefill_capacity(self, index: int, result: dict) -> None:
        required = {
            "kv_total_blocks",
            "kv_free_blocks",
            "state_total_slots",
            "state_free_slots",
        }
        if not required.issubset(result):
            return
        capacity = BackendCapacity(
            int(result["kv_total_blocks"]),
            int(result["kv_free_blocks"]),
            int(result["state_total_slots"]),
            int(result["state_free_slots"]),
        )
        with self._state_lock:
            self._ensure_hybrid_state()
            if not hasattr(self, "_prefill_capacity_versions"):
                self._prefill_capacity_versions = [-1] * len(
                    self._prefill_processes
                )
            version = int(result.get("response_sequence", -1))
            if version >= 0 and version < self._prefill_capacity_versions[index]:
                return
            self._prefill_capacities[index] = capacity
            if version >= 0:
                self._prefill_capacity_versions[index] = version

    def _pd_prefill_outstanding_tokens(self) -> int:
        self._ensure_hybrid_state()
        return int(sum(self._prefill_prefill_tokens))

    def _pd_prefill_budget_available(self, request: ServingRequest) -> bool:
        """Return whether adding this Long keeps the P/Hybrid token queue bounded."""

        budget = int(getattr(self.config, "pd_prefill_token_budget", 0) or 0)
        if budget <= 0:
            return True
        outstanding = self._pd_prefill_outstanding_tokens()
        return outstanding + len(request.token_ids) <= budget

    def _should_defer_long_for_prefill(self, request: ServingRequest) -> bool:
        """Token-aware Long admission guard for dynamic Hybrid topologies.

        If a Long request cannot currently claim a Hybrid/P slot or would push
        the P side over its prompt-token budget, keep it in the serving-loop
        admission queue until the configured overflow grace expires.  This
        avoids immediately falling back to D-bound collocated execution and
        reintroducing the Short decode pollution that H1 is meant to isolate.
        """

        threshold = int(getattr(self.config, "conditional_pd_tokens", 0) or 0)
        if threshold <= 0 or len(request.token_ids) < threshold:
            return False
        if not self._prefill_available():
            return False
        if self._long_overflow_ready(request) or self._idle_decode_slot_available(request):
            return False
        return not (
            self._hybrid_prefill_slot_available()
            and self._pd_prefill_budget_available(request)
        )

    def _bind_long_prefill(self, request: ServingRequest | int) -> int:
        """Move one hybrid worker to PENDING before async prefill dispatch."""

        request_id = request.request_id if isinstance(request, ServingRequest) else int(request)
        with self._state_lock:
            self._ensure_hybrid_state()
            existing = self._long_prefill_bound.get(request_id)
            if existing is not None:
                return existing
            if isinstance(
                request, ServingRequest
            ) and not self._pd_prefill_budget_available(request):
                raise WorkerUnavailableError("hybrid prefill token budget is full")
            candidates = [
                index
                for index, healthy in enumerate(self._prefill_healthy)
                if healthy and self._hybrid_roles[index] is HybridRole.DECODE
            ]
            if not candidates:
                raise WorkerUnavailableError("no healthy hybrid prefill worker")
            index = min(
                candidates,
                key=lambda candidate: (
                    self._prefill_prefill_tokens[candidate] > 0,
                    self._prefill_prefill_tokens[candidate],
                    self._prefill_assigned_work[candidate],
                    self._prefill_pending[candidate],
                    candidate,
                ),
            )
            self._long_prefill_bound[request_id] = index
            self._hybrid_roles[index] = HybridRole.PREFILL_PENDING
            return index

    def _hybrid_prefill_slot_available(self) -> bool:
        """Return whether a healthy Hybrid can accept a new Long now.

        Existing short requests bound to a decode-role Hybrid may continue and
        are serviced at chunk boundaries.  A Hybrid already PENDING/ACTIVE for
        another Long is not counted as another queue slot.
        """

        if not self._prefill_available():
            return False
        with self._state_lock:
            self._ensure_hybrid_state()
            return any(
                healthy and role is HybridRole.DECODE
                for healthy, role in zip(
                    self._prefill_healthy, self._hybrid_roles, strict=True
                )
            )

    def _long_overflow_ready(self, request: ServingRequest) -> bool:
        """Allow D-side fallback only after the Hybrid queue grace period."""

        wait_ms = max(
            0.0,
            float(getattr(self.config, "hybrid_long_overflow_ms", 5000.0)),
        )
        return (monotonic() - request.submitted_at) * 1000.0 >= wait_ms

    def _note_hybrid_long_pressure(self) -> None:
        """Temporarily reserve decode-role Hybrid workers for queued Long work.

        A Long request can be deferred before it binds a concrete Hybrid worker
        because all P slots are busy or the P-side token budget is full.  During
        that gap, an idle Hybrid should not immediately accept fresh Short
        collocated work, otherwise the queued Long loses the first available P
        opportunity and H1 degrades back toward a noisy DP baseline.
        """

        hold_ms = float(getattr(self.config, "hybrid_long_pressure_hold_ms", 0.0) or 0.0)
        if hold_ms <= 0:
            return
        with self._state_lock:
            self._ensure_hybrid_state()
            self._hybrid_long_pressure_until = max(
                self._hybrid_long_pressure_until,
                monotonic() + hold_ms / 1000.0,
            )

    def _hybrid_long_pressure_active(self) -> bool:
        hold_ms = float(getattr(self.config, "hybrid_long_pressure_hold_ms", 0.0) or 0.0)
        if hold_ms <= 0:
            return False
        self._ensure_hybrid_state()
        return monotonic() < self._hybrid_long_pressure_until

    def _idle_decode_slot_available(self, request: ServingRequest) -> bool:
        """Prefer immediate overflow only when it consumes an actually idle D."""

        required_blocks = self._required_blocks(request)
        return any(
            worker.healthy
            and worker.capacity.decode_load == 0.0
            and worker.capacity.kv_free_blocks >= required_blocks
            and worker.capacity.state_free_slots > 0
            for worker in self.registry.snapshots()
        )

    def _release_long_prefill_binding(self, request_id: int) -> None:
        with self._state_lock:
            self._ensure_hybrid_state()
            index = self._long_prefill_bound.pop(request_id, None)
            if index is None:
                return
            still_bound = any(
                candidate == index for candidate in self._long_prefill_bound.values()
            )
            if not still_bound and self._prefill_long_inflight[index] == 0:
                self._hybrid_roles[index] = HybridRole.DECODE

    @property
    def hybrid_role_states(self) -> tuple[str, ...]:
        with self._state_lock:
            self._ensure_hybrid_state()
            return tuple(role.value for role in self._hybrid_roles)

    def _pick_serve_prefill_worker(
        self,
        *,
        required_blocks: int = 1,
        competing_decode_load: float = 1.0,
        decode_candidates: int = 0,
        decode_candidate_ids: tuple[int, ...] = (),
        preferred_index: int | None = None,
    ) -> int | None:
        """Pick a decode-role hybrid worker for a collocated short request.

        Hybrid and permanent-D workers are compared by live decode load.  A
        worker in PENDING/ACTIVE never accepts a new short, while shorts already
        bound to it continue through cooperative chunk-boundary preemption.
        """
        with self._state_lock:
            self._ensure_hybrid_state()
            count = len(self._prefill_processes)
            long_inflight = getattr(self, "_prefill_long_inflight", [0] * count)
            if not hasattr(self, "_prefill_dispatch_claims"):
                self._prefill_dispatch_claims = [0] * count
            reserve_blocks = self._hybrid_reserve_blocks()
            idle = [
                index
                for index in range(count)
                if self._prefill_healthy[index]
                and self._hybrid_roles[index] is HybridRole.DECODE
                and long_inflight[index] == 0
                and self._prefill_dispatch_claims[index] == 0
                and self._prefill_capacities[index].state_free_slots > 0
                and (
                    self._prefill_capacities[index].kv_free_blocks - reserve_blocks
                    >= required_blocks
                )
            ]
            if not idle:
                return None
            if self._hybrid_long_pressure_active():
                self._hybrid_short_gate_closures += 1
                return None
            backlog_budget = int(
                getattr(
                    self.config,
                    "hybrid_short_max_prefill_backlog_tokens",
                    0,
                )
                or 0
            )
            assigned_budget = int(
                getattr(self.config, "hybrid_short_max_assigned_work", 0) or 0
            )
            if backlog_budget and all(
                self._prefill_prefill_tokens[candidate] > backlog_budget
                for candidate in idle
            ):
                return None
            if assigned_budget and all(
                self._prefill_assigned_work[candidate] > assigned_budget
                for candidate in idle
            ):
                return None
            index = (
                preferred_index
                if preferred_index in idle
                else min(
                    idle,
                    key=lambda candidate: (
                        self._prefill_capacities[candidate].decode_load,
                        candidate,
                    ),
                )
            )
            pending_load = min(
                1.0,
                self._prefill_pending[index]
                / max(1, self._prefill_capacities[index].state_total_slots),
            )
            hybrid_load = max(
                self._prefill_capacities[index].decode_load, pending_load
            )
            hybrid_score = (
                self._prefill_prefill_tokens[index] > 0,
                self._prefill_prefill_tokens[index],
                self._prefill_assigned_work[index],
                self._prefill_pending[index] + self._prefill_dispatch_claims[index],
                hybrid_load,
            )
            best_decode_score = (
                self._best_decode_short_score(decode_candidate_ids)
                if decode_candidates > 0
                else (True, 1 << 60, 1 << 60, 1 << 60, 1.0)
            )
            total_candidates = max(1, len(idle) + decode_candidates)
            tie_turn = self._short_round_robin % total_candidates
            self._short_round_robin += 1
            if preferred_index not in idle and (
                hybrid_load > competing_decode_load
                or hybrid_score > best_decode_score
                or (hybrid_load == competing_decode_load and tie_turn >= len(idle))
            ):
                return None
            self._prefill_dispatch_claims[index] += 1
            self._prefill_serve_round_robin = (index + 1) % count
            return index

    def _best_decode_short_score(
        self, candidate_ids: tuple[int, ...] = ()
    ) -> tuple[bool, int, int, int, float]:
        allowed = set(candidate_ids)
        snapshots = tuple(
            worker
            for worker in self.registry.snapshots()
            if worker.healthy and (not allowed or worker.worker_id in allowed)
        )
        if not snapshots:
            return (True, 1 << 60, 1 << 60, 1 << 60, 1.0)
        return min(
            (
                self._decode_prefill_tokens[worker.worker_id] > 0,
                self._decode_prefill_tokens[worker.worker_id],
                self._decode_assigned_work[worker.worker_id],
                worker.active_requests,
                worker.capacity.decode_load,
            )
            for worker in snapshots
        )

    def _prefill_rpc(self, command: dict, request_id: int) -> dict:
        self._ensure_hybrid_state()
        bound_index = self._long_prefill_bound.get(request_id)
        if bound_index is None:
            # Compatibility path for callers admitted before dynamic role
            # binding existed. Keep the legacy least-loaded selector rather
            # than removing support for those requests.
            index = self._claim_prefill_worker()
            release_claim = True
        else:
            index = bound_index
            release_claim = False
        failure = None
        try:
            with self._state_lock:
                self._hybrid_roles[index] = HybridRole.PREFILL_ACTIVE
            try:
                result = self._prefill_rpc_call(index, command, long_operation=True)
            except (TimeoutError, WorkerUnavailableError) as exc:
                failure = exc
        finally:
            if release_claim:
                self._release_prefill_dispatch_claim(index)
                with self._state_lock:
                    self._hybrid_roles[index] = HybridRole.DECODE
            else:
                self._release_long_prefill_binding(request_id)
        if failure is not None:
            with self._state_lock:
                if self._prefill_healthy[index]:
                    self._pd_failures += 1
                self._prefill_healthy[index] = False
            self._schedule_prefill_recovery(index)
            raise failure
        self._check(result, "prefill", request_id)
        self._update_prefill_capacity(index, result)
        with self._state_lock:
            self._prefill_chunk_preemptions = getattr(
                self, "_prefill_chunk_preemptions", 0
            ) + int(result.get("chunk_preemptions", 0))
        return result

    def _prefill_serving_rpc(
        self, index: int, command: dict, expected_op: str, request_id: int | None = None
    ) -> dict:
        """RPC for a prefill worker's collocated-serving operations (W4)."""
        failure = None
        try:
            result = self._prefill_rpc_call(index, command, long_operation=False)
        except (TimeoutError, WorkerUnavailableError) as exc:
            failure = exc
        if failure is not None:
            with self._state_lock:
                if self._prefill_healthy[index]:
                    self._pd_failures += 1
                self._prefill_healthy[index] = False
            self._schedule_prefill_recovery(index)
            raise failure
        self._check(result, expected_op, request_id)
        self._update_prefill_capacity(index, result)
        return result

    def _ensure_prefill_rpc_state(self) -> None:
        """Lazily initialize correlation state for lightweight test doubles."""

        count_workers = len(self._prefill_processes)
        if not hasattr(self, "_prefill_response_locks"):
            self._prefill_response_locks = [Lock() for _ in range(count_workers)]
        if not hasattr(self, "_prefill_waiters"):
            self._prefill_waiters = [{} for _ in range(count_workers)]
        if not hasattr(self, "_prefill_rpc_ids"):
            self._prefill_rpc_ids = count(1)
        if not hasattr(self, "_prefill_long_inflight"):
            self._prefill_long_inflight = [0 for _ in range(count_workers)]
        if not hasattr(self, "_prefill_dispatch_claims"):
            self._prefill_dispatch_claims = [0 for _ in range(count_workers)]

    def _prefill_rpc_call(
        self, index: int, command: dict, *, long_operation: bool
    ) -> dict:
        """Submit a correlated RPC without holding a worker-wide response lock.

        Any caller may drain the shared response queue and forwards a response
        to the waiter identified by ``rpc_id``.  Consequently a short decode
        can complete while the original thread is still waiting for a long
        prefill response.
        """

        self._ensure_prefill_rpc_state()
        rpc_id = next(self._prefill_rpc_ids)
        waiter: Queue = Queue(maxsize=1)
        # Serialize only the process/queue handoff with recovery.  The lock is
        # released immediately after put(), never held while waiting for the
        # response, so unrelated short RPCs remain concurrent.
        with self._prefill_locks[index]:
            if not self._prefill_processes[index].is_alive():
                raise WorkerUnavailableError(f"prefill worker {index} is not running")
            with self._state_lock:
                self._prefill_waiters[index][rpc_id] = waiter
                self._prefill_pending[index] += 1
                if long_operation:
                    self._prefill_long_inflight[index] += 1
            try:
                self._prefill_commands[index].put({**command, "rpc_id": rpc_id})
            except Exception:
                with self._state_lock:
                    self._prefill_waiters[index].pop(rpc_id, None)
                    self._prefill_pending[index] = max(
                        0, self._prefill_pending[index] - 1
                    )
                    if long_operation:
                        self._prefill_long_inflight[index] = max(
                            0, self._prefill_long_inflight[index] - 1
                        )
                raise
        deadline = monotonic() + self.operation_timeout
        try:
            while True:
                try:
                    return waiter.get_nowait()
                except Empty:
                    pass
                if not self._prefill_processes[index].is_alive():
                    raise WorkerUnavailableError(
                        f"prefill worker {index} exited during RPC"
                    )
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for prefill worker")
                acquired = self._prefill_response_locks[index].acquire(
                    timeout=min(0.05, remaining)
                )
                if not acquired:
                    continue
                try:
                    try:
                        response = self._prefill_responses[index].get(
                            timeout=min(0.05, remaining)
                        )
                    except Empty:
                        continue
                    response_id = response.get("rpc_id")
                    with self._state_lock:
                        target = self._prefill_waiters[index].get(response_id)
                        # Compatibility for unit-test queues and old workers:
                        # an untagged response is safe only with one waiter.
                        if (
                            response_id is None
                            and len(self._prefill_waiters[index]) == 1
                        ):
                            target = waiter
                    if target is not None:
                        target.put_nowait(response)
                finally:
                    self._prefill_response_locks[index].release()
        finally:
            with self._state_lock:
                self._prefill_waiters[index].pop(rpc_id, None)
                self._prefill_pending[index] = max(0, self._prefill_pending[index] - 1)
                if long_operation:
                    self._prefill_long_inflight[index] = max(
                        0, self._prefill_long_inflight[index] - 1
                    )

    def _prefill_available(self) -> bool:
        with self._state_lock:
            healthy = any(self._prefill_healthy)
            closed = self._closed
        if not healthy or closed:
            return healthy and not closed
        for index, process in enumerate(self._prefill_processes):
            if self._prefill_healthy[index] and not process.is_alive():
                with self._state_lock:
                    if self._prefill_healthy[index]:
                        self._prefill_healthy[index] = False
                        self._pd_failures += 1
                self._schedule_prefill_recovery(index)
        with self._state_lock:
            return any(self._prefill_healthy) and not closed

    def _get_prefill_response(self, index: int, timeout: float):
        deadline = monotonic() + timeout
        while True:
            if not self._prefill_processes[index].is_alive():
                raise WorkerUnavailableError(
                    f"prefill worker {index} exited during RPC"
                )
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for prefill worker")
            try:
                return self._prefill_responses[index].get(timeout=min(0.1, remaining))
            except Empty:
                continue

    def _schedule_prefill_recovery(self, index: int) -> None:
        with self._state_lock:
            if self._closed or self._prefill_recovering[index]:
                return
            self._prefill_recovering[index] = True
            thread = Thread(
                target=self._recover_prefill_worker,
                args=(index,),
                name=f"hydraserve-recover-prefill-{index}",
                daemon=True,
            )
            self._prefill_recovery_threads[index] = thread
        thread.start()

    def _recover_prefill_worker(self, index: int) -> None:
        try:
            for attempt in range(self.max_worker_restarts):
                if self._recovery_stop.is_set():
                    return
                with self._state_lock:
                    self._prefill_recovery_attempts[index] += 1
                try:
                    self._restart_prefill_worker_once(index)
                except Exception:
                    with self._state_lock:
                        self._prefill_recovery_failures[index] += 1
                    delay = self.worker_restart_backoff_s * (2**attempt)
                    if self._recovery_stop.wait(delay):
                        return
                    continue
                if self._recovery_stop.is_set():
                    return
                with self._state_lock:
                    self._prefill_healthy[index] = True
                    self._prefill_recovery_successes[index] += 1
                return
        finally:
            with self._state_lock:
                self._prefill_recovering[index] = False
                self._prefill_recovery_threads[index] = None

    def _restart_prefill_worker_once(self, index: int) -> None:
        with self._prefill_locks[index]:
            if self._recovery_stop.is_set():
                return
            previous = self._prefill_processes[index]
            if previous.is_alive():
                previous.terminate()
            previous.join(10)
            self._prefill_commands[index] = self._context.Queue()
            self._prefill_responses[index] = self._context.Queue()
            process = self._new_prefill_process(index)
            self._prefill_processes[index] = process
            process.start()
            result = self._get_prefill_response(index, self.startup_timeout)
            self._check(result, "ready")
            if result.get("model_name") != self.model_name:
                raise RuntimeError("restarted prefill worker loaded a different model")

    def _invalidate_worker(self, worker_id: int) -> tuple[int, ...]:
        self.registry.set_health(worker_id, False)
        request_ids = self.registry.release_worker(worker_id)
        with self._state_lock:
            self._lost_requests.update(request_ids)
            self._reserved_blocks[worker_id].clear()
            for request_id in request_ids:
                self._route_decisions.pop(request_id, None)
        return request_ids

    def _get_decode_response(self, worker_id: int, timeout: float):
        deadline = monotonic() + timeout
        while True:
            if not self._decode_processes[worker_id].is_alive():
                raise WorkerUnavailableError(
                    f"decode worker {worker_id} exited during RPC"
                )
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for cluster worker")
            try:
                return self._decode_responses[worker_id].get(
                    timeout=min(0.1, remaining)
                )
            except Empty:
                continue

    def _schedule_decode_recovery(self, worker_id: int) -> None:
        with self._state_lock:
            if self._closed or worker_id in self._recovering_workers:
                return
            self._recovering_workers.add(worker_id)
            thread = Thread(
                target=self._recover_decode_worker,
                args=(worker_id,),
                name=f"hydraserve-recover-decode-{worker_id}",
                daemon=True,
            )
            self._recovery_threads[worker_id] = thread
        thread.start()

    def _recover_decode_worker(self, worker_id: int) -> None:
        try:
            for attempt in range(self.max_worker_restarts):
                if self._recovery_stop.is_set():
                    return
                with self._state_lock:
                    self._recovery_attempts += 1
                try:
                    self._restart_decode_worker_once(worker_id)
                except Exception:
                    with self._state_lock:
                        self._recovery_failures += 1
                    delay = self.worker_restart_backoff_s * (2**attempt)
                    if self._recovery_stop.wait(delay):
                        return
                    continue
                if self._recovery_stop.is_set():
                    return
                with self._state_lock:
                    self._recovery_successes += 1
                return
        finally:
            with self._state_lock:
                self._recovering_workers.discard(worker_id)
                self._recovery_threads.pop(worker_id, None)

    def _restart_decode_worker_once(self, worker_id: int) -> None:
        with self._decode_locks[worker_id]:
            if self._recovery_stop.is_set():
                return
            previous = self._decode_processes[worker_id]
            if previous.is_alive():
                previous.terminate()
            previous.join(10)
            self._decode_commands[worker_id] = self._context.Queue()
            self._decode_responses[worker_id] = self._context.Queue()
            process = self._new_decode_process(worker_id)
            self._decode_processes[worker_id] = process
            process.start()
            result = self._get_decode_response(worker_id, self.startup_timeout)
            self._check(result, "ready")
            if result.get("model_name") != self.model_name:
                raise RuntimeError("restarted decode worker loaded a different model")
            self._update_worker_capacity(worker_id, result)
            with self._state_lock:
                self._reserved_blocks[worker_id].clear()
            self.registry.set_health(worker_id, True)

    def _new_decode_process(self, worker_id: int):
        return self._context.Process(
            target=_decode_worker,
            args=(
                self.config.worker_config(worker_id),
                self._namespaces[worker_id],
                self._decode_commands[worker_id],
                self._decode_responses[worker_id],
                worker_id,
                self._worker_log_path("decode", worker_id),
                self._bootstrap_address,
            ),
            name=f"hydraserve-decode-{worker_id}",
        )

    def _new_prefill_process(self, index: int):
        return self._context.Process(
            target=_prefill_worker,
            args=(
                self.config.prefill_config(index),
                self._namespaces,
                self._prefill_commands[index],
                self._prefill_responses[index],
                self._worker_log_path("prefill", index),
                self._bootstrap_address,
            ),
            name=f"hydraserve-prefill-{index}",
        )

    def _worker_log_path(self, kind: str, index: int) -> str | None:
        if not self.config.worker_log_dir:
            return None
        return str(Path(self.config.worker_log_dir) / f"{kind}-{index}.log")

    def _prefix_match(self, request: ServingRequest, worker_id: int) -> int:
        if self.prefix_affinity is not None:
            return max(0, int(self.prefix_affinity(request, worker_id)))
        if not self.config.prefix_cache_blocks:
            return 0
        try:
            result = self._decode_rpc(
                worker_id,
                {
                    "op": "prefix_probe",
                    "request_id": request.request_id,
                    "token_ids": request.token_ids,
                },
                "prefix_probe",
                request.request_id,
            )
        except (TimeoutError, WorkerUnavailableError):
            return 0
        return max(0, int(result.get("matched_tokens", 0)))

    def _update_worker_capacity(self, worker_id: int, result: dict) -> None:
        keys = (
            "kv_total_blocks",
            "kv_free_blocks",
            "state_total_slots",
            "state_free_slots",
        )
        version = int(result.get("response_sequence", -1))
        with self._state_lock:
            if not hasattr(self, "_decode_capacity_versions"):
                self._decode_capacity_versions = [-1] * len(self._decode_processes)
            if version >= 0 and version < self._decode_capacity_versions[worker_id]:
                return
            if version >= 0:
                self._decode_capacity_versions[worker_id] = version
        if all(key in result for key in keys):
            self.registry.update_capacity(
                worker_id,
                BackendCapacity(*(int(result[key]) for key in keys)),
            )
        cache_stats = result.get("kv_cache_stats")
        if isinstance(cache_stats, dict):
            with self._state_lock:
                self._worker_cache_stats[worker_id] = {
                    str(key): value
                    for key, value in cache_stats.items()
                    if isinstance(value, (int, float))
                }

    def _required_blocks(self, request: ServingRequest) -> int:
        total_tokens = len(request.token_ids) + max(0, request.max_new_tokens - 1)
        return (total_tokens + self.config.block_size - 1) // self.config.block_size

    def _total_blocks_per_worker(self) -> int:
        return (
            self.config.cache_tokens_per_worker + self.config.block_size - 1
        ) // self.config.block_size - self.config.kv_headroom_blocks

    @staticmethod
    def _request_command(op: str, request: ServingRequest) -> dict:
        return {
            "op": op,
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
        }

    @staticmethod
    def _get(queue, timeout: float):
        try:
            return queue.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("timed out waiting for cluster worker") from exc

    @staticmethod
    def _check(result, expected_op: str, request_id: int | None = None) -> None:
        if result.get("op") in {"error", "startup_error"}:
            raise RuntimeError(result.get("message", "cluster worker failed"))
        if result.get("op") != expected_op:
            raise RuntimeError(
                f"expected worker response {expected_op!r}, got {result.get('op')!r}"
            )
        if request_id is not None and result.get("request_id") != request_id:
            raise RuntimeError("worker returned a different request")
