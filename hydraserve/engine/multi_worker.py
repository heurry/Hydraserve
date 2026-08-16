"""One-prefill, many-decode worker generation backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from threading import Event, Lock, RLock, Thread
from time import monotonic
from typing import Callable
from uuid import uuid4

from hydraserve.engine.pd_service import (
    PDWorkerConfig,
    RoutingStats,
    _decode_worker,
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
    worker_log_dir: str = ""

    def __post_init__(self) -> None:
        if not self.model_dir or not self.decode_devices:
            raise ValueError("cluster requires a model and decode workers")
        if len(set(self.decode_devices)) != len(self.decode_devices):
            raise ValueError("decode devices must be unique")
        if not self.prefill_devices:
            object.__setattr__(self, "prefill_devices", (self.prefill_device,))
        if len(set(self.prefill_devices)) != len(self.prefill_devices):
            raise ValueError("prefill devices must be unique")
        if set(self.prefill_devices) & set(self.decode_devices):
            raise ValueError("prefill and decode devices must be distinct")
        if min(
            self.cache_tokens_per_worker,
            self.block_size,
            self.max_state_slots_per_worker,
            self.max_decode_batch_size_per_worker,
            self.prefill_chunk_size,
            self.prefix_cache_min_frequency,
        ) <= 0:
            raise ValueError("cluster resource limits must be positive")
        if self.prefix_cache_blocks < 0:
            raise ValueError("prefix cache blocks cannot be negative")
        total_blocks = (
            self.cache_tokens_per_worker + self.block_size - 1
        ) // self.block_size
        if not 0 <= self.kv_headroom_blocks < total_blocks:
            raise ValueError("KV headroom must be below physical cache blocks")
        if self.topologies and len(self.topologies) != len(self.decode_devices):
            raise ValueError("topologies must match decode devices")

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
    def prefill_parallelism(self) -> int:
        """Concurrent prefill slots the serving loop may drive in parallel."""
        return len(self.config.prefill_devices)

    def __init__(
        self,
        config: PDClusterConfig,
        *,
        router: AdaptiveRouter | None = None,
        prefix_affinity: PrefixAffinity | None = None,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
        max_worker_restarts: int = 3,
        worker_restart_backoff_s: float = 0.5,
    ) -> None:
        if max_worker_restarts <= 0 or worker_restart_backoff_s < 0:
            raise ValueError("invalid decode worker recovery policy")
        self.config = config
        self.router = router or CostAwareRouter()
        self.prefix_affinity = prefix_affinity
        self.operation_timeout = operation_timeout
        self.startup_timeout = startup_timeout
        self.max_worker_restarts = max_worker_restarts
        self.worker_restart_backoff_s = worker_restart_backoff_s
        self.namespace = f"hydraserve-cluster-{uuid4().hex}"
        worker_count = len(config.decode_devices)
        self._namespaces = tuple(
            f"{self.namespace}-decode-{index}" for index in range(worker_count)
        )
        self._context = mp.get_context("spawn")
        if config.worker_log_dir:
            Path(config.worker_log_dir).mkdir(parents=True, exist_ok=True)
        prefill_count = len(config.prefill_devices)
        self._prefill_commands = [self._context.Queue() for _ in range(prefill_count)]
        self._prefill_responses = [self._context.Queue() for _ in range(prefill_count)]
        self._prefill_locks = [Lock() for _ in range(prefill_count)]
        self._decode_commands = [self._context.Queue() for _ in range(worker_count)]
        self._decode_responses = [self._context.Queue() for _ in range(worker_count)]
        self._decode_locks = [Lock() for _ in range(worker_count)]
        self._decode_processes = [
            self._new_decode_process(index)
            for index in range(worker_count)
        ]
        self._prefill_processes = [
            self._new_prefill_process(index)
            for index in range(prefill_count)
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
        self._worker_cache_stats: list[dict[str, int | float]] = [
            {} for _ in range(worker_count)
        ]
        self._route_decisions: dict[int, RouteDecision] = {}
        self._lost_requests: set[int] = set()
        self._state_lock = RLock()
        self._prefill_healthy = [True] * prefill_count
        self._prefill_pending = [0] * prefill_count
        self._closed = False
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
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
        # request_id -> prefill worker index for collocated requests served on a
        # prefill worker (W4); decode-worker-bound requests stay in the registry.
        self._prefill_bound: dict[int, int] = {}
        self._replay_mismatches = 0

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
        except Exception:
            self.close(force=True)
            raise

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        try:
            self.registry.worker_for(request.request_id)
            return AdmissionDecision.accept()
        except KeyError:
            pass
        # W4: short requests that will be routed collocated can be served on an
        # idle prefill worker (using its otherwise-idle decode-phase compute)
        # instead of a decode worker. Long requests always stay on the PD path.
        force_pd = int(getattr(getattr(self.router, "config", None), "force_pd_tokens", 0) or 0)
        if (
            force_pd
            and len(request.token_ids) < force_pd
            and self._prefill_available()
        ):
            index = self._pick_serve_prefill_worker()
            if index is not None:
                try:
                    result = self._prefill_serving_rpc(
                        index,
                        self._request_command("reserve", request),
                        "admission",
                        request.request_id,
                    )
                except (TimeoutError, WorkerUnavailableError):
                    result = {}
                if result.get("admitted"):
                    self._prefill_bound[request.request_id] = index
                    request.route = "collocated"
                    request.route_reason = "prefill_worker_collocated"
                    request.worker_id = index
                    return AdmissionDecision.accept()
        required_blocks = self._required_blocks(request)
        prefix_matches = {
            worker.worker_id: self._prefix_match(request, worker.worker_id)
            for worker in self.registry.snapshots()
        }
        candidates = self.registry.candidates(
            required_blocks=required_blocks,
            prompt_tokens=len(request.token_ids),
            prefix_matches=prefix_matches,
        )
        if not candidates:
            total_blocks = self._total_blocks_per_worker()
            if required_blocks > total_blocks:
                return AdmissionDecision.reject(
                    f"request needs {required_blocks} KV blocks, worker capacity is {total_blocks}"
                )
            return AdmissionDecision.defer("all decode workers are temporarily full")

        last_retryable = None
        prefill_available = self._prefill_available()
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
            with self._state_lock:
                if prefill_available:
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
                self._route_decisions[request.request_id] = decision
                request.route = decision.route.value
                request.route_reason = decision.reason.value
                request.route_collocated_cost_ms = decision.collocated_cost_ms
                request.route_pd_cost_ms = decision.pd_cost_ms
                request.route_estimated_savings_ms = decision.estimated_savings_ms
                request.route_cost_confidence = decision.cost_model_confidence
                request.route_decode_load = decision.decode_load
                request.route_prefill_load = decision.prefill_load
                request.route_prefill_queue_ahead_ms = (
                    decision.prefill_queue_ahead_ms
                )
            return AdmissionDecision.accept()
        return AdmissionDecision.defer(
            last_retryable or "all decode workers rejected the reservation"
        )

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
            sample = result.get("sample")
            return sample if isinstance(sample, TokenSample) else int(result["token_id"])
        worker_id = self.registry.worker_for(request.request_id)
        decision = self.route_for(request.request_id)
        if decision.route is Route.COLLOCATED:
            token_id = self._collocated_prefill(worker_id, request)
            self._observe_route_cost(request, decision, started)
            with self._state_lock:
                self._collocated_count += 1
            return token_id
        command = self._request_command("prefill", request)
        command["worker_index"] = worker_id
        result = self._prefill_rpc(command, request.request_id)
        if result.get("worker_index") != worker_id:
            raise RuntimeError("prefill worker returned a different decode target")
        prepared = self._decode_rpc(
            worker_id,
            {
                **command,
                "op": "prepare",
                "timeout": self.operation_timeout,
            },
            "prepare",
            request.request_id,
        )
        if result["token_id"] != prepared["token_id"]:
            raise RuntimeError("prefill/decode first-token mismatch")
        if not prepared.get("replay_consistent", True):
            with self._state_lock:
                self._replay_mismatches += 1
        self._observe_route_cost(request, decision, started)
        with self._state_lock:
            self._pd_count += 1
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

    def release(self, request_id: int) -> None:
        prefill_index = self._prefill_bound.pop(request_id, None)
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
                    self._route_decisions.pop(request_id, None)
                    self._lost_requests.discard(request_id)
            return
        worker_id = self.registry.release(request_id)
        if worker_id is None:
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
            with self._state_lock:
                self._reserved_blocks[worker_id].pop(request_id, None)
                self._route_decisions.pop(request_id, None)
                self._lost_requests.discard(request_id)

    def abandon(self, request_id: int) -> None:
        """Forget host-side ownership after device-local state is known lost."""
        if request_id in self._prefill_bound:
            self._prefill_bound.pop(request_id, None)
            with self._state_lock:
                self._route_decisions.pop(request_id, None)
                self._lost_requests.discard(request_id)
            return
        worker_id = self.registry.release(request_id)
        with self._state_lock:
            if worker_id is not None:
                self._reserved_blocks[worker_id].pop(request_id, None)
            else:
                for reservations in self._reserved_blocks:
                    reservations.pop(request_id, None)
            self._route_decisions.pop(request_id, None)
            self._lost_requests.discard(request_id)

    def is_recoverable_decode_error(
        self, request_id: int, error: BaseException
    ) -> bool:
        if isinstance(error, (TimeoutError, WorkerUnavailableError, WorkerStateLostError)):
            return True
        with self._state_lock:
            return request_id in self._lost_requests

    def preempt(self, request_id: int) -> None:
        self.release(request_id)

    def recover(self, request: ServingRequest) -> AdmissionDecision:
        decision = self.admit(request)
        if not decision.admitted:
            return decision
        worker_id = self.registry.worker_for(request.request_id)
        command = {
            **self._request_command("recover", request),
            "generated_token_ids": tuple(request.generated_token_ids),
            "replay_token_ids": request.token_ids
            + tuple(request.generated_token_ids[:-1]),
        }
        try:
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
        return BackendCapacity(
            kv_total_blocks=sum(item.capacity.kv_total_blocks for item in snapshots),
            kv_free_blocks=sum(item.capacity.kv_free_blocks for item in snapshots),
            state_total_slots=sum(item.capacity.state_total_slots for item in snapshots),
            state_free_slots=sum(item.capacity.state_free_slots for item in snapshots),
        )

    def cache_stats(self) -> dict[str, int | float]:
        with self._state_lock:
            keys = {
                key for worker in self._worker_cache_stats for key in worker
            }
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

    def _reserve_on(
        self, worker_id: int, request: ServingRequest
    ) -> AdmissionDecision:
        result = self._decode_rpc(
            worker_id,
            self._request_command("reserve", request),
            "admission",
            request.request_id,
        )
        if result.get("admitted"):
            with self._state_lock:
                self._reserved_blocks[worker_id][request.request_id] = self._required_blocks(
                    request
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
    ) -> dict:
        failure = None
        with self._decode_locks[worker_id]:
            try:
                if not self._decode_processes[worker_id].is_alive():
                    raise WorkerUnavailableError(
                        f"decode worker {worker_id} is not running"
                    )
                self._decode_commands[worker_id].put(command)
                result = self._get_decode_response(worker_id, self.operation_timeout)
            except (TimeoutError, WorkerUnavailableError) as exc:
                failure = exc
        if failure is not None:
            self._invalidate_worker(worker_id)
            self._schedule_decode_recovery(worker_id)
            raise failure
        self._check(result, expected_op, request_id)
        self._update_worker_capacity(worker_id, result)
        return result

    def _pick_prefill_worker(self) -> int:
        with self._state_lock:
            count = len(self._prefill_processes)
            for offset in range(count):
                index = (self._prefill_round_robin + offset) % count
                if self._prefill_healthy[index]:
                    self._prefill_round_robin = (index + 1) % count
                    return index
            return self._prefill_round_robin % count

    def _pick_serve_prefill_worker(self) -> int | None:
        """Pick an idle prefill worker for a collocated short request (W4).

        A prefill worker whose command loop is busy running a long prefill must
        be skipped, otherwise the short request blocks behind the multi-second
        prefill on the shared ``_prefill_locks`` and its decode stalls. Returns
        ``None`` when every prefill worker is busy.
        """
        with self._state_lock:
            count = len(self._prefill_processes)
            idle = [
                index
                for index in range(count)
                if self._prefill_healthy[index]
                and not self._prefill_locks[index].locked()
            ]
            if not idle:
                return None
            # Prefer the least-loaded idle worker (fewest in-flight commands);
            # the command queue is shared with PD prefills, so a worker with
            # queued work would stall the short request behind it.
            index = min(idle, key=lambda candidate: self._prefill_pending[candidate])
            self._prefill_serve_round_robin = (index + 1) % count
            return index

    def _prefill_rpc(self, command: dict, request_id: int) -> dict:
        index = self._pick_prefill_worker()
        failure = None
        with self._prefill_locks[index]:
            try:
                if not self._prefill_processes[index].is_alive():
                    raise WorkerUnavailableError(
                        f"prefill worker {index} is not running"
                    )
                with self._state_lock:
                    self._prefill_pending[index] += 1
                self._prefill_commands[index].put(command)
                result = self._get_prefill_response(index, self.operation_timeout)
            except (TimeoutError, WorkerUnavailableError) as exc:
                failure = exc
            finally:
                with self._state_lock:
                    self._prefill_pending[index] -= 1
        if failure is not None:
            with self._state_lock:
                if self._prefill_healthy[index]:
                    self._pd_failures += 1
                self._prefill_healthy[index] = False
            self._schedule_prefill_recovery(index)
            raise failure
        self._check(result, "prefill", request_id)
        return result

    def _prefill_serving_rpc(
        self, index: int, command: dict, expected_op: str, request_id: int | None = None
    ) -> dict:
        """RPC for a prefill worker's collocated-serving operations (W4)."""
        failure = None
        with self._prefill_locks[index]:
            try:
                if not self._prefill_processes[index].is_alive():
                    raise WorkerUnavailableError(
                        f"prefill worker {index} is not running"
                    )
                with self._state_lock:
                    self._prefill_pending[index] += 1
                self._prefill_commands[index].put(command)
                result = self._get_prefill_response(index, self.operation_timeout)
            except (TimeoutError, WorkerUnavailableError) as exc:
                failure = exc
            finally:
                with self._state_lock:
                    self._prefill_pending[index] -= 1
        if failure is not None:
            with self._state_lock:
                if self._prefill_healthy[index]:
                    self._pd_failures += 1
                self._prefill_healthy[index] = False
            self._schedule_prefill_recovery(index)
            raise failure
        self._check(result, expected_op, request_id)
        return result

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
