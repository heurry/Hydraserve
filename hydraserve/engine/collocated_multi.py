"""N-GPU collocated (data-parallel) multi-worker generation backend.

Each GPU runs one ``_prefill_worker`` process that serves the full collocated
request lifecycle (reserve -> collocated_prepare -> decode -> release) on its
own device, exactly like a single-card serve process. The backend coordinator
routes each request to the least-loaded healthy worker, so a benchmark can drive
4xDP engine-only (no HTTP server, no proxy) with a single in-process backend.

This mirrors the W4 collocated path of :class:`MultiWorkerGenerationBackend`
without the prefill/decode disaggregation machinery: there are no separate
decode workers, no PD transfer, and no cost-aware router.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import count
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, RLock
from time import monotonic
from typing import Callable
from uuid import uuid4

from hydraserve.engine.pd_service import PDWorkerConfig, _prefill_worker
from hydraserve.engine.sampling import TokenSample
from hydraserve.engine.serving_loop import (
    AdmissionDecision,
    BackendCapacity,
    PartialDecodeError,
    ServingRequest,
)
from hydraserve.router import WorkerTopology


@dataclass(frozen=True, slots=True)
class CollocatedClusterConfig:
    """One collocated worker per device, all sharing the model checkpoint."""

    model_dir: str
    devices: tuple[str, ...]
    cache_tokens_per_worker: int = 65_536
    block_size: int = 16
    max_state_slots_per_worker: int = 64
    use_flash_attention: bool = True
    prefill_chunk_size: int = 4096
    prefix_cache_blocks: int = 0
    prefix_cache_min_frequency: int = 2
    kv_headroom_blocks: int = 0
    max_decode_batch_size_per_worker: int = 64
    kv_quant: str | None = None
    worker_log_dir: str = ""
    topologies: tuple[WorkerTopology, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_dir or not self.devices:
            raise ValueError("cluster requires a model and at least one device")
        devices = []
        for device in self.devices:
            value = device.strip()
            devices.append(f"cuda:{value}" if value.isdigit() else value)
        object.__setattr__(self, "devices", tuple(devices))
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("collocated devices must be unique")
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
        if self.topologies and len(self.topologies) != len(self.devices):
            raise ValueError("topologies must match devices")

    def worker_config(self, index: int) -> PDWorkerConfig:
        """Collocated worker: prefill and decode live on the same device."""
        return PDWorkerConfig(
            self.model_dir,
            prefill_device=self.devices[index],
            decode_device=self.devices[index],
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
            worker_log_dir=self.worker_log_dir,
        )


class MultiGPUCollocatedBackend:
    """Persistent ND collocated backend with load-aware worker routing."""

    supports_async_prefill = True

    @property
    def supports_independent_decode(self) -> bool:
        """Allow each physical DP worker to advance without a global barrier."""

        return True

    @property
    def decode_executor_parallelism(self) -> int:
        return len(self.config.devices)

    def decode_executor_group(self, request: ServingRequest) -> tuple[str, int]:
        """Return the immutable physical owner of a request's decode state."""

        with self._state_lock:
            return ("collocated", self._bound[request.request_id])

    @property
    def release_parallelism(self) -> int:
        return len(self.config.devices)

    @property
    def prefill_parallelism(self) -> int:
        """Concurrent prefill slots = worker count (each worker collocates)."""
        return len(self.config.devices)

    @property
    def prefill_executor_limits(self) -> dict[str, int]:
        """Expose one running and one chunk-preemptible prefill per worker.

        A physical GPU still executes only one kernel stream at a time.  The
        second host slot exists so a short collocated prepare can reach the
        worker command queue while a long prefill is running; the worker then
        services it at the next chunk boundary.  Deeper host queues would only
        reserve KV early and increase recursive/queueing pressure.
        """

        try:
            queue_depth = min(
                2,
                max(
                    1,
                    int(os.environ.get("HYDRASERVE_DP_PREFILL_QUEUE_DEPTH", "2")),
                ),
            )
        except ValueError:
            queue_depth = 2
        return {
            self._prefill_group(index): queue_depth
            for index in range(len(self.config.devices))
        }

    @staticmethod
    def _prefill_group(index: int) -> str:
        return f"collocated:{index}"

    def prefill_executor_group(self, request: ServingRequest) -> str:
        """Return the immutable worker selected during admission."""

        with self._state_lock:
            return self._prefill_group(self._bound[request.request_id])

    def prefill_executor_group_hint(self, request: ServingRequest) -> str | None:
        """Predict the least-loaded worker before admission without claiming it."""

        try:
            return self._prefill_group(self._select_worker(request, claim=False))
        except LookupError:
            return None

    def prefill_admission_tokens(self, request: ServingRequest) -> int:
        """Charge one executable chunk, not the request's full prompt."""

        return min(len(request.token_ids), self.config.prefill_chunk_size)

    def __init__(
        self,
        config: CollocatedClusterConfig,
        *,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
    ) -> None:
        self.config = config
        self.operation_timeout = operation_timeout
        self.startup_timeout = startup_timeout
        self.namespace = f"hydraserve-dp-{uuid4().hex}"
        worker_count = len(config.devices)
        self._namespaces = tuple(
            f"{self.namespace}-worker-{index}" for index in range(worker_count)
        )
        self._context = mp.get_context("spawn")
        try:
            from hydraserve.transfer import BootstrapServer

            self._bootstrap_server = BootstrapServer().start()
            self._bootstrap_address = self._bootstrap_server.address
        except PermissionError:
            self._bootstrap_server = None
            self._bootstrap_address = None
        if config.worker_log_dir:
            Path(config.worker_log_dir).mkdir(parents=True, exist_ok=True)
        self._commands = [self._context.Queue() for _ in range(worker_count)]
        self._responses = [self._context.Queue() for _ in range(worker_count)]
        # Command locks cover only process/queue handoff. Responses are routed
        # by rpc_id so callers waiting on one worker do not serialize behind a
        # lock for the full duration of a GPU operation.
        self._locks = [Lock() for _ in range(worker_count)]
        self._response_locks = [Lock() for _ in range(worker_count)]
        self._waiters: list[dict[int, Queue]] = [
            {} for _ in range(worker_count)
        ]
        self._rpc_ids = count(1)
        self._processes = [
            self._context.Process(
                target=_prefill_worker,
                args=(
                    config.worker_config(index),
                    self._namespaces[index],
                    self._commands[index],
                    self._responses[index],
                    self._worker_log_path(index),
                    self._bootstrap_address,
                ),
                name=f"hydraserve-collocated-{index}",
            )
            for index in range(worker_count)
        ]
        self._pending = [0] * worker_count
        # Reservations are counted from worker selection until release. RPC
        # depth alone drops back to zero after prefill and makes burst traffic
        # repeatedly choose worker 0.
        self._assigned = [0] * worker_count
        # Request count alone treats a 2K/128 request like an 8K/500 request.
        # Keep simple token-weighted work and outstanding-prefill counters so
        # admission can avoid workers that are about to run expensive prompt
        # work. These are routing estimates, not device-capacity accounting.
        self._assigned_work = [0] * worker_count
        self._prefill_tokens = [0] * worker_count
        self._request_loads: dict[int, tuple[int, int]] = {}
        self._healthy = [True] * worker_count
        self._bound: dict[int, int] = {}
        self._capacity: list[BackendCapacity | None] = [None] * worker_count
        self._capacity_versions = [-1] * worker_count
        self._state_lock = RLock()
        self._closed = False
        self._round_robin = 0
        self._decode_executor = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="hydraserve-dp-decode"
        )

        for process in self._processes:
            process.start()
        try:
            ready = [self._get(queue, startup_timeout) for queue in self._responses]
            for index, result in enumerate(ready):
                try:
                    self._check(result, "ready")
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"collocated worker {index} failed startup: {exc}"
                    ) from exc
            names = {result["model_name"] for result in ready}
            if len(names) != 1:
                raise RuntimeError("cluster workers loaded different models")
            self.model_name = names.pop()
            for index, result in enumerate(ready):
                self._update_capacity(index, result)
        except Exception:
            self.close(force=True)
            raise

    def _worker_log_path(self, index: int) -> str | None:
        if not self.config.worker_log_dir:
            return None
        return str(Path(self.config.worker_log_dir) / f"dp-{index}.log")

    def _update_capacity(self, index: int, result: dict) -> None:
        payload = result.get("capacity")
        if not isinstance(payload, dict):
            # ``_prefill_worker`` returns capacity fields at the top level.
            # Older/test workers may wrap them in ``capacity``; accept both.
            required = {
                "kv_total_blocks",
                "kv_free_blocks",
                "state_total_slots",
                "state_free_slots",
            }
            if not required.issubset(result):
                return
            payload = result
        capacity = BackendCapacity(
            kv_total_blocks=int(payload.get("kv_total_blocks", 0)),
            kv_free_blocks=int(payload.get("kv_free_blocks", 0)),
            state_total_slots=int(payload.get("state_total_slots", 0)),
            state_free_slots=int(payload.get("state_free_slots", 0)),
        )
        with self._state_lock:
            version = int(result.get("response_sequence", -1))
            if version >= 0 and version < self._capacity_versions[index]:
                return
            self._capacity[index] = capacity
            if version >= 0:
                self._capacity_versions[index] = version

    def _required_blocks(self, request: ServingRequest) -> int:
        total_tokens = len(request.token_ids) + max(0, request.max_new_tokens - 1)
        return (total_tokens + self.config.block_size - 1) // self.config.block_size

    @staticmethod
    def _request_work(request: ServingRequest) -> int:
        """Cheap phase-aware routing proxy measured in token work units."""

        return len(request.token_ids) + max(1, request.max_new_tokens)

    def _select_worker(
        self,
        request: ServingRequest | None = None,
        *,
        excluded: set[int] | None = None,
        claim: bool,
    ) -> int:
        """Select the best capacity-compatible worker for one request.

        Outstanding prompt work is the strongest signal because a collocated
        prefill serializes that worker's decode stream. RPC depth and
        token-weighted live work then distinguish workers with the same request
        count. Round-robin remains the deterministic final tie breaker.
        """

        excluded = excluded or set()
        with self._state_lock:
            count = len(self._processes)
            candidates = [
                i for i in range(count) if self._healthy[i] and i not in excluded
            ]
            if not candidates:
                raise LookupError("no healthy collocated worker is available")
            if request is not None:
                required_blocks = self._required_blocks(request)
                candidates = [
                    i
                    for i in candidates
                    if self._capacity[i] is None
                    or (
                        self._capacity[i].kv_free_blocks >= required_blocks
                        and self._capacity[i].state_free_slots > 0
                    )
                ]
                if not candidates:
                    raise LookupError("all collocated workers are temporarily full")
            start = self._round_robin % count

            def score(i: int):
                capacity = self._capacity[i]
                capacity_load = 1.0 if capacity is None else capacity.decode_load
                return (
                    self._prefill_tokens[i] > 0,
                    self._prefill_tokens[i],
                    self._assigned_work[i],
                    self._pending[i],
                    capacity_load,
                    self._assigned[i],
                    (i - start) % count,
                )

            index = min(
                candidates,
                key=score,
            )
            if claim:
                self._assigned[index] += 1
                if request is not None:
                    self._assigned_work[index] += self._request_work(request)
                    self._prefill_tokens[index] += len(request.token_ids)
                self._round_robin = (index + 1) % count
            return index

    def _pick_worker(
        self,
        request: ServingRequest | None = None,
        *,
        excluded: set[int] | None = None,
    ) -> int:
        """Select and claim one worker for request admission."""

        return self._select_worker(request, excluded=excluded, claim=True)

    def _unassign_worker(
        self,
        index: int,
        request: ServingRequest | None = None,
        *,
        load: tuple[int, int] | None = None,
    ) -> None:
        with self._state_lock:
            self._assigned[index] = max(0, self._assigned[index] - 1)
            if request is not None:
                load = (self._request_work(request), len(request.token_ids))
            if load is not None:
                work, outstanding_prefill = load
                self._assigned_work[index] = max(
                    0, self._assigned_work[index] - work
                )
                self._prefill_tokens[index] = max(
                    0, self._prefill_tokens[index] - outstanding_prefill
                )

    def _mark_prefill_complete(self, request_id: int, index: int) -> None:
        with self._state_lock:
            load = self._request_loads.get(request_id)
            if load is None:
                return
            work, outstanding_prefill = load
            if outstanding_prefill:
                self._prefill_tokens[index] = max(
                    0, self._prefill_tokens[index] - outstanding_prefill
                )
                remaining_work = max(1, work - outstanding_prefill)
                self._assigned_work[index] = max(
                    0, self._assigned_work[index] - outstanding_prefill
                )
                self._request_loads[request_id] = (remaining_work, 0)

    def _rpc(self, index: int, command: dict, expected_op: str, request_id=None) -> dict:
        rpc_id = next(self._rpc_ids)
        waiter: Queue = Queue(maxsize=1)
        # Serialize only the command handoff. The worker response sink tags
        # every response with rpc_id, allowing any caller to drain the shared
        # response queue and forward it to the correct waiter.
        try:
            with self._locks[index]:
                if not self._processes[index].is_alive():
                    from hydraserve.engine import WorkerUnavailableError

                    raise WorkerUnavailableError(
                        f"collocated worker {index} is not running"
                    )
                with self._state_lock:
                    self._waiters[index][rpc_id] = waiter
                    self._pending[index] += 1
                try:
                    self._commands[index].put({**command, "rpc_id": rpc_id})
                except Exception:
                    with self._state_lock:
                        self._waiters[index].pop(rpc_id, None)
                        self._pending[index] = max(0, self._pending[index] - 1)
                    raise
        except Exception:
            with self._state_lock:
                self._healthy[index] = False
            raise

        deadline = monotonic() + self.operation_timeout
        try:
            while True:
                try:
                    result = waiter.get_nowait()
                    break
                except Empty:
                    pass
                if not self._processes[index].is_alive():
                    from hydraserve.engine import WorkerUnavailableError

                    raise WorkerUnavailableError(
                        f"collocated worker {index} exited during RPC"
                    )
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for collocated worker")
                acquired = self._response_locks[index].acquire(
                    timeout=min(0.05, remaining)
                )
                if not acquired:
                    continue
                try:
                    try:
                        response = self._responses[index].get(
                            timeout=min(0.05, remaining)
                        )
                    except Empty:
                        continue
                    response_id = response.get("rpc_id")
                    with self._state_lock:
                        target = self._waiters[index].get(response_id)
                        # Compatibility for old/test workers that omit rpc_id.
                        if response_id is None and len(self._waiters[index]) == 1:
                            target = waiter
                    if target is not None:
                        target.put_nowait(response)
                finally:
                    self._response_locks[index].release()
        except Exception:
            with self._state_lock:
                self._healthy[index] = False
            raise
        finally:
            with self._state_lock:
                self._waiters[index].pop(rpc_id, None)
                self._pending[index] = max(0, self._pending[index] - 1)
        self._check(result, expected_op, request_id)
        self._update_capacity(index, result)
        return result

    def _get_response(self, index: int, timeout: float):
        deadline = monotonic() + timeout
        while True:
            if not self._processes[index].is_alive():
                from hydraserve.engine import WorkerUnavailableError

                raise WorkerUnavailableError(
                    f"collocated worker {index} exited during RPC"
                )
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for collocated worker")
            try:
                return self._responses[index].get(timeout=min(0.1, remaining))
            except Empty:
                continue

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

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        required_blocks = self._required_blocks(request)
        with self._state_lock:
            healthy = [i for i, value in enumerate(self._healthy) if value]
            known_capacities = [
                self._capacity[i]
                for i in healthy
                if self._capacity[i] is not None
            ]
        if (
            healthy
            and len(known_capacities) == len(healthy)
            and required_blocks
            > max(capacity.kv_total_blocks for capacity in known_capacities)
        ):
            return AdmissionDecision.reject(
                f"request needs {required_blocks} KV blocks, largest worker capacity "
                f"is {max(capacity.kv_total_blocks for capacity in known_capacities)}"
            )
        attempted: set[int] = set()
        retryable_reasons: list[str] = []
        permanent_reasons: list[str] = []
        while len(attempted) < len(self._processes):
            try:
                index = self._pick_worker(request, excluded=attempted)
            except LookupError as exc:
                retryable_reasons.append(str(exc))
                break
            attempted.add(index)
            try:
                result = self._rpc(
                    index,
                    self._request_command("reserve", request),
                    "admission",
                    request.request_id,
                )
            except Exception as exc:
                self._unassign_worker(index, request)
                retryable_reasons.append(
                    f"collocated worker {index} reserve failed: {exc}"
                )
                continue
            if not result.get("admitted"):
                self._unassign_worker(index, request)
                reason = result.get("reason", "worker rejected the reservation")
                if result.get("retryable", True):
                    retryable_reasons.append(reason)
                else:
                    permanent_reasons.append(reason)
                continue
            with self._state_lock:
                self._bound[request.request_id] = index
                self._request_loads[request.request_id] = (
                    self._request_work(request),
                    len(request.token_ids),
                )
            request.worker_id = index
            request.worker_pool = "collocated"
            request.route = "collocated"
            request.route_reason = "load_aware_collocated"
            return AdmissionDecision.accept()
        if retryable_reasons:
            return AdmissionDecision.defer(retryable_reasons[-1])
        if permanent_reasons:
            return AdmissionDecision.reject(permanent_reasons[-1])
        return AdmissionDecision.defer(
            "no collocated worker accepted the reservation"
        )

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        index = self._bound[request.request_id]
        try:
            result = self._rpc(
                index,
                self._request_command("collocated_prepare", request),
                "collocated_prepare",
                request.request_id,
            )
        finally:
            self._mark_prefill_complete(request.request_id, index)
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(result["token_id"])

    def decode(self, requests: tuple[ServingRequest, ...]) -> tuple[TokenSample, ...]:
        if not requests:
            return ()
        groups: dict[int, list[tuple[int, ServingRequest]]] = {}
        failures: dict[int, BaseException] = {}
        for position, request in enumerate(requests):
            try:
                index = self._bound[request.request_id]
            except KeyError:
                failures[request.request_id] = RuntimeError(
                    "request has no collocated worker binding"
                )
                continue
            groups.setdefault(index, []).append((position, request))
        output: list[TokenSample | None] = [None] * len(requests)

        def execute(index, indexed_requests):
            request_ids = tuple(item.request_id for _, item in indexed_requests)
            result = self._rpc(
                index,
                {"op": "decode", "request_ids": request_ids},
                "decode",
            )
            if tuple(result["request_ids"]) != request_ids:
                raise RuntimeError("worker returned a different request batch")
            samples = result.get("samples")
            if samples is not None:
                return tuple(samples)
            return tuple(int(token) for token in result["token_ids"])

        # Dispatch every device concurrently. A sequential coordinator loop
        # serializes N independent GPUs and invalidates the 4xDP baseline.
        successes: dict[int, TokenSample] = {}
        futures = {
            self._decode_executor.submit(execute, index, indexed_requests): indexed_requests
            for index, indexed_requests in groups.items()
        }
        for future in as_completed(futures):
            indexed_requests = futures[future]
            try:
                for (position, _), sample in zip(
                    indexed_requests, future.result(), strict=True
                ):
                    output[position] = sample
                    request = requests[position]
                    successes[request.request_id] = sample
            except Exception as exc:
                for _, request in indexed_requests:
                    failures[request.request_id] = exc
        if failures:
            raise PartialDecodeError(successes, failures)
        return tuple(output)

    def decode_batch_sizes(
        self, requests: tuple[ServingRequest, ...]
    ) -> dict[int, int]:
        """Report per-request width after coordinator-to-GPU grouping."""

        with self._state_lock:
            bindings = dict(self._bound)
        groups: dict[int, list[int]] = {}
        for request in requests:
            index = bindings.get(request.request_id)
            if index is not None:
                groups.setdefault(index, []).append(request.request_id)
        return {
            request_id: len(request_ids)
            for request_ids in groups.values()
            for request_id in request_ids
        }

    def release(self, request_id: int) -> None:
        with self._state_lock:
            index = self._bound.get(request_id)
        if index is None:
            return
        try:
            self._rpc(
                index,
                {"op": "release", "request_id": request_id},
                "release",
                request_id,
            )
        except Exception:
            pass
        finally:
            with self._state_lock:
                self._bound.pop(request_id, None)
                load = self._request_loads.pop(request_id, None)
            self._unassign_worker(index, load=load)

    def capacity(self) -> BackendCapacity:
        with self._state_lock:
            caps = [c for c in self._capacity if c is not None]
        if not caps:
            total_blocks = (
                self.config.cache_tokens_per_worker + self.config.block_size - 1
            ) // self.config.block_size - self.config.kv_headroom_blocks
            return BackendCapacity(
                total_blocks * len(self.config.devices),
                total_blocks * len(self.config.devices),
                self.config.max_state_slots_per_worker * len(self.config.devices),
                self.config.max_state_slots_per_worker * len(self.config.devices),
            )
        return BackendCapacity(
            kv_total_blocks=sum(c.kv_total_blocks for c in caps),
            kv_free_blocks=sum(c.kv_free_blocks for c in caps),
            state_total_slots=sum(c.state_total_slots for c in caps),
            state_free_slots=sum(c.state_free_slots for c in caps),
        )

    def close(self, *, force: bool = False) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._decode_executor.shutdown(wait=not force, cancel_futures=force)
        for index in range(len(self._processes)):
            try:
                self._commands[index].put({"op": "shutdown"})
            except Exception:
                pass
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=5 if not force else 1)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        if self._bootstrap_server is not None:
            try:
                self._bootstrap_server.close()
            except Exception:
                pass
