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
import multiprocessing as mp
from pathlib import Path
from queue import Empty
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
    def release_parallelism(self) -> int:
        return len(self.config.devices)

    @property
    def prefill_parallelism(self) -> int:
        """Concurrent prefill slots = worker count (each worker collocates)."""
        return len(self.config.devices)

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
        self._locks = [Lock() for _ in range(worker_count)]
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
        self._healthy = [True] * worker_count
        self._bound: dict[int, int] = {}
        self._capacity: list[BackendCapacity | None] = [None] * worker_count
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
            return
        self._capacity[index] = BackendCapacity(
            kv_total_blocks=int(payload.get("kv_total_blocks", 0)),
            kv_free_blocks=int(payload.get("kv_free_blocks", 0)),
            state_total_slots=int(payload.get("state_total_slots", 0)),
            state_free_slots=int(payload.get("state_free_slots", 0)),
        )

    def _pick_worker(self) -> int:
        """Reserve the least-loaded healthy worker, round-robin on ties."""
        with self._state_lock:
            count = len(self._processes)
            candidates = [i for i in range(count) if self._healthy[i]]
            if not candidates:
                candidates = list(range(count))
            start = self._round_robin % count
            index = min(
                candidates,
                key=lambda i: (
                    self._assigned[i],
                    self._pending[i],
                    (i - start) % count,
                ),
            )
            self._assigned[index] += 1
            self._round_robin = (index + 1) % count
            return index

    def _unassign_worker(self, index: int) -> None:
        with self._state_lock:
            self._assigned[index] = max(0, self._assigned[index] - 1)

    def _rpc(self, index: int, command: dict, expected_op: str, request_id=None) -> dict:
        failure = None
        with self._locks[index]:
            try:
                if not self._processes[index].is_alive():
                    from hydraserve.engine import WorkerUnavailableError

                    raise WorkerUnavailableError(
                        f"collocated worker {index} is not running"
                    )
                with self._state_lock:
                    self._pending[index] += 1
                self._commands[index].put(command)
                result = self._get_response(index, self.operation_timeout)
            except Exception as exc:
                failure = exc
            finally:
                with self._state_lock:
                    self._pending[index] = max(0, self._pending[index] - 1)
        if failure is not None:
            with self._state_lock:
                self._healthy[index] = False
            raise failure
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
        index = self._pick_worker()
        try:
            result = self._rpc(
                index,
                self._request_command("reserve", request),
                "admission",
                request.request_id,
            )
        except Exception as exc:
            self._unassign_worker(index)
            return AdmissionDecision.defer(f"collocated reserve failed: {exc}")
        if not result.get("admitted"):
            self._unassign_worker(index)
            reason = result.get("reason", "worker rejected the reservation")
            return AdmissionDecision.defer(reason)
        with self._state_lock:
            self._bound[request.request_id] = index
        request.worker_id = index
        request.worker_pool = "collocated"
        request.route = "collocated"
        request.route_reason = "fixed_collocated"
        return AdmissionDecision.accept()

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        index = self._bound[request.request_id]
        result = self._rpc(
            index,
            self._request_command("collocated_prepare", request),
            "collocated_prepare",
            request.request_id,
        )
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
            self._unassign_worker(index)

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
