"""Persistent two-process PARTIAL_TRANSFER serving backend."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from threading import Lock, RLock
from time import monotonic
from uuid import uuid4

from hydraserve.engine.serving_loop import (
    AdmissionDecision,
    BackendCapacity,
    ServingRequest,
)
from hydraserve.engine.sampling import SamplingParams, TokenSample, sample_logits
from hydraserve.router import (
    AdaptiveRouter,
    CostAwareRouter,
    Route,
    RouteDecision,
    RouteReason,
)


@dataclass(frozen=True, slots=True)
class PDWorkerConfig:
    model_dir: str
    prefill_device: str = "cuda:0"
    decode_device: str = "cuda:1"
    cache_tokens: int = 65536
    block_size: int = 16
    use_flash_attention: bool = True
    prefill_chunk_size: int = 4096
    max_state_slots: int = 64
    prefix_cache_blocks: int = 0
    prefix_cache_min_frequency: int = 2


@dataclass(frozen=True, slots=True)
class TransferValidationStats:
    replay_mismatches: int


def _request(
    request_id: int,
    token_ids,
    max_new_tokens: int,
    *,
    transferred: bool,
    route: Route = Route.PD_DISAGGREGATED,
    sampling_params: SamplingParams | None = None,
):
    from hydraserve.engine.scheduler import Request, RequestState
    request = Request(
        request_id,
        tuple(token_ids),
        max_new_tokens,
        route,
        sampling_params=sampling_params or SamplingParams(),
    )
    if transferred:
        request.transition(RequestState.PREFILL_RUNNING)
        request.transition(RequestState.TRANSFER_PENDING)
    return request


def _prefill_worker(
    config: PDWorkerConfig,
    namespace: str | tuple[str, ...],
    commands,
    responses,
) -> None:
    backends = []
    try:
        import torch

        from hydraserve.engine.pd_worker import PrefillWorker
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.transfer import SharedMemoryTransferBackend, TransferPipeline

        device = torch.device(config.prefill_device)
        torch.cuda.set_device(device)
        runtime = QwenTextRuntime.from_checkpoint(
            config.model_dir,
            device=device,
            dtype=torch.bfloat16,
            use_triton=True,
            use_flash_attention=config.use_flash_attention,
        )
        namespaces = (namespace,) if isinstance(namespace, str) else tuple(namespace)
        if not namespaces:
            raise ValueError("prefill worker requires at least one decode namespace")
        workers = []
        for worker_index, worker_namespace in enumerate(namespaces):
            backend = SharedMemoryTransferBackend(namespace=worker_namespace)
            backends.append(backend)
            workers.append(
                PrefillWorker(
                    runtime,
                    TransferPipeline(
                        backend, src_gpu=0, dst_gpu=worker_index + 1
                    ),
                )
            )
        responses.put({"op": "ready", "model_name": runtime.config.name})
        while True:
            command = commands.get()
            if command["op"] == "shutdown":
                responses.put({"op": "shutdown"})
                return
            request_id = command["request_id"]
            try:
                worker_index = int(command.get("worker_index", 0))
                if not 0 <= worker_index < len(workers):
                    raise ValueError(f"unknown decode worker {worker_index}")
                request = _request(
                    request_id,
                    command["token_ids"],
                    command["max_new_tokens"],
                    transferred=False,
                    sampling_params=command.get("sampling_params"),
                )
                result = workers[worker_index].process(
                    request,
                    n_minus_one=True,
                    chunk_size=config.prefill_chunk_size,
                )
                responses.put(
                    {
                        "op": "prefill",
                        "request_id": request_id,
                        "worker_index": worker_index,
                        "token_id": result.first_token_id,
                        "sample": result.sample,
                    }
                )
            except Exception as exc:
                responses.put(
                    {
                        "op": "error",
                        "request_id": request_id,
                        "message": repr(exc),
                    }
                )
    except Exception as exc:
        responses.put({"op": "startup_error", "message": repr(exc)})
    finally:
        for backend in backends:
            backend.close()


def _decode_worker(
    config: PDWorkerConfig,
    namespace: str,
    commands,
    responses,
    worker_index: int = 0,
) -> None:
    backend = None
    try:
        import torch

        from hydraserve.cache import (
            CacheNamespace,
            CostAwarePrefixPolicy,
            GpuLinearStatePool,
            KVBlockManager,
            PagedKVCache,
            PrefixCache,
        )
        from hydraserve.engine.pd_worker import DecodeWorker
        from hydraserve.engine.scheduler import RequestState
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.transfer import SharedMemoryTransferBackend, TransferPipeline

        device = torch.device(config.decode_device)
        torch.cuda.set_device(device)
        runtime = QwenTextRuntime.from_checkpoint(
            config.model_dir,
            device=device,
            dtype=torch.bfloat16,
            use_triton=True,
            # PARTIAL decode recomputes the whole prompt; do not require the
            # optional prefill-only FlashAttention package on decode workers.
            use_flash_attention=False,
        )
        blocks = (config.cache_tokens + config.block_size - 1) // config.block_size
        prefix_cache = (
            PrefixCache(
                config.block_size,
                max_blocks=config.prefix_cache_blocks,
                policy=CostAwarePrefixPolicy(
                    minimum_frequency=config.prefix_cache_min_frequency
                ),
            )
            if config.prefix_cache_blocks
            else None
        )
        revision = str(Path(config.model_dir).resolve())
        cache = PagedKVCache(
            runtime.config,
            KVBlockManager(blocks, block_size=config.block_size),
            device=device,
            dtype=torch.bfloat16,
            prefix_cache=prefix_cache,
            cache_namespace=CacheNamespace(
                model=runtime.config.name,
                tokenizer_revision=revision,
                model_revision=revision,
            ),
        )
        backend = SharedMemoryTransferBackend(namespace=namespace)
        worker = DecodeWorker(
            runtime,
            TransferPipeline(backend, src_gpu=0, dst_gpu=worker_index + 1),
            cache,
        )
        requests = {}
        states = {}
        state_pool = GpuLinearStatePool(
            config.max_state_slots, runtime.config, device=device
        )
        state_capacity = state_pool.capacity_snapshot().total_slots
        reservations = set()
        def capacity_payload():
            capacity = cache.block_manager.capacity()
            live = len(set(states) | reservations)
            return {
                "kv_total_blocks": capacity.total_blocks,
                "kv_free_blocks": capacity.free_blocks,
                "state_total_slots": state_capacity,
                "state_free_slots": max(0, state_capacity - live),
            }

        responses.put(
            {"op": "ready", "model_name": runtime.config.name, **capacity_payload()}
        )
        while True:
            command = commands.get()
            operation = command["op"]
            if operation == "shutdown":
                for request_id in tuple(set(states) | reservations):
                    cache.free(request_id)
                responses.put({"op": "shutdown"})
                return
            try:
                if operation == "reserve":
                    request_id = command["request_id"]
                    total_tokens = len(command["token_ids"]) + max(
                        0, command["max_new_tokens"] - 1
                    )
                    required = cache.block_manager.blocks_required(total_tokens)
                    live_requests = set(states) | reservations
                    if required > cache.block_manager.num_blocks:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": False,
                                "retryable": False,
                                "reason": (
                                    f"request needs {required} KV blocks, worker capacity "
                                    f"is {cache.block_manager.num_blocks}"
                                ),
                                **capacity_payload(),
                            }
                        )
                    elif request_id in live_requests:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                                **capacity_payload(),
                            }
                        )
                    elif len(live_requests) >= state_capacity:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": False,
                                "retryable": True,
                                "reason": "recurrent-state slots are exhausted",
                                **capacity_payload(),
                            }
                        )
                    else:
                        try:
                            cache.allocate(
                                request_id,
                                len(command["token_ids"]),
                                reserve_tokens=total_tokens,
                                token_ids=command["token_ids"],
                            )
                        except MemoryError:
                            responses.put(
                                {
                                    "op": "admission",
                                    "request_id": request_id,
                                    "admitted": False,
                                    "retryable": True,
                                    "reason": "decode worker KV capacity is exhausted",
                                    **capacity_payload(),
                                }
                            )
                            continue
                        reservations.add(request_id)
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                                **capacity_payload(),
                            }
                        )
                elif operation == "prepare":
                    request_id = command["request_id"]
                    request = _request(
                        request_id,
                        command["token_ids"],
                        command["max_new_tokens"],
                        transferred=True,
                        sampling_params=command.get("sampling_params"),
                    )
                    prepared = worker.receive_and_prepare(
                        request,
                        timeout=command.get("timeout"),
                        preallocated=request_id in reservations,
                        chunk_size=config.prefill_chunk_size,
                    )
                    requests[request_id] = request
                    states[request_id] = state_pool.install(
                        request_id, prepared.state
                    )
                    responses.put(
                        {
                            "op": "prepare",
                            "request_id": request_id,
                            "token_id": prepared.first_token_id,
                            "replay_consistent": prepared.replay_consistent,
                        }
                    )
                elif operation == "collocated_prepare":
                    request_id = command["request_id"]
                    if request_id not in reservations:
                        raise RuntimeError("collocated prefill requires a KV reservation")
                    request = _request(
                        request_id,
                        command["token_ids"],
                        command["max_new_tokens"],
                        transferred=False,
                        route=Route.COLLOCATED,
                        sampling_params=command.get("sampling_params"),
                    )
                    request.transition(RequestState.PREFILL_RUNNING)
                    input_ids = torch.tensor(
                        [request.token_ids], device=device, dtype=torch.long
                    )
                    with torch.inference_mode():
                        logits, state = runtime.prefill(
                            input_ids,
                            chunk_size=config.prefill_chunk_size,
                            paged_cache=cache,
                            request_id=request_id,
                        )
                    cache.publish_prefix(request_id, request.token_ids)
                    sample = sample_logits(
                        logits[:, -1],
                        (request.token_ids,),
                        (request.sampling_params,),
                        steps=(0,),
                    )[0]
                    token_id = sample.token_id
                    request.generated_token_ids.append(token_id)
                    requests[request_id] = request
                    states[request_id] = state_pool.install(request_id, state)
                    responses.put(
                        {
                            "op": "collocated_prepare",
                            "request_id": request_id,
                            "token_id": token_id,
                            "sample": sample,
                        }
                    )
                elif operation == "decode":
                    request_ids = tuple(command["request_ids"])
                    cache.block_manager.grow_many(request_ids, additional_tokens=1)
                    input_ids = torch.tensor(
                        [requests[request_id].generated_token_ids[-1] for request_id in request_ids],
                        device=device,
                        dtype=torch.long,
                    ).unsqueeze(1)
                    batch_states = [states[request_id] for request_id in request_ids]
                    with torch.inference_mode():
                        logits, _ = runtime.decode_batch(
                            input_ids, batch_states, cache, request_ids
                        )
                    samples = sample_logits(
                        logits[:, -1],
                        (
                            requests[request_id].token_ids
                            + tuple(requests[request_id].generated_token_ids)
                            for request_id in request_ids
                        ),
                        (requests[request_id].sampling_params for request_id in request_ids),
                        steps=(
                            len(requests[request_id].generated_token_ids)
                            for request_id in request_ids
                        ),
                    )
                    token_ids = tuple(sample.token_id for sample in samples)
                    for request_id, token_id in zip(request_ids, token_ids, strict=True):
                        requests[request_id].generated_token_ids.append(token_id)
                    responses.put(
                        {
                            "op": "decode",
                            "request_ids": request_ids,
                            "token_ids": token_ids,
                            "samples": samples,
                        }
                    )
                elif operation == "release":
                    request_id = command["request_id"]
                    reservations.discard(request_id)
                    states.pop(request_id, None)
                    state_pool.free(request_id)
                    requests.pop(request_id, None)
                    cache.free(request_id)
                    responses.put(
                        {"op": "release", "request_id": request_id, **capacity_payload()}
                    )
                elif operation == "prefix_probe":
                    match = cache.probe_prefix(command["token_ids"])
                    responses.put(
                        {
                            "op": "prefix_probe",
                            "request_id": command["request_id"],
                            "matched_tokens": match.matched_tokens,
                            **capacity_payload(),
                        }
                    )
                else:
                    raise ValueError(f"unknown decode-worker operation {operation!r}")
            except Exception as exc:
                if operation in {"prepare", "collocated_prepare"}:
                    reservations.discard(command.get("request_id"))
                    state_pool.free(command.get("request_id"))
                    cache.free(command.get("request_id"))
                responses.put(
                    {
                        "op": "error",
                        "request_id": command.get("request_id"),
                        "request_ids": command.get("request_ids"),
                        "message": repr(exc),
                    }
                )
    except Exception as exc:
        responses.put({"op": "startup_error", "message": repr(exc)})
    finally:
        if backend is not None:
            backend.close()


class DisaggregatedGenerationBackend:
    """GenerationBackend backed by persistent prefill and decode GPU processes."""

    def __init__(
        self,
        config: PDWorkerConfig,
        *,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
    ) -> None:
        if config.prefill_device == config.decode_device:
            raise ValueError("PD serving requires distinct prefill and decode devices")
        if min(
            config.cache_tokens,
            config.block_size,
            config.prefill_chunk_size,
            config.max_state_slots,
            config.prefix_cache_min_frequency,
        ) <= 0:
            raise ValueError("cache limits must be positive")
        if config.prefix_cache_blocks < 0:
            raise ValueError("prefix cache blocks cannot be negative")
        self.config = config
        self.supports_async_prefill = True
        self.operation_timeout = operation_timeout
        self.namespace = f"hydraserve-pd-{uuid4().hex}"
        context = mp.get_context("spawn")
        self._prefill_commands = context.Queue()
        self._prefill_responses = context.Queue()
        self._decode_commands = context.Queue()
        self._decode_responses = context.Queue()
        self._prefill = context.Process(
            target=_prefill_worker,
            args=(config, self.namespace, self._prefill_commands, self._prefill_responses),
            name="hydraserve-prefill",
        )
        self._decode = context.Process(
            target=_decode_worker,
            args=(config, self.namespace, self._decode_commands, self._decode_responses),
            name="hydraserve-decode",
        )
        self._closed = False
        self._decode_lock = Lock()
        self._admitted_requests: set[int] = set()
        self._reserved_blocks: dict[int, int] = {}
        self._last_capacity: BackendCapacity | None = None
        self._replay_mismatches = 0
        self._decode.start()
        self._prefill.start()
        try:
            prefill_ready = self._get(self._prefill_responses, startup_timeout)
            decode_ready = self._get(self._decode_responses, startup_timeout)
            self._check(prefill_ready, "ready")
            self._check(decode_ready, "ready")
            if prefill_ready["model_name"] != decode_ready["model_name"]:
                raise RuntimeError("prefill/decode workers loaded different models")
            self.model_name = prefill_ready["model_name"]
            self._update_capacity(decode_ready)
        except Exception:
            self.close(force=True)
            raise

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        return self._reserve_decode(request)

    def _reserve_decode(
        self, request: ServingRequest, *, force_rpc: bool = False
    ) -> AdmissionDecision:
        total_tokens = len(request.token_ids) + max(0, request.max_new_tokens - 1)
        required_blocks = (total_tokens + self.config.block_size - 1) // self.config.block_size
        command = {
            "op": "reserve",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
        }
        with self._decode_lock:
            if request.request_id in self._admitted_requests and not force_rpc:
                return AdmissionDecision.accept()
            self._decode_commands.put(command)
            result = self._get(self._decode_responses, self.operation_timeout)
            self._check(result, "admission", request.request_id)
            self._update_capacity(result)
            if result.get("admitted"):
                self._admitted_requests.add(request.request_id)
                self._reserved_blocks[request.request_id] = required_blocks
                return AdmissionDecision.accept()
            if result.get("retryable"):
                return AdmissionDecision.defer(
                    result.get("reason", "decode worker is temporarily full")
                )
            return AdmissionDecision.reject(
                result.get("reason", "request exceeds decode worker capacity")
            )

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        decision = self.admit(request)
        if not decision.admitted:
            raise MemoryError(decision.reason or "request cannot be admitted")
        return self._prefill_pd(request)

    def _prefill_pd(self, request: ServingRequest) -> int | TokenSample:
        command = {
            "op": "prefill",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
        }
        self._prefill_commands.put(command)
        result = self._get(self._prefill_responses, self.operation_timeout)
        self._check(result, "prefill", request.request_id)
        with self._decode_lock:
            self._decode_commands.put(
                {
                    **command,
                    "op": "prepare",
                    "timeout": self.operation_timeout,
                }
            )
            prepared = self._get(self._decode_responses, self.operation_timeout)
        self._check(prepared, "prepare", request.request_id)
        if result["token_id"] != prepared["token_id"]:
            raise RuntimeError("prefill/decode first-token mismatch")
        if not prepared.get("replay_consistent", True):
            self._replay_mismatches += 1
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(prepared["token_id"])

    def _prefill_collocated(self, request: ServingRequest) -> int | TokenSample:
        command = {
            "op": "collocated_prepare",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
        }
        with self._decode_lock:
            self._decode_commands.put(command)
            result = self._get(self._decode_responses, self.operation_timeout)
        self._check(result, "collocated_prepare", request.request_id)
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(result["token_id"])

    def decode(
        self, requests: tuple[ServingRequest, ...]
    ) -> tuple[int | TokenSample, ...]:
        request_ids = tuple(request.request_id for request in requests)
        with self._decode_lock:
            self._decode_commands.put({"op": "decode", "request_ids": request_ids})
            result = self._get(self._decode_responses, self.operation_timeout)
        self._check(result, "decode")
        if tuple(result["request_ids"]) != request_ids:
            raise RuntimeError("decode worker returned a different request batch")
        samples = result.get("samples")
        if samples is not None:
            return tuple(samples)
        return tuple(int(token) for token in result["token_ids"])

    def release(self, request_id: int) -> None:
        if self._closed:
            return
        with self._decode_lock:
            self._decode_commands.put({"op": "release", "request_id": request_id})
            result = self._get(self._decode_responses, self.operation_timeout)
            self._admitted_requests.discard(request_id)
            self._reserved_blocks.pop(request_id, None)
        self._check(result, "release", request_id)
        self._update_capacity(result)

    def capacity(self) -> BackendCapacity:
        with self._decode_lock:
            if self._last_capacity is not None:
                return self._last_capacity
            total_blocks = (
                self.config.cache_tokens + self.config.block_size - 1
            ) // self.config.block_size
            allocated_blocks = sum(self._reserved_blocks.values())
            allocated_slots = len(self._admitted_requests)
            return BackendCapacity(
                kv_total_blocks=total_blocks,
                kv_free_blocks=max(0, total_blocks - allocated_blocks),
                state_total_slots=self.config.max_state_slots,
                state_free_slots=max(0, self.config.max_state_slots - allocated_slots),
            )

    def transfer_validation_stats(self) -> TransferValidationStats:
        return TransferValidationStats(self._replay_mismatches)

    def prefix_match_tokens(self, token_ids) -> int:
        command = {
            "op": "prefix_probe",
            "request_id": -1,
            "token_ids": tuple(int(token) for token in token_ids),
        }
        with self._decode_lock:
            self._decode_commands.put(command)
            result = self._get(self._decode_responses, self.operation_timeout)
        self._check(result, "prefix_probe", -1)
        self._update_capacity(result)
        return int(result["matched_tokens"])

    def _update_capacity(self, result: dict) -> None:
        keys = (
            "kv_total_blocks",
            "kv_free_blocks",
            "state_total_slots",
            "state_free_slots",
        )
        if all(key in result for key in keys):
            self._last_capacity = BackendCapacity(*(int(result[key]) for key in keys))

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        for commands in (self._prefill_commands, self._decode_commands):
            commands.put({"op": "shutdown"})
        if not force:
            for responses in (self._prefill_responses, self._decode_responses):
                try:
                    self._get(responses, 10.0)
                except Exception:
                    pass
        for process in (self._prefill, self._decode):
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)

    @staticmethod
    def _get(queue, timeout: float):
        try:
            return queue.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("timed out waiting for PD worker") from exc

    @staticmethod
    def _check(result, expected_op: str, request_id: int | None = None) -> None:
        if result.get("op") in {"error", "startup_error"}:
            raise RuntimeError(result.get("message", "PD worker failed"))
        if result.get("op") != expected_op:
            raise RuntimeError(
                f"expected PD worker response {expected_op!r}, got {result.get('op')!r}"
            )
        if request_id is not None and result.get("request_id") != request_id:
            raise RuntimeError("PD worker returned a different request")


@dataclass(frozen=True, slots=True)
class RoutingStats:
    collocated: int
    pd_disaggregated: int
    pd_failures: int
    prefill_healthy: bool


class AdaptiveGenerationBackend(DisaggregatedGenerationBackend):
    """Route each request between collocated and PD execution.

    Both routes use HydraServe's resident decode worker and the same KV/state
    admission transaction. The route is immutable once admitted. An ambiguous
    PD timeout fails that request and quarantines the prefill route; only later
    requests safely degrade to collocated execution.
    """

    def __init__(
        self,
        config: PDWorkerConfig,
        *,
        router: AdaptiveRouter | None = None,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
    ) -> None:
        self.router = router or CostAwareRouter()
        self._route_decisions: dict[int, RouteDecision] = {}
        self._route_lock = RLock()
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
        self._prefill_healthy = True
        super().__init__(
            config,
            startup_timeout=startup_timeout,
            operation_timeout=operation_timeout,
        )

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        with self._route_lock:
            if request.request_id in self._route_decisions:
                return super().admit(request)
        capacity = self.capacity()
        with self._route_lock:
            prefill_healthy = self._prefill_healthy
        if prefill_healthy:
            decision = self.router.decide(
                len(request.token_ids),
                capacity.decode_load,
                capacity.has_request_slot,
            )
        else:
            decision = RouteDecision(
                route=Route.COLLOCATED,
                reason=RouteReason.PREFILL_UNAVAILABLE,
                prompt_tokens=len(request.token_ids),
                decode_load=capacity.decode_load,
                decode_has_slot=capacity.has_request_slot,
            )
        admitted = super().admit(request)
        if admitted.admitted:
            with self._route_lock:
                bound = self._route_decisions.setdefault(request.request_id, decision)
                request.route = bound.route.value
                request.route_reason = bound.reason.value
                request.worker_id = 0
                request.route_collocated_cost_ms = bound.collocated_cost_ms
                request.route_pd_cost_ms = bound.pd_cost_ms
                request.route_estimated_savings_ms = bound.estimated_savings_ms
                request.route_cost_confidence = bound.cost_model_confidence
        return admitted

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        admitted = self.admit(request)
        if not admitted.admitted:
            raise MemoryError(admitted.reason or "request cannot be admitted")
        decision = self.route_for(request.request_id)
        started = monotonic()
        if decision.route is Route.COLLOCATED:
            token_id = self._prefill_collocated(request)
            self._observe_route_cost(decision, started)
            with self._route_lock:
                self._collocated_count += 1
            return token_id
        try:
            token_id = self._prefill_pd(request)
        except TimeoutError:
            with self._route_lock:
                self._pd_failures += 1
                self._prefill_healthy = False
            raise
        self._observe_route_cost(decision, started)
        with self._route_lock:
            self._pd_count += 1
        return token_id

    def _observe_route_cost(self, decision: RouteDecision, started: float) -> None:
        observe = getattr(self.router, "observe", None)
        if observe is not None:
            observe(
                decision.route,
                decision.prompt_tokens,
                (monotonic() - started) * 1000.0,
                decision.decode_load,
            )

    def route_for(self, request_id: int) -> RouteDecision:
        with self._route_lock:
            try:
                return self._route_decisions[request_id]
            except KeyError as exc:
                raise KeyError(f"request {request_id} has no bound route") from exc

    def release(self, request_id: int) -> None:
        try:
            super().release(request_id)
        finally:
            with self._route_lock:
                self._route_decisions.pop(request_id, None)

    def routing_stats(self) -> RoutingStats:
        with self._route_lock:
            return RoutingStats(
                collocated=self._collocated_count,
                pd_disaggregated=self._pd_count,
                pd_failures=self._pd_failures,
                prefill_healthy=self._prefill_healthy,
            )

    def routing_cost_stats(self):
        stats = getattr(self.router, "stats", None)
        return None if stats is None else stats()
