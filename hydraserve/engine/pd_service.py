"""Persistent two-process PARTIAL_TRANSFER serving backend."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from threading import Event, Lock, RLock, Thread
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
    kv_headroom_blocks: int = 0


@dataclass(frozen=True, slots=True)
class TransferValidationStats:
    replay_mismatches: int


class PDWorkerUnavailableError(RuntimeError):
    """A fixed-PD worker exited or timed out during an RPC."""


@dataclass(frozen=True, slots=True)
class PDDecodeRecoveryStats:
    total_workers: int
    healthy_workers: int
    attempts: int
    successes: int
    failures: int
    recovering_workers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PDPrefillRecoveryStats:
    healthy: bool
    attempts: int
    successes: int
    failures: int
    recovering: bool


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
            KVBlockManager(
                blocks,
                block_size=config.block_size,
                headroom_blocks=config.kv_headroom_blocks,
            ),
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
                "kv_cache_stats": cache.stats(),
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
                elif operation == "recover":
                    request_id = command["request_id"]
                    if request_id not in reservations:
                        raise RuntimeError("recovery requires a KV/state reservation")
                    replay_token_ids = tuple(command["replay_token_ids"])
                    generated_token_ids = tuple(command["generated_token_ids"])
                    if not replay_token_ids or not generated_token_ids:
                        raise RuntimeError("recovery requires consumed and emitted tokens")
                    expected_replay = tuple(command["token_ids"]) + generated_token_ids[:-1]
                    if replay_token_ids != expected_replay:
                        raise RuntimeError("recovery replay does not match request history")
                    total_tokens = len(command["token_ids"]) + max(
                        0, command["max_new_tokens"] - 1
                    )
                    state_pool.free(request_id)
                    states.pop(request_id, None)
                    requests.pop(request_id, None)
                    cache.free(request_id)
                    cache.allocate(
                        request_id,
                        len(replay_token_ids),
                        reserve_tokens=total_tokens,
                        token_ids=replay_token_ids,
                    )
                    replay = torch.tensor(
                        [replay_token_ids], device=device, dtype=torch.long
                    )
                    with torch.inference_mode():
                        _, state = runtime.prefill(
                            replay,
                            chunk_size=config.prefill_chunk_size,
                            paged_cache=cache,
                            request_id=request_id,
                        )
                    request = _request(
                        request_id,
                        command["token_ids"],
                        command["max_new_tokens"],
                        transferred=False,
                        route=Route.COLLOCATED,
                        sampling_params=command.get("sampling_params"),
                    )
                    request.generated_token_ids.extend(generated_token_ids)
                    requests[request_id] = request
                    states[request_id] = state_pool.install(request_id, state)
                    responses.put(
                        {"op": "recover", "request_id": request_id, **capacity_payload()}
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
                if operation in {"prepare", "collocated_prepare", "recover"}:
                    reservations.discard(command.get("request_id"))
                    state_pool.free(command.get("request_id"))
                    states.pop(command.get("request_id"), None)
                    requests.pop(command.get("request_id"), None)
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
        max_worker_restarts: int = 3,
        worker_restart_backoff_s: float = 0.5,
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
        if max_worker_restarts <= 0 or worker_restart_backoff_s < 0:
            raise ValueError("invalid worker recovery policy")
        total_blocks = (
            config.cache_tokens + config.block_size - 1
        ) // config.block_size
        if not 0 <= config.kv_headroom_blocks < total_blocks:
            raise ValueError("KV headroom must be below physical cache blocks")
        self.config = config
        self.supports_async_prefill = True
        self.operation_timeout = operation_timeout
        self.startup_timeout = startup_timeout
        self.max_worker_restarts = max_worker_restarts
        self.worker_restart_backoff_s = worker_restart_backoff_s
        self.namespace = f"hydraserve-pd-{uuid4().hex}"
        self._context = mp.get_context("spawn")
        self._prefill_commands = self._context.Queue()
        self._prefill_responses = self._context.Queue()
        self._decode_commands = self._context.Queue()
        self._decode_responses = self._context.Queue()
        self._prefill = self._new_prefill_process()
        self._decode = self._new_decode_process()
        self._closed = False
        self._prefill_lock = Lock()
        self._decode_lock = Lock()
        self._recovery_lock = RLock()
        self._recovery_stop = Event()
        self._prefill_healthy = True
        self._decode_healthy = True
        self._prefill_recovering = False
        self._decode_recovering = False
        self._prefill_recovery_thread: Thread | None = None
        self._decode_recovery_thread: Thread | None = None
        self._prefill_recovery_attempts = 0
        self._prefill_recovery_successes = 0
        self._prefill_recovery_failures = 0
        self._decode_recovery_attempts = 0
        self._decode_recovery_successes = 0
        self._decode_recovery_failures = 0
        self._admitted_requests: set[int] = set()
        self._lost_requests: set[int] = set()
        self._reserved_blocks: dict[int, int] = {}
        self._last_capacity: BackendCapacity | None = None
        self._last_cache_stats: dict[str, int | float] = {}
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
        if not self._decode_available():
            return AdmissionDecision.defer("decode worker is restarting")
        if not self._prefill_available():
            return AdmissionDecision.defer("prefill worker is restarting")
        return self._reserve_decode(request)

    def _reserve_decode(
        self, request: ServingRequest, *, force_rpc: bool = False
    ) -> AdmissionDecision:
        if not self._decode_available():
            return AdmissionDecision.defer("decode worker is restarting")
        initial_capacity = self.capacity()
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
        result = self._decode_rpc(command, "admission", request.request_id)
        with self._decode_lock:
            self._update_capacity(result)
            if result.get("admitted"):
                self._admitted_requests.add(request.request_id)
                self._reserved_blocks[request.request_id] = required_blocks
                if request.route is None:
                    request.route = Route.PD_DISAGGREGATED.value
                    request.route_reason = "fixed_pd"
                    request.worker_id = 0
                    request.route_decode_load = initial_capacity.decode_load
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
        result = self._prefill_rpc(command, request.request_id)
        prepared = self._decode_rpc(
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
        result = self._decode_rpc(
            command, "collocated_prepare", request.request_id
        )
        sample = result.get("sample")
        return sample if isinstance(sample, TokenSample) else int(result["token_id"])

    def decode(
        self, requests: tuple[ServingRequest, ...]
    ) -> tuple[int | TokenSample, ...]:
        request_ids = tuple(request.request_id for request in requests)
        lost = tuple(
            request_id for request_id in request_ids if request_id in self._lost_requests
        )
        if lost:
            from hydraserve.engine.serving_loop import PartialDecodeError

            live_requests = tuple(
                request for request in requests if request.request_id not in lost
            )
            successes = {}
            if live_requests:
                live_samples = self.decode(live_requests)
                successes.update(
                    (request.request_id, sample)
                    for request, sample in zip(
                        live_requests, live_samples, strict=True
                    )
                )
            raise PartialDecodeError(
                successes,
                {
                    request_id: PDWorkerUnavailableError(
                        f"request {request_id} lost decode-worker state"
                    )
                    for request_id in lost
                },
            )
        result = self._decode_rpc(
            {"op": "decode", "request_ids": request_ids}, "decode"
        )
        if tuple(result["request_ids"]) != request_ids:
            raise RuntimeError("decode worker returned a different request batch")
        samples = result.get("samples")
        if samples is not None:
            return tuple(samples)
        return tuple(int(token) for token in result["token_ids"])

    def preempt(self, request_id: int) -> None:
        self.release(request_id)

    def recover(self, request: ServingRequest) -> AdmissionDecision:
        decision = self._reserve_decode(request)
        if not decision.admitted:
            return decision
        command = {
            "op": "recover",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "generated_token_ids": tuple(request.generated_token_ids),
            "replay_token_ids": request.token_ids
            + tuple(request.generated_token_ids[:-1]),
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
        }
        try:
            result = self._decode_rpc(command, "recover", request.request_id)
            self._update_capacity(result)
            return AdmissionDecision.accept()
        except Exception:
            try:
                self.release(request.request_id)
            except Exception:
                pass
            raise

    def release(self, request_id: int) -> None:
        if self._closed:
            return
        with self._decode_lock:
            known = request_id in self._admitted_requests
            self._admitted_requests.discard(request_id)
            self._reserved_blocks.pop(request_id, None)
            self._lost_requests.discard(request_id)
        if not known:
            return
        result = self._decode_rpc(
            {"op": "release", "request_id": request_id}, "release", request_id
        )
        self._update_capacity(result)

    def capacity(self) -> BackendCapacity:
        with self._decode_lock:
            if self._last_capacity is not None:
                return self._last_capacity
            total_blocks = (
                self.config.cache_tokens + self.config.block_size - 1
            ) // self.config.block_size - self.config.kv_headroom_blocks
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

    def cache_stats(self) -> dict[str, int | float]:
        with self._decode_lock:
            return dict(self._last_cache_stats)

    def recovery_stats(self) -> PDDecodeRecoveryStats:
        self._decode_available()
        with self._recovery_lock:
            return PDDecodeRecoveryStats(
                1,
                1 if self._decode_healthy else 0,
                self._decode_recovery_attempts,
                self._decode_recovery_successes,
                self._decode_recovery_failures,
                (0,) if self._decode_recovering else (),
            )

    def prefill_recovery_stats(self) -> PDPrefillRecoveryStats:
        self._prefill_available()
        with self._recovery_lock:
            return PDPrefillRecoveryStats(
                self._prefill_healthy,
                self._prefill_recovery_attempts,
                self._prefill_recovery_successes,
                self._prefill_recovery_failures,
                self._prefill_recovering,
            )

    def abandon(self, request_id: int) -> None:
        with self._decode_lock:
            self._admitted_requests.discard(request_id)
            self._reserved_blocks.pop(request_id, None)
            self._lost_requests.discard(request_id)

    def is_recoverable_decode_error(
        self, request_id: int, error: BaseException
    ) -> bool:
        return isinstance(error, (PDWorkerUnavailableError, TimeoutError)) or (
            request_id in self._lost_requests
        )

    def prefix_match_tokens(self, token_ids) -> int:
        command = {
            "op": "prefix_probe",
            "request_id": -1,
            "token_ids": tuple(int(token) for token in token_ids),
        }
        result = self._decode_rpc(command, "prefix_probe", -1)
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
        cache_stats = result.get("kv_cache_stats")
        if isinstance(cache_stats, dict):
            self._last_cache_stats = {
                str(key): value
                for key, value in cache_stats.items()
                if isinstance(value, (int, float))
            }

    def _prefill_available(self) -> bool:
        with self._recovery_lock:
            if self._closed or not self._prefill_healthy:
                return False
        if self._prefill.is_alive():
            return True
        with self._recovery_lock:
            self._prefill_healthy = False
        self._schedule_recovery("prefill")
        return False

    def _decode_available(self) -> bool:
        with self._recovery_lock:
            if self._closed or not self._decode_healthy:
                return False
        if self._decode.is_alive():
            return True
        self._invalidate_decode()
        self._schedule_recovery("decode")
        return False

    def _prefill_rpc(self, command: dict, request_id: int) -> dict:
        failure = None
        with self._prefill_lock:
            try:
                if not self._prefill.is_alive():
                    raise PDWorkerUnavailableError("prefill worker is not running")
                self._prefill_commands.put(command)
                result = self._get_worker_response(
                    "prefill", self.operation_timeout
                )
            except (TimeoutError, PDWorkerUnavailableError) as exc:
                failure = exc
        if failure is not None:
            with self._recovery_lock:
                self._prefill_healthy = False
            self._schedule_recovery("prefill")
            raise failure
        self._check(result, "prefill", request_id)
        return result

    def _decode_rpc(
        self, command: dict, expected_op: str, request_id: int | None = None
    ) -> dict:
        failure = None
        with self._decode_lock:
            try:
                if not self._decode.is_alive():
                    raise PDWorkerUnavailableError("decode worker is not running")
                self._decode_commands.put(command)
                result = self._get_worker_response("decode", self.operation_timeout)
            except (TimeoutError, PDWorkerUnavailableError) as exc:
                failure = exc
        if failure is not None:
            self._invalidate_decode()
            self._schedule_recovery("decode")
            raise failure
        self._check(result, expected_op, request_id)
        return result

    def _get_worker_response(self, kind: str, timeout: float):
        process = self._prefill if kind == "prefill" else self._decode
        responses = self._prefill_responses if kind == "prefill" else self._decode_responses
        deadline = monotonic() + timeout
        while True:
            if not process.is_alive():
                raise PDWorkerUnavailableError(f"{kind} worker exited during RPC")
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {kind} worker")
            try:
                return responses.get(timeout=min(0.1, remaining))
            except Empty:
                continue

    def _invalidate_decode(self) -> None:
        with self._recovery_lock, self._decode_lock:
            self._decode_healthy = False
            self._lost_requests.update(self._admitted_requests)
            self._admitted_requests.clear()
            self._reserved_blocks.clear()
            self._last_capacity = None

    def _schedule_recovery(self, kind: str) -> None:
        with self._recovery_lock:
            if self._closed:
                return
            attribute = f"_{kind}_recovering"
            if getattr(self, attribute):
                return
            setattr(self, attribute, True)
            thread = Thread(
                target=self._recover_worker,
                args=(kind,),
                name=f"hydraserve-recover-{kind}",
                daemon=True,
            )
            setattr(self, f"_{kind}_recovery_thread", thread)
        thread.start()

    def _recover_worker(self, kind: str) -> None:
        try:
            for attempt in range(self.max_worker_restarts):
                if self._recovery_stop.is_set():
                    return
                with self._recovery_lock:
                    name = f"_{kind}_recovery_attempts"
                    setattr(self, name, getattr(self, name) + 1)
                try:
                    self._restart_worker_once(kind)
                except Exception:
                    with self._recovery_lock:
                        name = f"_{kind}_recovery_failures"
                        setattr(self, name, getattr(self, name) + 1)
                    if self._recovery_stop.wait(
                        self.worker_restart_backoff_s * (2**attempt)
                    ):
                        return
                    continue
                if self._recovery_stop.is_set():
                    return
                with self._recovery_lock:
                    setattr(self, f"_{kind}_healthy", True)
                    name = f"_{kind}_recovery_successes"
                    setattr(self, name, getattr(self, name) + 1)
                return
        finally:
            with self._recovery_lock:
                setattr(self, f"_{kind}_recovering", False)
                setattr(self, f"_{kind}_recovery_thread", None)

    def _restart_worker_once(self, kind: str) -> None:
        lock = self._prefill_lock if kind == "prefill" else self._decode_lock
        with lock:
            if self._recovery_stop.is_set():
                return
            process = self._prefill if kind == "prefill" else self._decode
            if process.is_alive():
                process.terminate()
            process.join(10)
            commands = self._context.Queue()
            responses = self._context.Queue()
            if kind == "prefill":
                self._prefill_commands = commands
                self._prefill_responses = responses
                process = self._new_prefill_process()
                self._prefill = process
            else:
                self._decode_commands = commands
                self._decode_responses = responses
                process = self._new_decode_process()
                self._decode = process
            process.start()
            result = self._get_worker_response(kind, self.startup_timeout)
            self._check(result, "ready")
            if result.get("model_name") != self.model_name:
                raise RuntimeError(f"restarted {kind} worker loaded a different model")
            if kind == "decode":
                self._update_capacity(result)

    def _new_prefill_process(self):
        return self._context.Process(
            target=_prefill_worker,
            args=(
                self.config,
                self.namespace,
                self._prefill_commands,
                self._prefill_responses,
            ),
            name="hydraserve-prefill",
        )

    def _new_decode_process(self):
        return self._context.Process(
            target=_decode_worker,
            args=(
                self.config,
                self.namespace,
                self._decode_commands,
                self._decode_responses,
            ),
            name="hydraserve-decode",
        )

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._recovery_stop.set()
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
        max_worker_restarts: int = 3,
        worker_restart_backoff_s: float = 0.5,
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
            max_worker_restarts=max_worker_restarts,
            worker_restart_backoff_s=worker_restart_backoff_s,
        )

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        with self._route_lock:
            if request.request_id in self._route_decisions:
                return self._reserve_decode(request)
        capacity = self.capacity()
        prefill_healthy = self._prefill_available()
        if prefill_healthy:
            decision = self.router.decide(
                len(request.token_ids),
                capacity.decode_load,
                capacity.has_request_slot,
                request.route_prefill_queue_ahead_ms,
            )
        else:
            decision = RouteDecision(
                route=Route.COLLOCATED,
                reason=RouteReason.PREFILL_UNAVAILABLE,
                prompt_tokens=len(request.token_ids),
                decode_load=capacity.decode_load,
                decode_has_slot=capacity.has_request_slot,
            )
        admitted = self._reserve_decode(request)
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
                request.route_decode_load = bound.decode_load
                request.route_prefill_queue_ahead_ms = bound.prefill_queue_ahead_ms
        return admitted

    def prefill(self, request: ServingRequest) -> int | TokenSample:
        admitted = self.admit(request)
        if not admitted.admitted:
            raise MemoryError(admitted.reason or "request cannot be admitted")
        decision = self.route_for(request.request_id)
        started = monotonic()
        if decision.route is Route.COLLOCATED:
            token_id = self._prefill_collocated(request)
            self._observe_route_cost(request, decision, started)
            with self._route_lock:
                self._collocated_count += 1
            return token_id
        try:
            token_id = self._prefill_pd(request)
        except (TimeoutError, PDWorkerUnavailableError):
            with self._route_lock:
                self._pd_failures += 1
                self._prefill_healthy = False
            raise
        self._observe_route_cost(request, decision, started)
        with self._route_lock:
            self._pd_count += 1
        return token_id

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

    def reset_routing_calibration(self) -> None:
        reset = getattr(self.router, "reset_online_state", None)
        if reset is not None:
            reset()
