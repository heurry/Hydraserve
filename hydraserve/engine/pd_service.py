"""Persistent two-process PARTIAL_TRANSFER serving backend."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sys
from threading import Event, Lock, RLock, Thread, local
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


def _pd_protocol_trace(phase: str, request_id: int, **fields) -> None:
    """Emit opt-in machine-readable PD phase diagnostics to worker stderr."""
    if os.environ.get("HYDRASERVE_PROTOCOL_TRACE", "0") != "1":
        return
    record = {
        "component": "hydraserve_pd_protocol",
        "phase": phase,
        "request_id": request_id,
        "monotonic_s": monotonic(),
        **fields,
    }
    print(json.dumps(record, separators=(",", ":")), file=sys.stderr, flush=True)


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
    max_decode_batch_size: int = 64
    kv_quant: str | None = None
    host_prefix_cache_bytes: int = 0
    transfer_backend: str = "shm-ring"
    transfer_quant: str | None = None
    transfer_target_bytes: int = 8 << 20
    max_inflight_transfer_chunks: int = 2
    max_concurrent_prepares: int = 2
    shm_ring_slots: int = 3
    shm_ring_slot_bytes: int = 64 << 20
    worker_log_dir: str = ""
    prefill_preempt_max_ops: int = 8


def _make_pd_transfer_backend(config: PDWorkerConfig, namespace: str, model=None):
    from hydraserve.transfer import (
        SharedMemoryRingTransferBackend,
        SharedMemoryTransferBackend,
        TransferMode,
    )

    mode = (
        TransferMode.INT8_TRANSFER
        if config.transfer_quant == "int8"
        else TransferMode.FULL_TRANSFER
    )

    if config.transfer_backend == "shm-ring":
        slot_bytes = config.shm_ring_slot_bytes
        if model is not None:
            slot_bytes = max(
                slot_bytes,
                model.recurrent_state_bytes + model.conv_state_bytes + (8 << 20),
            )
        return SharedMemoryRingTransferBackend(
            namespace=namespace,
            slots=config.shm_ring_slots,
            slot_bytes=slot_bytes,
            mode=mode,
        )
    if config.transfer_backend == "shm":
        return SharedMemoryTransferBackend(namespace=namespace, mode=mode)
    raise ValueError(f"unsupported PD transfer backend: {config.transfer_backend}")


@dataclass(frozen=True, slots=True)
class TransferValidationStats:
    replay_mismatches: int


class PDWorkerUnavailableError(RuntimeError):
    """A fixed-PD worker exited or timed out during an RPC."""


class _CorrelatedResponseSink:
    """Attach the active command id to worker responses.

    A prefill worker may emit a short-decode response while a long-prefill
    command is still running.  Tagging every response lets the parent process
    multiplex those in-flight RPCs without relying on FIFO response order.
    """

    def __init__(self, queue) -> None:
        self.queue = queue
        self._context = local()
        self._sequence_lock = Lock()
        self.response_sequence = 0

    @property
    def rpc_id(self) -> int | None:
        return getattr(self._context, "rpc_id", None)

    @rpc_id.setter
    def rpc_id(self, value: int | None) -> None:
        self._context.rpc_id = value

    def put(self, payload: dict) -> None:
        with self._sequence_lock:
            self.response_sequence += 1
            response_sequence = self.response_sequence
        if "response_sequence" not in payload:
            payload = {**payload, "response_sequence": response_sequence}
        if self.rpc_id is not None and "rpc_id" not in payload:
            payload = {**payload, "rpc_id": self.rpc_id}
        self.queue.put(payload)


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
    stderr_path: str | None = None,
    bootstrap_address: tuple[str, int] | None = None,
) -> None:
    if stderr_path:
        import sys

        sys.stderr = open(stderr_path, "a", buffering=1)
    from hydraserve.diagnostics import enable_stall_diagnostics

    enable_stall_diagnostics(f"prefill:{config.prefill_device}")
    responses = _CorrelatedResponseSink(responses)
    backends = []
    try:
        import torch

        from hydraserve.cache import (
            CacheNamespace,
            CostAwarePrefixPolicy,
            GpuLinearStatePool,
            KVBlockManager,
            PagedKVCache,
            PrefixCache,
            plan_paged_kv_blocks,
        )
        from hydraserve.engine.pd_worker import PrefillWorker
        from hydraserve.engine.scheduler import RequestState
        from hydraserve.engine.sampling import sample_logits
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.router import Route
        from hydraserve.transfer import (
            NetworkBootstrapClient,
            TransferPipeline,
        )

        device = torch.device(config.prefill_device)
        torch.cuda.set_device(device)
        runtime = QwenTextRuntime.from_checkpoint(
            config.model_dir,
            device=device,
            dtype=torch.bfloat16,
            use_triton=True,
            use_flash_attention=config.use_flash_attention,
            requested_cache_tokens=config.cache_tokens,
        )
        requested_blocks = (
            config.cache_tokens + config.block_size - 1
        ) // config.block_size
        memory_plan = plan_paged_kv_blocks(
            runtime.config,
            requested_blocks,
            block_size=config.block_size,
            dtype=torch.bfloat16,
            device=device,
            state_slots=config.max_state_slots,
            state_workspace_slots=min(
                config.max_state_slots, config.max_decode_batch_size
            ),
            kv_quant=config.kv_quant,
        )
        blocks = memory_plan.planned_blocks
        if config.kv_headroom_blocks >= blocks:
            raise MemoryError("KV headroom consumes the memory-planned cache")
        revision = str(Path(config.model_dir).resolve())
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
            memory_plan=memory_plan,
            kv_quant=config.kv_quant,
        )
        namespaces = (namespace,) if isinstance(namespace, str) else tuple(namespace)
        if not namespaces:
            raise ValueError("prefill worker requires at least one decode namespace")
        workers = []
        bootstrap = (
            NetworkBootstrapClient(bootstrap_address)
            if bootstrap_address is not None
            else None
        )
        for worker_index, worker_namespace in enumerate(namespaces):
            backend = _make_pd_transfer_backend(
                config, worker_namespace, model=runtime.config
            )
            backends.append(backend)
            # ``dst_gpu`` is a logical mailbox id (decode index + 1) shared by
            # every prefill worker and the matching decode worker; it is not
            # the physical GPU number and must stay consistent across nP.
            workers.append(
                PrefillWorker(
                    runtime,
                    TransferPipeline(
                        backend,
                        src_gpu=0,
                        dst_gpu=worker_index + 1,
                        bootstrap=bootstrap,
                    ),
                    cache,
                )
            )

        # W4: the prefill worker also serves collocated short requests, so it
        # keeps a recurrent-state pool and per-request bookkeeping like the
        # decode worker.
        requests = {}
        states = {}
        state_pool = GpuLinearStatePool(
            config.max_state_slots,
            runtime.config,
            device=device,
            workspace_capacity=min(
                config.max_state_slots, config.max_decode_batch_size
            ),
        )
        state_capacity = state_pool.capacity_snapshot().total_slots
        reservations = set()
        from threading import Lock as _ThreadLock

        state_lock = _ThreadLock()

        def capacity_payload():
            capacity = cache.block_manager.capacity()
            live = len(set(states) | reservations)
            return {
                "kv_total_blocks": capacity.total_blocks,
                "kv_free_blocks": capacity.free_blocks,
                "state_total_slots": state_capacity,
                "state_free_slots": max(0, state_capacity - live),
                "kv_cache_stats": {**cache.stats(), **state_pool.stats()},
            }

        responses.put(
            {"op": "ready", "model_name": runtime.config.name, **capacity_payload()}
        )
        # W5: short-request operations (admission, decode steps, collocated
        # prepare) jump ahead of queued long prefills instead of stalling
        # behind the multi-second GPU work. A short-op budget keeps a
        # continuous short burst from starving the long prefill queue.
        from collections import deque
        from queue import Empty

        SHORT_OPS = {
            "decode",
            "reserve",
            "release",
            "collocated_prepare",
            "recover",
        }
        PREEMPTIBLE_OPS = {
            "decode",
            "reserve",
            "release",
            "collocated_prepare",
        }
        pending_short = deque()
        pending_long = deque()
        short_budget = 64

        def service_preemptible_short_ops() -> int:
            """Run bounded short decode work at a long-prefill chunk boundary."""

            serviced = 0
            deferred = []
            while serviced < config.prefill_preempt_max_ops:
                try:
                    queued = commands.get_nowait()
                except Empty:
                    break
                operation = queued.get("op")
                if operation not in PREEMPTIBLE_OPS:
                    deferred.append(queued)
                    continue
                if (
                    operation == "collocated_prepare"
                    and len(queued.get("token_ids", ())) > config.prefill_chunk_size
                ):
                    # Only genuinely short prepares may interrupt a long
                    # prefill.  Running another multi-chunk prompt recursively
                    # would merely invert the queue and could starve the outer
                    # request while consuming an additional state/KV reserve.
                    deferred.append(queued)
                    continue
                previous_rpc_id = responses.rpc_id
                responses.rpc_id = queued.get("rpc_id")
                try:
                    if operation == "reserve":
                        request_id = queued["request_id"]
                        total_tokens = len(queued["token_ids"]) + max(
                            0, queued["max_new_tokens"] - 1
                        )
                        required = cache.block_manager.blocks_required(total_tokens)
                        with state_lock:
                            live_requests = set(states) | reservations
                        response = {
                            "op": "admission",
                            "request_id": request_id,
                            "admitted": False,
                        }
                        if required > cache.block_manager.num_blocks:
                            response.update(
                                retryable=False,
                                reason=(
                                    f"request needs {required} KV blocks, worker capacity "
                                    f"is {cache.block_manager.num_blocks}"
                                ),
                            )
                        elif request_id in live_requests:
                            response["admitted"] = True
                        elif len(live_requests) >= state_capacity:
                            response.update(
                                retryable=True,
                                reason="recurrent-state slots are exhausted",
                            )
                        else:
                            try:
                                cache.allocate(
                                    request_id,
                                    len(queued["token_ids"]),
                                    reserve_tokens=total_tokens,
                                    token_ids=queued["token_ids"],
                                )
                            except MemoryError:
                                response.update(
                                    retryable=True,
                                    reason="prefill worker KV capacity is exhausted",
                                )
                            else:
                                with state_lock:
                                    reservations.add(request_id)
                                response["admitted"] = True
                        responses.put({**response, **capacity_payload()})
                    elif operation == "collocated_prepare":
                        request_id = queued["request_id"]
                        if request_id not in reservations:
                            raise RuntimeError(
                                "collocated prefill requires a KV reservation"
                            )
                        request = _request(
                            request_id,
                            queued["token_ids"],
                            queued["max_new_tokens"],
                            transferred=False,
                            route=Route.COLLOCATED,
                            sampling_params=queued.get("sampling_params"),
                        )
                        request.transition(RequestState.PREFILL_RUNNING)
                        input_ids = torch.tensor(
                            [request.token_ids],
                            device=runtime.input_device,
                            dtype=torch.long,
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
                        with state_lock:
                            requests[request_id] = request
                            states[request_id] = state_pool.install(request_id, state)
                        responses.put(
                            {
                                "op": "collocated_prepare",
                                "request_id": request_id,
                                "token_id": token_id,
                                "sample": sample,
                                "chunk_preempted": True,
                            }
                        )
                    elif operation == "decode":
                        request_ids = tuple(queued["request_ids"])
                        cache.block_manager.grow_many(
                            request_ids, additional_tokens=1
                        )
                        input_ids = torch.tensor(
                            [
                                requests[request_id].generated_token_ids[-1]
                                for request_id in request_ids
                            ],
                            device=runtime.input_device,
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
                            (
                                requests[request_id].sampling_params
                                for request_id in request_ids
                            ),
                            steps=(
                                len(requests[request_id].generated_token_ids)
                                for request_id in request_ids
                            ),
                        )
                        token_ids = tuple(sample.token_id for sample in samples)
                        for request_id, token_id in zip(
                            request_ids, token_ids, strict=True
                        ):
                            requests[request_id].generated_token_ids.append(token_id)
                        responses.put(
                            {
                                "op": "decode",
                                "request_ids": request_ids,
                                "token_ids": token_ids,
                                "samples": samples,
                                "chunk_preempted": True,
                                **capacity_payload(),
                            }
                        )
                    else:  # release
                        request_id = queued["request_id"]
                        with state_lock:
                            reservations.discard(request_id)
                            states.pop(request_id, None)
                            state_pool.free(request_id)
                            requests.pop(request_id, None)
                        cache.free(request_id)
                        responses.put(
                            {
                                "op": "release",
                                "request_id": request_id,
                                "chunk_preempted": True,
                                **capacity_payload(),
                            }
                        )
                    serviced += 1
                except Exception as exc:
                    if operation == "collocated_prepare":
                        request_id = queued.get("request_id")
                        reservations.discard(request_id)
                        state_pool.free(request_id)
                        states.pop(request_id, None)
                        requests.pop(request_id, None)
                        cache.free(request_id)
                    responses.put(
                        {
                            "op": "error",
                            "request_id": queued.get("request_id"),
                            "request_ids": queued.get("request_ids"),
                            "message": repr(exc),
                        }
                    )
                    serviced += 1
                finally:
                    responses.rpc_id = previous_rpc_id
            for queued in deferred:
                if queued.get("op") in SHORT_OPS:
                    pending_short.append(queued)
                else:
                    pending_long.append(queued)
            return serviced

        while True:
            while True:
                try:
                    queued = commands.get_nowait()
                except Empty:
                    break
                if queued.get("op") in SHORT_OPS:
                    pending_short.append(queued)
                else:
                    pending_long.append(queued)
            if pending_short and (short_budget > 0 or not pending_long):
                # Short ops always make progress: a drained budget must not
                # strand queued collocated work while no long op is pending
                # (that deadlocked the worker on commands.get()).
                command = pending_short.popleft()
                short_budget = max(0, short_budget - 1)
            elif pending_long:
                command = pending_long.popleft()
                short_budget = 64
            else:
                command = commands.get()
                if command.get("op") in SHORT_OPS:
                    short_budget = max(0, short_budget - 1)
                else:
                    short_budget = 64
            operation = command["op"]
            responses.rpc_id = command.get("rpc_id")
            if operation == "shutdown":
                with state_lock:
                    live_ids = tuple(set(states) | reservations)
                for request_id in live_ids:
                    cache.free(request_id)
                responses.put({"op": "shutdown"})
                return
            try:
                if operation == "prefill":
                    request_id = command["request_id"]
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
                    try:
                        result = workers[worker_index].process(
                            request,
                            n_minus_one=True,
                            chunk_size=config.prefill_chunk_size,
                            streamed_transfer=bool(
                                command.get("streamed_transfer", False)
                            ),
                            reuse_host_kv=bool(command.get("host_cache_hit", False)),
                            host_prefix_tokens=int(command.get("host_prefix_tokens", 0)),
                            transfer_target_bytes=config.transfer_target_bytes,
                            max_inflight_chunks=config.max_inflight_transfer_chunks,
                            chunk_yield_callback=service_preemptible_short_ops,
                        )
                    finally:
                        # The chunked transfer path extracts KV on a separate
                        # transfer stream and process() only synchronises that
                        # stream.  Freeing the request's KV blocks while the
                        # compute stream still has queued kernels referencing
                        # them lets the next request reuse (and overwrite) the
                        # blocks before those kernels run -> intermittent CUDA
                        # IMA / segfault under load.  Drain the device so block
                        # reuse is strictly ordered after every async access.
                        torch.cuda.synchronize(cache.device)
                        cache.free(request_id)
                    responses.put(
                        {
                            "op": "prefill",
                            "request_id": request_id,
                            "worker_index": worker_index,
                            "token_id": result.first_token_id,
                            "sample": result.sample,
                            "chunk_preemptions": result.chunk_preemptions,
                        }
                    )
                elif operation == "reserve":
                    request_id = command["request_id"]
                    total_tokens = len(command["token_ids"]) + max(
                        0, command["max_new_tokens"] - 1
                    )
                    required = cache.block_manager.blocks_required(total_tokens)
                    with state_lock:
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
                                    "reason": "prefill worker KV capacity is exhausted",
                                    **capacity_payload(),
                                }
                            )
                            continue
                        with state_lock:
                            reservations.add(request_id)
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                                **capacity_payload(),
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
                        [request.token_ids], device=runtime.input_device, dtype=torch.long
                    )
                    with torch.inference_mode():
                        logits, state = runtime.prefill(
                            input_ids,
                            chunk_size=config.prefill_chunk_size,
                            paged_cache=cache,
                            request_id=request_id,
                            chunk_callback=lambda _start, _end, _state: (
                                service_preemptible_short_ops()
                            ),
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
                    with state_lock:
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
                        device=runtime.input_device,
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
                            **capacity_payload(),
                        }
                    )
                elif operation == "release":
                    request_id = command["request_id"]
                    with state_lock:
                        reservations.discard(request_id)
                        states.pop(request_id, None)
                        state_pool.free(request_id)
                        requests.pop(request_id, None)
                    cache.free(request_id)
                    responses.put(
                        {"op": "release", "request_id": request_id, **capacity_payload()}
                    )
                elif operation == "recover":
                    request_id = command["request_id"]
                    if request_id not in reservations:
                        raise RuntimeError("recovery requires a KV/state reservation")
                    replay_token_ids = tuple(command["replay_token_ids"])
                    generated_token_ids = tuple(command["generated_token_ids"])
                    expected_replay = tuple(command["token_ids"]) + generated_token_ids[:-1]
                    if replay_token_ids != expected_replay:
                        raise RuntimeError("recovery replay does not match request history")
                    total_tokens = len(command["token_ids"]) + max(
                        0, command["max_new_tokens"] - 1
                    )
                    with state_lock:
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
                        [replay_token_ids], device=runtime.input_device, dtype=torch.long
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
                    with state_lock:
                        requests[request_id] = request
                        states[request_id] = state_pool.install(request_id, state)
                    responses.put(
                        {"op": "recover", "request_id": request_id, **capacity_payload()}
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
                    raise ValueError(f"unknown prefill-worker operation {operation!r}")
            except Exception as exc:
                if operation in {"collocated_prepare", "recover"}:
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
        for backend in backends:
            backend.close()


def _decode_worker(
    config: PDWorkerConfig,
    namespace: str,
    commands,
    responses,
    worker_index: int = 0,
    stderr_path: str | None = None,
    bootstrap_address: tuple[str, int] | None = None,
) -> None:
    if stderr_path:
        import sys

        sys.stderr = open(stderr_path, "a", buffering=1)
    from hydraserve.diagnostics import enable_stall_diagnostics

    enable_stall_diagnostics(f"decode:{config.decode_device}")
    responses = _CorrelatedResponseSink(responses)
    backend = None
    prepare_executor = None
    try:
        import torch

        from hydraserve.cache import (
            CacheNamespace,
            CostAwarePrefixPolicy,
            GpuLinearStatePool,
            KVBlockManager,
            PagedKVCache,
            PrefixCache,
            plan_paged_kv_blocks,
            HostPrefixCache,
        )
        from hydraserve.engine.pd_worker import DecodeWorker
        from hydraserve.engine.scheduler import RequestState
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.transfer import (
            NetworkBootstrapClient,
            TransferCancelledError,
            TransferPipeline,
        )

        device = torch.device(config.decode_device)
        torch.cuda.set_device(device)
        runtime = QwenTextRuntime.from_checkpoint(
            config.model_dir,
            device=device,
            dtype=torch.bfloat16,
            use_triton=True,
            # PARTIAL decode recomputes the whole prompt (full-attention KV must be
            # rebuilt from the transferred recurrent state). FlashAttention is what
            # makes that recompute fast; disabling it here silently falls back to the
            # O(n^2) Triton paged-attention path and made 32K PD ~28x slower than DP.
            use_flash_attention=config.use_flash_attention,
            requested_cache_tokens=config.cache_tokens,
        )
        requested_blocks = (
            config.cache_tokens + config.block_size - 1
        ) // config.block_size
        memory_plan = plan_paged_kv_blocks(
            runtime.config,
            requested_blocks,
            block_size=config.block_size,
            dtype=torch.bfloat16,
            device=device,
            state_slots=config.max_state_slots,
            state_workspace_slots=min(
                config.max_state_slots, config.max_decode_batch_size
            ),
            kv_quant=config.kv_quant,
        )
        blocks = memory_plan.planned_blocks
        if config.kv_headroom_blocks >= blocks:
            raise MemoryError("KV headroom consumes the memory-planned cache")
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
            memory_plan=memory_plan,
            kv_quant=config.kv_quant,
        )
        backend = _make_pd_transfer_backend(config, namespace, model=runtime.config)
        bootstrap = (
            NetworkBootstrapClient(bootstrap_address)
            if bootstrap_address is not None
            else None
        )
        host_cache = (
            HostPrefixCache(
                config.host_prefix_cache_bytes,
                block_size=config.block_size,
            )
            if config.host_prefix_cache_bytes > 0
            else None
        )
        worker = DecodeWorker(
            runtime,
            TransferPipeline(
                backend,
                src_gpu=0,
                dst_gpu=worker_index + 1,
                bootstrap=bootstrap,
            ),
            cache,
            host_cache=host_cache,
        )
        # Receive-side ring demultiplexing makes concurrent prepare safe.
        # Keep this limit independent from P-side chunk submission depth.
        prepare_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_prepares
        )
        requests = {}
        states = {}
        state_pool = GpuLinearStatePool(
            config.max_state_slots,
            runtime.config,
            device=device,
            workspace_capacity=min(
                config.max_state_slots, config.max_decode_batch_size
            ),
        )
        state_capacity = state_pool.capacity_snapshot().total_slots
        reservations = set()
        # Guards the request/state dicts shared with background prepare
        # threads (transfer+install runs concurrently with the decode loop).
        from threading import Lock as _ThreadLock

        state_lock = _ThreadLock()
        host_reservations = {}
        preparing = set()
        prepare_cancellations = {}

        def capacity_payload():
            with state_lock:
                capacity = cache.block_manager.capacity()
                live = len(set(states) | reservations)
            cache_stats = {**cache.stats(), **state_pool.stats()}
            if host_cache is not None:
                host_stats = host_cache.stats()
                cache_stats.update(
                    {
                        "host_prefix_entries": host_stats.entries,
                        "host_prefix_bytes": host_stats.bytes_used,
                        "host_prefix_hits": host_stats.hits,
                        "host_prefix_misses": host_stats.misses,
                        "host_prefix_evictions": host_stats.evictions,
                    }
                )
            return {
                "kv_total_blocks": capacity.total_blocks,
                "kv_free_blocks": capacity.free_blocks,
                "state_total_slots": state_capacity,
                "state_free_slots": max(0, state_capacity - live),
                "kv_cache_stats": cache_stats,
            }

        responses.put(
            {"op": "ready", "model_name": runtime.config.name, **capacity_payload()}
        )
        # A D-bound worker may run a collocated short/overflow prefill while it
        # already owns active decode requests.  Correlated parent RPCs can put
        # those decode commands into this queue concurrently; service a bounded
        # number at chunk boundaries so local prefill does not become a TPOT
        # blackout.  Non-decode commands retain FIFO order after the prefill.
        from collections import deque
        from queue import Empty

        deferred_commands = deque()

        def execute_decode(queued: dict, *, chunk_preempted: bool = False) -> None:
            request_ids = tuple(queued["request_ids"])
            cache.block_manager.grow_many(request_ids, additional_tokens=1)
            input_ids = torch.tensor(
                [
                    requests[request_id].generated_token_ids[-1]
                    for request_id in request_ids
                ],
                device=runtime.input_device,
                dtype=torch.long,
            ).unsqueeze(1)
            batch_states = [states[request_id] for request_id in request_ids]
            with state_lock:
                allow_cuda_graph = not preparing
            with torch.inference_mode():
                logits, _ = runtime.decode_batch(
                    input_ids,
                    batch_states,
                    cache,
                    request_ids,
                    use_cuda_graphs=allow_cuda_graph,
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
                    "chunk_preempted": chunk_preempted,
                    **capacity_payload(),
                }
            )

        def service_pending_decode() -> int:
            serviced = 0
            deferred = []
            while serviced < config.prefill_preempt_max_ops:
                try:
                    queued = commands.get_nowait()
                except Empty:
                    break
                if queued.get("op") != "decode":
                    deferred.append(queued)
                    continue
                previous_rpc_id = responses.rpc_id
                responses.rpc_id = queued.get("rpc_id")
                try:
                    execute_decode(queued, chunk_preempted=True)
                except Exception as exc:
                    responses.put(
                        {
                            "op": "error",
                            "request_ids": queued.get("request_ids"),
                            "message": repr(exc),
                        }
                    )
                finally:
                    responses.rpc_id = previous_rpc_id
                serviced += 1
            deferred_commands.extend(deferred)
            return serviced

        while True:
            command = (
                deferred_commands.popleft()
                if deferred_commands
                else commands.get()
            )
            operation = command["op"]
            responses.rpc_id = command.get("rpc_id")
            if operation == "shutdown":
                with state_lock:
                    active_prepares = tuple(prepare_cancellations.items())
                    active_prepare_ids = set(prepare_cancellations)
                    live_ids = tuple(
                        (set(states) | reservations) - active_prepare_ids
                    )
                    for _, cancel_event in active_prepares:
                        cancel_event.set()
                for request_id, _ in active_prepares:
                    worker.pipeline.cancel_receive(request_id)
                for request_id in live_ids:
                    cache.free(request_id)
                if host_cache is not None:
                    for request_id, lease in tuple(host_reservations.items()):
                        if request_id in preparing:
                            continue
                        host_cache.unpin(lease)
                        host_reservations.pop(request_id, None)
                responses.put({"op": "shutdown"})
                return
            try:
                if operation == "reserve":
                    request_id = command["request_id"]
                    host_prefix_tokens = (
                        host_cache.longest_prefix_tokens(
                            runtime.config.name, command["token_ids"]
                        )
                        if host_cache is not None
                        else 0
                    )
                    existing_lease = host_reservations.get(request_id)
                    if existing_lease is not None:
                        host_prefix_tokens = existing_lease.matched_tokens
                    host_cache_hit = host_prefix_tokens == len(command["token_ids"])
                    total_tokens = len(command["token_ids"]) + max(
                        0, command["max_new_tokens"] - 1
                    )
                    required = cache.block_manager.blocks_required(total_tokens)
                    with state_lock:
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
                                "host_cache_hit": host_cache_hit,
                                "host_prefix_tokens": host_prefix_tokens,
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
                        with state_lock:
                            reservations.add(request_id)
                        if host_cache is not None and host_prefix_tokens:
                            lease = host_cache.pin(
                                runtime.config.name, command["token_ids"]
                            )
                            host_prefix_tokens = lease.matched_tokens
                            host_cache_hit = host_prefix_tokens == len(
                                command["token_ids"]
                            )
                            if host_prefix_tokens:
                                host_reservations[request_id] = lease
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                                "host_cache_hit": host_cache_hit,
                                "host_prefix_tokens": host_prefix_tokens,
                                **capacity_payload(),
                            }
                        )
                elif operation == "prepare":
                    request_id = command["request_id"]
                    with state_lock:
                        duplicate_prepare = request_id in preparing
                    if duplicate_prepare:
                        responses.put(
                            {
                                "op": "error",
                                "request_id": request_id,
                                "message": "request prepare is already in progress",
                            }
                        )
                        continue
                    _pd_protocol_trace(
                        "d_prepare_command", request_id, worker_index=worker_index
                    )
                    host_lease = host_reservations.get(request_id)
                    request = _request(
                        request_id,
                        command["token_ids"],
                        command["max_new_tokens"],
                        transferred=True,
                        sampling_params=command.get("sampling_params"),
                    )
                    cancel_event = Event()

                    # Overlap the transfer/install (SHM receive, KV install,
                    # N-1 replay) with the decode loop: other requests keep
                    # decoding on the default stream while this request lands
                    # on DecodeWorker's dedicated install stream.
                    def prepare_in_background(
                        request=request,
                        request_id=request_id,
                        command=command,
                        host_lease=host_lease,
                        rpc_id=command.get("rpc_id"),
                        cancel_event=cancel_event,
                    ):
                        responses.rpc_id = rpc_id
                        failed = False
                        try:
                            def receiver_armed() -> None:
                                _pd_protocol_trace(
                                    "d_receiver_armed",
                                    request_id,
                                    worker_index=worker_index,
                                )
                                responses.put(
                                    {
                                        "op": "prepare_armed",
                                        "request_id": request_id,
                                    }
                                )

                            prepared = worker.receive_and_prepare(
                                request,
                                timeout=command.get("timeout"),
                                preallocated=request_id in reservations,
                                chunk_size=config.prefill_chunk_size,
                                streamed_transfer=bool(
                                    command.get("streamed_transfer", False)
                                ),
                                host_prefix_tokens=int(
                                    command.get("host_prefix_tokens", 0)
                                ),
                                host_match=host_lease,
                                receiver_armed_callback=receiver_armed,
                                cancel_event=cancel_event,
                            )
                            _pd_protocol_trace(
                                "d_transfer_installed",
                                request_id,
                                worker_index=worker_index,
                            )
                            with state_lock:
                                if cancel_event.is_set():
                                    raise TransferCancelledError(
                                        "decode prepare was cancelled before install"
                                    )
                                requests[request_id] = request
                                states[request_id] = state_pool.install(
                                    request_id, prepared.state
                                )
                            if cancel_event.is_set():
                                raise TransferCancelledError(
                                    "decode prepare was cancelled after install"
                                )
                            responses.put(
                                {
                                    "op": "prepare",
                                    "request_id": request_id,
                                    "token_id": prepared.first_token_id,
                                    "replay_consistent": prepared.replay_consistent,
                                }
                            )
                        except Exception as exc:
                            failed = True
                            _pd_protocol_trace(
                                "d_prepare_failed",
                                request_id,
                                worker_index=worker_index,
                                error=repr(exc),
                            )
                            responses.put(
                                {
                                    "op": "error",
                                    "request_id": request_id,
                                    "message": repr(exc),
                                }
                            )
                        finally:
                            with state_lock:
                                preparing.discard(request_id)
                                prepare_cancellations.pop(request_id, None)
                                should_cleanup = failed or cancel_event.is_set()
                                if should_cleanup:
                                    reservations.discard(request_id)
                                    states.pop(request_id, None)
                                    state_pool.free(request_id)
                                    requests.pop(request_id, None)
                            if should_cleanup:
                                try:
                                    # Mark all remaining request keys as
                                    # cancelled before freeing device state.
                                    # The resident ring dispatcher then drains
                                    # P-side chunks that were already in flight.
                                    worker.pipeline.cancel_receive(request_id)
                                except Exception as cancel_exc:
                                    _pd_protocol_trace(
                                        "d_prepare_cancel_failed",
                                        request_id,
                                        worker_index=worker_index,
                                        error=repr(cancel_exc),
                                    )
                                cache.free(request_id)
                            lease = host_reservations.pop(request_id, None)
                            if lease is not None and host_cache is not None:
                                host_cache.unpin(lease)
                            responses.rpc_id = None

                    with state_lock:
                        preparing.add(request_id)
                        prepare_cancellations[request_id] = cancel_event
                    try:
                        prepare_executor.submit(prepare_in_background)
                    except Exception:
                        with state_lock:
                            preparing.discard(request_id)
                            prepare_cancellations.pop(request_id, None)
                        lease = host_reservations.pop(request_id, None)
                        if lease is not None and host_cache is not None:
                            host_cache.unpin(lease)
                        raise
                elif operation == "cancel_prepare":
                    request_id = command["request_id"]
                    _pd_protocol_trace(
                        "d_prepare_cancel", request_id, worker_index=worker_index
                    )
                    with state_lock:
                        cancel_event = prepare_cancellations.get(request_id)
                        if cancel_event is not None:
                            cancel_event.set()
                    if cancel_event is not None:
                        worker.pipeline.cancel_receive(request_id)
                    if not command.get("fire_and_forget", False):
                        responses.put(
                            {
                                "op": "cancel_prepare",
                                "request_id": request_id,
                                "cancelled": cancel_event is not None,
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
                        [request.token_ids], device=runtime.input_device, dtype=torch.long
                    )
                    with torch.inference_mode():
                        logits, state = runtime.prefill(
                            input_ids,
                            chunk_size=config.prefill_chunk_size,
                            paged_cache=cache,
                            request_id=request_id,
                            chunk_callback=lambda _start, _end, _state: (
                                service_pending_decode()
                            ),
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
                    with state_lock:
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
                    with state_lock:
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
                        [replay_token_ids], device=runtime.input_device, dtype=torch.long
                    )
                    with torch.inference_mode():
                        _, state = runtime.prefill(
                            replay,
                            chunk_size=config.prefill_chunk_size,
                            paged_cache=cache,
                            request_id=request_id,
                            chunk_callback=lambda _start, _end, _state: (
                                service_pending_decode()
                            ),
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
                    with state_lock:
                        requests[request_id] = request
                        states[request_id] = state_pool.install(request_id, state)
                    responses.put(
                        {"op": "recover", "request_id": request_id, **capacity_payload()}
                    )
                elif operation == "decode":
                    execute_decode(command)
                elif operation == "release":
                    request_id = command["request_id"]
                    with state_lock:
                        is_preparing = request_id in preparing
                        cancel_event = prepare_cancellations.get(request_id)
                        if is_preparing and cancel_event is not None:
                            cancel_event.set()
                        if not is_preparing:
                            reservations.discard(request_id)
                            states.pop(request_id, None)
                            state_pool.free(request_id)
                            requests.pop(request_id, None)
                    if is_preparing:
                        worker.pipeline.cancel_receive(request_id)
                    else:
                        lease = host_reservations.pop(request_id, None)
                        if lease is not None and host_cache is not None:
                            host_cache.unpin(lease)
                        cache.free(request_id)
                    responses.put(
                        {"op": "release", "request_id": request_id, **capacity_payload()}
                    )
                elif operation == "prefix_probe":
                    match = cache.probe_prefix(command["token_ids"])
                    host_matched = (
                        host_cache.longest_prefix_tokens(
                            runtime.config.name, command["token_ids"]
                        )
                        if host_cache is not None
                        else 0
                    )
                    responses.put(
                        {
                            "op": "prefix_probe",
                            "request_id": command["request_id"],
                            "matched_tokens": max(match.matched_tokens, host_matched),
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
        if prepare_executor is not None:
            prepare_executor.shutdown(wait=True, cancel_futures=True)
        if backend is not None:
            backend.close()


class DisaggregatedGenerationBackend:
    """GenerationBackend backed by persistent prefill and decode GPU processes."""

    release_parallelism = 1

    def prefill_admission_tokens(self, request: ServingRequest) -> int:
        """Charge one executable prefill chunk rather than the full prompt."""

        return min(len(request.token_ids), self.config.prefill_chunk_size)

    def __init__(
        self,
        config: PDWorkerConfig,
        *,
        startup_timeout: float = 180.0,
        operation_timeout: float = 600.0,
        receiver_arm_timeout: float = 10.0,
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
            config.max_decode_batch_size,
            config.prefix_cache_min_frequency,
            config.transfer_target_bytes,
            config.max_inflight_transfer_chunks,
            config.max_concurrent_prepares,
            config.shm_ring_slots,
            config.shm_ring_slot_bytes,
            config.prefill_preempt_max_ops,
        ) <= 0:
            raise ValueError("cache limits must be positive")
        if config.prefix_cache_blocks < 0:
            raise ValueError("prefix cache blocks cannot be negative")
        if config.host_prefix_cache_bytes < 0:
            raise ValueError("host prefix cache bytes cannot be negative")
        if config.transfer_backend not in {"shm-ring", "shm"}:
            raise ValueError("transfer_backend must be shm-ring or shm")
        if config.transfer_quant not in {None, "int8"}:
            raise ValueError("transfer_quant must be None or int8")
        if (
            max_worker_restarts <= 0
            or worker_restart_backoff_s < 0
            or receiver_arm_timeout <= 0
            or operation_timeout <= 0
        ):
            raise ValueError("invalid worker recovery policy")
        total_blocks = (
            config.cache_tokens + config.block_size - 1
        ) // config.block_size
        if not 0 <= config.kv_headroom_blocks < total_blocks:
            raise ValueError("KV headroom must be below physical cache blocks")
        self.config = config
        self.supports_async_prefill = True
        self.operation_timeout = operation_timeout
        self.receiver_arm_timeout = min(receiver_arm_timeout, operation_timeout)
        self.startup_timeout = startup_timeout
        self.max_worker_restarts = max_worker_restarts
        self.worker_restart_backoff_s = worker_restart_backoff_s
        self.namespace = f"hydraserve-pd-{uuid4().hex}"
        self._bootstrap_server = None
        self._bootstrap_address = None
        try:
            from hydraserve.transfer import BootstrapServer

            self._bootstrap_server = BootstrapServer().start()
            self._bootstrap_address = self._bootstrap_server.address
        except PermissionError:
            # Restricted sandboxes can still use the SHM manifest fallback.
            self._bootstrap_server = None
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
        self._pd_executor = ThreadPoolExecutor(max_workers=4)
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
        self._host_prefix_tokens: dict[int, int] = {}
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
                self._host_prefix_tokens[request.request_id] = int(
                    result.get("host_prefix_tokens", 0)
                )
                if request.route is None:
                    request.route = Route.PD_DISAGGREGATED.value
                    request.route_reason = "fixed_pd"
                    request.worker_id = 0
                    request.worker_pool = "decode"
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
        host_prefix_tokens = self._host_prefix_tokens.get(request.request_id, 0)
        host_cache_hit = host_prefix_tokens == len(request.token_ids)
        command = {
            "op": "prefill",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
            "sampling_params": request.sampling_params,
            "streamed_transfer": (
                not host_cache_hit
                and os.environ.get("HYDRASERVE_CHUNKED_TRANSFER", "1") != "0"
            ),
            "host_cache_hit": host_cache_hit,
            "host_prefix_tokens": host_prefix_tokens,
        }
        # Start the D-side receive thread before prefill compute. Queue.put()
        # alone is not a receiver-ready handshake: the acknowledgement below
        # is emitted from the actual D prepare executor thread.
        receiver_armed = Event()
        prepare_future = self._pd_executor.submit(
            self._decode_rpc,
            {
                **command,
                "op": "prepare",
                "timeout": self.operation_timeout,
            },
            "prepare",
            request.request_id,
            receiver_armed=receiver_armed,
        )
        _pd_protocol_trace("coordinator_prepare_dispatched", request.request_id)
        try:
            arm_timeout = getattr(
                self, "receiver_arm_timeout", min(10.0, self.operation_timeout)
            )
            if not receiver_armed.wait(arm_timeout):
                if prepare_future.done():
                    prepare_future.result()
                raise TimeoutError("decode receiver was not armed before PD transfer")
            _pd_protocol_trace("coordinator_receiver_armed", request.request_id)
            _pd_protocol_trace("coordinator_prefill_started", request.request_id)
            prefill_future = self._pd_executor.submit(
                self._prefill_rpc, command, request.request_id
            )
            # Observe both halves instead of blocking on P first. If D rejects
            # an install, its cleanup marks the remaining ring keys cancelled
            # and this coordinator can fail immediately rather than waiting
            # for a producer stuck behind ring backpressure.
            done, _ = wait(
                (prefill_future, prepare_future),
                return_when=FIRST_COMPLETED,
            )
            if prepare_future in done:
                prepared = prepare_future.result()
            if prefill_future in done:
                result = prefill_future.result()
            result = prefill_future.result()
            _pd_protocol_trace("coordinator_prefill_finished", request.request_id)
            prepared = prepare_future.result()
            _pd_protocol_trace("coordinator_prepare_finished", request.request_id)
        except Exception:
            if not prepare_future.done() and self._decode.is_alive():
                # Fixed-PD uses a FIFO response queue guarded by _decode_lock.
                # This command must therefore be response-free; the cancelled
                # prepare's own terminal error wakes its existing RPC waiter.
                self._decode_commands.put(
                    {
                        "op": "cancel_prepare",
                        "request_id": request.request_id,
                        "fire_and_forget": True,
                    }
                )
            raise
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
        if not known:
            return
        try:
            result = self._decode_rpc(
                {"op": "release", "request_id": request_id}, "release", request_id
            )
            self._update_capacity(result)
        finally:
            with self._decode_lock:
                self._admitted_requests.discard(request_id)
                self._reserved_blocks.pop(request_id, None)
                self._host_prefix_tokens.pop(request_id, None)
                self._lost_requests.discard(request_id)

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
            self._host_prefix_tokens.pop(request_id, None)
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
        self,
        command: dict,
        expected_op: str,
        request_id: int | None = None,
        *,
        receiver_armed: Event | None = None,
    ) -> dict:
        failure = None
        with self._decode_lock:
            try:
                if not self._decode.is_alive():
                    raise PDWorkerUnavailableError("decode worker is not running")
                self._decode_commands.put(command)
                while True:
                    result = self._get_worker_response(
                        "decode", self.operation_timeout
                    )
                    if result.get("op") != "prepare_armed":
                        break
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
                self._worker_log_path("prefill"),
                self._bootstrap_address,
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
                0,
                self._worker_log_path("decode"),
                self._bootstrap_address,
            ),
            name="hydraserve-decode",
        )

    def _worker_log_path(self, kind: str) -> str | None:
        if not self.config.worker_log_dir:
            return None
        Path(self.config.worker_log_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.config.worker_log_dir) / f"{kind}.log")

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
        if self._bootstrap_server is not None:
            self._bootstrap_server.close()
            self._bootstrap_server = None
        self._pd_executor.shutdown(wait=not force, cancel_futures=force)

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
    prefill_short_collocated: int = 0
    prefill_chunk_preemptions: int = 0


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
        receiver_arm_timeout: float = 10.0,
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
            receiver_arm_timeout=receiver_arm_timeout,
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
                request.worker_pool = "decode"
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
