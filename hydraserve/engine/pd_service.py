"""Persistent two-process PARTIAL_TRANSFER serving backend."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
from threading import Lock
from uuid import uuid4

from hydraserve.engine.serving_loop import (
    AdmissionDecision,
    BackendCapacity,
    ServingRequest,
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


def _request(request_id: int, token_ids, max_new_tokens: int, *, transferred: bool):
    from hydraserve.engine.scheduler import Request, RequestState
    from hydraserve.router import Route

    request = Request(
        request_id,
        tuple(token_ids),
        max_new_tokens,
        Route.PD_DISAGGREGATED,
    )
    if transferred:
        request.transition(RequestState.PREFILL_RUNNING)
        request.transition(RequestState.TRANSFER_PENDING)
    return request


def _prefill_worker(config: PDWorkerConfig, namespace: str, commands, responses) -> None:
    backend = None
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
        backend = SharedMemoryTransferBackend(namespace=namespace)
        worker = PrefillWorker(runtime, TransferPipeline(backend, src_gpu=0, dst_gpu=1))
        responses.put({"op": "ready", "model_name": runtime.config.name})
        while True:
            command = commands.get()
            if command["op"] == "shutdown":
                responses.put({"op": "shutdown"})
                return
            request_id = command["request_id"]
            try:
                request = _request(
                    request_id,
                    command["token_ids"],
                    command["max_new_tokens"],
                    transferred=False,
                )
                result = worker.process(
                    request,
                    n_minus_one=True,
                    chunk_size=config.prefill_chunk_size,
                )
                responses.put(
                    {
                        "op": "prefill",
                        "request_id": request_id,
                        "token_id": result.first_token_id,
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
        if backend is not None:
            backend.close()


def _decode_worker(config: PDWorkerConfig, namespace: str, commands, responses) -> None:
    backend = None
    try:
        import torch

        from hydraserve.cache import KVBlockManager, PagedKVCache
        from hydraserve.engine.pd_worker import DecodeWorker
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
        cache = PagedKVCache(
            runtime.config,
            KVBlockManager(blocks, block_size=config.block_size),
            device=device,
            dtype=torch.bfloat16,
        )
        backend = SharedMemoryTransferBackend(namespace=namespace)
        worker = DecodeWorker(
            runtime,
            TransferPipeline(backend, src_gpu=0, dst_gpu=1),
            cache,
        )
        requests = {}
        states = {}
        reservations = set()
        responses.put({"op": "ready", "model_name": runtime.config.name})
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
                            }
                        )
                    elif request_id in live_requests:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                            }
                        )
                    elif len(live_requests) >= config.max_state_slots:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": False,
                                "retryable": True,
                                "reason": "recurrent-state slots are exhausted",
                            }
                        )
                    elif required > cache.block_manager.num_free_blocks:
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": False,
                                "retryable": True,
                                "reason": (
                                    f"request needs {required} KV blocks, only "
                                    f"{cache.block_manager.num_free_blocks} are free"
                                ),
                            }
                        )
                    else:
                        cache.allocate(
                            request_id,
                            len(command["token_ids"]),
                            reserve_tokens=total_tokens,
                        )
                        reservations.add(request_id)
                        responses.put(
                            {
                                "op": "admission",
                                "request_id": request_id,
                                "admitted": True,
                            }
                        )
                elif operation == "prepare":
                    request_id = command["request_id"]
                    request = _request(
                        request_id,
                        command["token_ids"],
                        command["max_new_tokens"],
                        transferred=True,
                    )
                    prepared = worker.receive_and_prepare(
                        request,
                        timeout=command.get("timeout"),
                        preallocated=request_id in reservations,
                    )
                    requests[request_id] = request
                    states[request_id] = prepared.state
                    responses.put(
                        {
                            "op": "prepare",
                            "request_id": request_id,
                            "token_id": prepared.first_token_id,
                        }
                    )
                elif operation == "decode":
                    request_ids = tuple(command["request_ids"])
                    for request_id in request_ids:
                        cache.reserve_append(request_id)
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
                    token_ids = tuple(
                        int(token) for token in logits[:, -1].argmax(dim=-1).tolist()
                    )
                    for request_id, token_id in zip(request_ids, token_ids, strict=True):
                        requests[request_id].generated_token_ids.append(token_id)
                    responses.put(
                        {"op": "decode", "request_ids": request_ids, "token_ids": token_ids}
                    )
                elif operation == "release":
                    request_id = command["request_id"]
                    reservations.discard(request_id)
                    states.pop(request_id, None)
                    requests.pop(request_id, None)
                    cache.free(request_id)
                    responses.put({"op": "release", "request_id": request_id})
                else:
                    raise ValueError(f"unknown decode-worker operation {operation!r}")
            except Exception as exc:
                if operation == "prepare":
                    reservations.discard(command.get("request_id"))
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
        ) <= 0:
            raise ValueError("cache limits must be positive")
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
        except Exception:
            self.close(force=True)
            raise

    def admit(self, request: ServingRequest) -> AdmissionDecision:
        total_tokens = len(request.token_ids) + max(0, request.max_new_tokens - 1)
        required_blocks = (total_tokens + self.config.block_size - 1) // self.config.block_size
        command = {
            "op": "reserve",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
        }
        with self._decode_lock:
            if request.request_id in self._admitted_requests:
                return AdmissionDecision.accept()
            self._decode_commands.put(command)
            result = self._get(self._decode_responses, self.operation_timeout)
            self._check(result, "admission", request.request_id)
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

    def prefill(self, request: ServingRequest) -> int:
        decision = self.admit(request)
        if not decision.admitted:
            raise MemoryError(decision.reason or "request cannot be admitted")
        command = {
            "op": "prefill",
            "request_id": request.request_id,
            "token_ids": request.token_ids,
            "max_new_tokens": request.max_new_tokens,
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
        return int(prepared["token_id"])

    def decode(self, requests: tuple[ServingRequest, ...]) -> tuple[int, ...]:
        request_ids = tuple(request.request_id for request in requests)
        with self._decode_lock:
            self._decode_commands.put({"op": "decode", "request_ids": request_ids})
            result = self._get(self._decode_responses, self.operation_timeout)
        self._check(result, "decode")
        if tuple(result["request_ids"]) != request_ids:
            raise RuntimeError("decode worker returned a different request batch")
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

    def capacity(self) -> BackendCapacity:
        total_blocks = (
            self.config.cache_tokens + self.config.block_size - 1
        ) // self.config.block_size
        with self._decode_lock:
            allocated_blocks = sum(self._reserved_blocks.values())
            allocated_slots = len(self._admitted_requests)
        return BackendCapacity(
            kv_total_blocks=total_blocks,
            kv_free_blocks=max(0, total_blocks - allocated_blocks),
            state_total_slots=self.config.max_state_slots,
            state_free_slots=max(0, self.config.max_state_slots - allocated_slots),
        )

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
