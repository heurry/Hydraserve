from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty
from uuid import uuid4

import pytest

torch = pytest.importorskip("torch")


MODEL_DIR = "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B"
PROMPT = (1, 42, 17, 9)


def _request_in_transfer_state():
    from hydraserve.engine import CentralScheduler, RequestState

    request = CentralScheduler().submit(PROMPT, max_new_tokens=4)
    request.transition(RequestState.PREFILL_RUNNING)
    request.transition(RequestState.TRANSFER_PENDING)
    return request


def _prefill_process(namespace: str, ready, consumed, results) -> None:
    try:
        import torch

        from hydraserve.engine import CentralScheduler, PrefillWorker
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.transfer import SharedMemoryTransferBackend, TransferPipeline

        torch.cuda.set_device(0)
        runtime = QwenTextRuntime.from_checkpoint(
            MODEL_DIR,
            device="cuda:0",
            dtype=torch.bfloat16,
            use_triton=True,
            use_flash_attention=False,
        )
        backend = SharedMemoryTransferBackend(namespace=namespace)
        request = CentralScheduler().submit(PROMPT, max_new_tokens=4)
        result = PrefillWorker(runtime, TransferPipeline(backend)).process(request)
        layer = runtime.config.linear_layer_indices[0]
        results.put(
            (
                "prefill",
                result.first_token_id,
                float(result.state.recurrent[layer].sum().cpu()),
                float(result.state.convolution[layer].float().sum().cpu()),
            )
        )
        ready.set()
        if not consumed.wait(120):
            raise TimeoutError("decode worker did not consume shared memory")
        backend.close()
    except Exception as exc:
        results.put(("prefill_error", repr(exc)))
        raise


def _decode_process(namespace: str, ready, consumed, results) -> None:
    try:
        import torch

        from hydraserve.cache import KVBlockManager, PagedKVCache
        from hydraserve.engine import DecodeWorker
        from hydraserve.model.runtime import QwenTextRuntime
        from hydraserve.transfer import SharedMemoryTransferBackend, TransferPipeline

        torch.cuda.set_device(1)
        runtime = QwenTextRuntime.from_checkpoint(
            MODEL_DIR,
            device="cuda:1",
            dtype=torch.bfloat16,
            use_triton=True,
            use_flash_attention=False,
        )
        cache = PagedKVCache(
            runtime.config,
            KVBlockManager(32, block_size=16),
            device="cuda:1",
            dtype=torch.bfloat16,
        )
        if not ready.wait(120):
            raise TimeoutError("prefill worker did not publish shared memory")
        backend = SharedMemoryTransferBackend(namespace=namespace)
        request = _request_in_transfer_state()
        prepared = DecodeWorker(
            runtime, TransferPipeline(backend), cache
        ).receive_and_prepare(request, timeout=30)
        layer = runtime.config.linear_layer_indices[0]
        restored_recurrent = float(prepared.state.recurrent[layer].sum().cpu())
        restored_convolution = float(prepared.state.convolution[layer].sum().cpu())
        cache.reserve_append(request.request_id)
        with torch.inference_mode():
            next_logits, prepared.state = runtime.forward(
                torch.tensor([[prepared.first_token_id]], device="cuda:1"),
                prepared.state,
                paged_cache=cache,
                request_id=request.request_id,
            )
        assert torch.isfinite(next_logits).all()
        next_token = int(next_logits[0, -1].argmax())
        results.put(
            (
                "decode",
                prepared.first_token_id,
                restored_recurrent,
                restored_convolution,
                float(prepared.state.recurrent[layer].sum().cpu()),
                float(prepared.state.convolution[layer].sum().cpu()),
                prepared.state.sequence_length,
                next_token,
            )
        )
        consumed.set()
        backend.close()
    except Exception as exc:
        results.put(("decode_error", repr(exc)))
        consumed.set()
        raise


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_PD_E2E") != "1",
        reason="set HYDRASERVE_RUN_PD_E2E=1 to load one 4B model per GPU",
    ),
]


def test_real_two_gpu_partial_transfer() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    context = mp.get_context("spawn")
    namespace = f"hydraserve-e2e-{uuid4().hex}"
    ready = context.Event()
    consumed = context.Event()
    results = context.Queue()
    prefill = context.Process(
        target=_prefill_process, args=(namespace, ready, consumed, results)
    )
    decode = context.Process(
        target=_decode_process, args=(namespace, ready, consumed, results)
    )
    decode.start()
    prefill.start()
    prefill.join(150)
    decode.join(150)
    for process in (prefill, decode):
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert prefill.exitcode == 0
    assert decode.exitcode == 0
    records = {}
    for _ in range(2):
        try:
            record = results.get(timeout=5)
        except Empty as exc:
            raise AssertionError("a PD worker did not report its result") from exc
        records[record[0]] = record[1:]
    assert set(records) == {"prefill", "decode"}
    first_token, recurrent_sum, convolution_sum = records["prefill"]
    (
        restored_token,
        restored_recurrent,
        restored_convolution,
        advanced_recurrent,
        advanced_convolution,
        sequence_length,
        next_token,
    ) = records["decode"]
    assert restored_token == first_token
    assert restored_recurrent == pytest.approx(recurrent_sum, abs=1e-4)
    assert restored_convolution == pytest.approx(convolution_sum, abs=1e-4)
    assert advanced_recurrent != pytest.approx(restored_recurrent, abs=1e-7)
    assert advanced_convolution != pytest.approx(restored_convolution, abs=1e-7)
    assert sequence_length == len(PROMPT) + 1
    assert isinstance(next_token, int)
