from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache
from hydraserve.model.runtime import QwenTextRuntime
from hydraserve.model.weights import LANGUAGE_PREFIX, PackedInt4Weight


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_REAL_27B_AWQ") != "1",
        reason="set HYDRASERVE_RUN_REAL_27B_AWQ=1 to load the 26GB checkpoint",
    ),
]


def test_real_qwen36_27b_awq_single_token() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.cuda.set_device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
    )
    embedding = runtime.weights[f"{LANGUAGE_PREFIX}.embed_tokens.weight"]
    assert embedding.device.type == "cpu"
    assert runtime.weights["lm_head.weight"].device.type == "cuda"
    assert any(isinstance(weight, PackedInt4Weight) for weight in runtime.weights.values())
    cache = PagedKVCache(
        runtime.config,
        KVBlockManager(16, block_size=16),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    cache.allocate(0, 1)
    with torch.inference_mode():
        logits, state = runtime.forward(
            torch.tensor([[1]], device="cuda:0"),
            paged_cache=cache,
            request_id=0,
        )
    assert logits.shape == (1, 1, runtime.config.vocab_size)
    assert torch.isfinite(logits).all()
    assert state.sequence_length == 1
    cache.reserve_append(0)
    with torch.inference_mode():
        next_logits, state = runtime.forward(
            logits[:, -1].argmax(dim=-1, keepdim=True),
            state,
            paged_cache=cache,
            request_id=0,
        )
    assert torch.isfinite(next_logits).all()
    assert state.sequence_length == 2
    assert torch.cuda.max_memory_allocated() < torch.cuda.get_device_properties("cuda:0").total_memory
