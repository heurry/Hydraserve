from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import (
    GpuLinearStatePool,
    KVBlockManager,
    PagedKVCache,
    plan_paged_kv_blocks,
)
from hydraserve.model.runtime import QwenTextRuntime


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_REAL_27B_FP8") != "1",
        reason="set HYDRASERVE_RUN_REAL_27B_FP8=1 to load the 27B checkpoint",
    ),
]


def test_real_27b_fp8_prefill_and_pooled_decode() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
        requested_cache_tokens=65_536,
    )
    embedding = runtime.weights["model.language_model.embed_tokens.weight"]
    lm_head = runtime.weights["lm_head.weight"]
    assert embedding.device.type == "cpu"
    if torch.cuda.get_device_properties(0).total_memory <= 32 * 1024**3:
        assert lm_head.device.type == "cpu"
    memory_plan = plan_paged_kv_blocks(
        runtime.config,
        4096,
        block_size=16,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    assert memory_plan.planned_blocks == memory_plan.requested_blocks == 4096
    cache = PagedKVCache(
        runtime.config,
        KVBlockManager(memory_plan.planned_blocks, block_size=16),
        device="cuda:0",
        dtype=torch.bfloat16,
        memory_plan=memory_plan,
    )
    pool = GpuLinearStatePool(
        1, runtime.config, device="cuda:0", workspace_capacity=1
    )
    cache.allocate(1, 1, reserve_tokens=2)
    with torch.inference_mode():
        logits, state = runtime.forward(
            torch.tensor([[1]], device=runtime.input_device),
            paged_cache=cache,
            request_id=1,
        )
    state = pool.install(1, state)
    cache.reserve_append(1)
    with torch.inference_mode():
        decoded, _ = runtime.decode_batch(
            logits[:, -1]
            .argmax(dim=-1)
            .reshape(1, 1)
            .to(runtime.input_device),
            [state],
            cache,
            (1,),
        )
    assert decoded.shape == (1, 1, runtime.config.vocab_size)
    assert bool(torch.isfinite(decoded).all())
    assert state.sequence_length == 2
