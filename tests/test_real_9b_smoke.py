from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache
from hydraserve.model.runtime import QwenTextRuntime


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_REAL_9B") != "1",
        reason="set HYDRASERVE_RUN_REAL_9B=1 to load the 19GB checkpoint",
    ),
]


def test_real_qwen35_9b_chunked_prefill_with_independent_lm_head() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.5-9B",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
    )
    assert "lm_head.weight" in runtime.weights
    assert runtime._output_weight().data_ptr() != runtime.weights[
        "model.language_model.embed_tokens.weight"
    ].data_ptr()
    cache = PagedKVCache(
        runtime.config,
        KVBlockManager(16, block_size=16),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([[1, 42, 17, 9]], device="cuda:0")
    cache.allocate(0, token_ids.shape[1])
    with torch.inference_mode():
        logits, state = runtime.prefill(
            token_ids,
            chunk_size=2,
            paged_cache=cache,
            request_id=0,
        )
    assert logits.shape == (1, 2, runtime.config.vocab_size)
    assert torch.isfinite(logits).all()
    assert state.sequence_length == token_ids.shape[1]
