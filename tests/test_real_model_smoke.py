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
        os.environ.get("HYDRASERVE_RUN_REAL_MODEL") != "1",
        reason="set HYDRASERVE_RUN_REAL_MODEL=1 to load the 8.8GB checkpoint",
    ),
]


def test_real_qwen35_4b_single_token() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
    )
    manager = KVBlockManager(16, block_size=16)
    cache = PagedKVCache(runtime.config, manager, device="cuda:0", dtype=torch.bfloat16)
    cache.allocate(0, 1)
    with torch.inference_mode():
        logits, state = runtime.forward(
            torch.tensor([[1]], device="cuda:0"), paged_cache=cache, request_id=0
        )
    assert logits.shape == (1, 1, runtime.config.vocab_size)
    assert torch.isfinite(logits).all()
    assert state.sequence_length == 1
