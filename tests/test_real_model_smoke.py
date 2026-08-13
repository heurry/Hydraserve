from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import GpuLinearStatePool, KVBlockManager, PagedKVCache
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
    assert state.keys == {}


def test_real_qwen35_4b_pooled_batch_decode_matches_sequential() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
    )

    def prepare():
        cache = PagedKVCache(
            runtime.config,
            KVBlockManager(32, block_size=16),
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        states = []
        for request_id, prompt in ((10, (1, 2, 3)), (11, (4, 5))):
            cache.allocate(request_id, len(prompt), reserve_tokens=len(prompt) + 1)
            with torch.inference_mode():
                _, state = runtime.forward(
                    torch.tensor([prompt], device="cuda:0"),
                    paged_cache=cache,
                    request_id=request_id,
                )
            states.append(state)
        return cache, states

    batch_cache, batch_states = prepare()
    sequential_cache, sequential_states = prepare()
    pool = GpuLinearStatePool(
        2, runtime.config, device="cuda:0", workspace_capacity=2
    )
    batch_states = [
        pool.install(request_id, state)
        for request_id, state in zip((10, 11), batch_states, strict=True)
    ]
    for cache in (batch_cache, sequential_cache):
        cache.reserve_append(10)
        cache.reserve_append(11)
    tokens = torch.tensor([[7], [8]], device="cuda:0")
    with torch.inference_mode():
        actual, _ = runtime.decode_batch(
            tokens, batch_states, batch_cache, (10, 11)
        )
        expected = torch.cat(
            [
                runtime.forward(
                    tokens[row : row + 1],
                    sequential_states[row],
                    paged_cache=sequential_cache,
                    request_id=request_id,
                )[0]
                for row, request_id in enumerate((10, 11))
            ],
            dim=0,
        )

    torch.testing.assert_close(actual, expected, atol=1.2e-1, rtol=8e-2)
    torch.testing.assert_close(
        actual.argmax(dim=-1), expected.argmax(dim=-1), atol=0, rtol=0
    )
    assert all(
        state.sequence_length == length
        for state, length in zip(batch_states, (4, 3), strict=True)
    )
