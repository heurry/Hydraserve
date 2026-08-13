from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.kernels.flash_prefill import flash_attention_varlen


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_flash_varlen_gqa_smoke() -> None:
    pytest.importorskip("flash_attn")
    torch.manual_seed(4)
    lengths = (5, 3)
    total = sum(lengths)
    query = torch.randn(total, 4, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(total, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    cu_seqlens = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    output = flash_attention_varlen(query, key, value, cu_seqlens, max(lengths))
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
