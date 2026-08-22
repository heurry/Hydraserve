from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from hydraserve.kernels.flash_prefill import flash_attention_varlen, paged_flash_prefill
from hydraserve.kernels.reference import causal_gqa_attention


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


def test_paged_flash_prefill_passes_physical_cache_without_copy(monkeypatch) -> None:
    calls = []
    module = ModuleType("flash_attn")

    def fake_flash(query, key_cache, value_cache, **kwargs):
        calls.append((query, key_cache, value_cache, kwargs))
        return torch.empty_like(query)

    module.flash_attn_with_kvcache = fake_flash
    monkeypatch.setitem(sys.modules, "flash_attn", module)
    query = torch.randn(1, 3, 4, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(8, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.tensor([[6, 1]], device="cuda", dtype=torch.int32)

    output = paged_flash_prefill(query, key, value, table, 19)

    assert output.shape == query.shape
    assert len(calls) == 1
    _, actual_key, actual_value, options = calls[0]
    assert actual_key.data_ptr() == key.data_ptr()
    assert actual_value.data_ptr() == value.data_ptr()
    assert actual_key.shape == (8, 16, 2, 32)
    assert options["block_table"].data_ptr() == table.data_ptr()
    assert options["cache_seqlens"].tolist() == [19]
    assert options["causal"] is True


def test_paged_flash_prefill_matches_gqa_reference() -> None:
    pytest.importorskip("flash_attn")
    torch.manual_seed(29)
    block_size = 256
    cache_length = block_size + 3
    query_length = 3
    query = torch.randn(
        1, query_length, 4, 64, device="cuda", dtype=torch.bfloat16
    )
    key_pages = torch.randn(
        2, block_size, 2, 64, device="cuda", dtype=torch.bfloat16
    )
    value_pages = torch.randn_like(key_pages)
    block_table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)

    actual = paged_flash_prefill(
        query,
        key_pages,
        value_pages,
        block_table,
        cache_length,
    )
    logical_key = key_pages[block_table[0]].reshape(1, -1, 2, 64)[:, :cache_length]
    logical_value = value_pages[block_table[0]].reshape(1, -1, 2, 64)[:, :cache_length]
    expected = causal_gqa_attention(
        query,
        logical_key,
        logical_value,
        query_start=cache_length - query_length,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
