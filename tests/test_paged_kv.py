from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache
from hydraserve.kernels.paged_attention import paged_prefill_attention
from hydraserve.kernels.reference import causal_gqa_attention


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paged_kv_write_and_batch_metadata(tiny_model, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    manager = KVBlockManager(8, block_size=4)
    cache = PagedKVCache(tiny_model, manager, device=device, dtype=torch.float32)
    cache.allocate(1, 6)
    cache.allocate(2, 3)
    positions = torch.tensor([0, 1, 4, 5], device=device)
    key = torch.arange(4 * 2 * 8, device=device, dtype=torch.float32).reshape(4, 2, 8)
    value = -key
    layer = tiny_model.full_attention_layer_indices[0]
    cache.write(1, layer, positions, key, value)
    key_pages, value_pages = cache.layer_cache(layer)
    allocation = manager.get(1)
    torch.testing.assert_close(key_pages[allocation.block_ids[0], 0], key[0])
    torch.testing.assert_close(key_pages[allocation.block_ids[1], 1], key[3])
    torch.testing.assert_close(value_pages[allocation.block_ids[1], 1], value[3])
    table, lengths = cache.batch_metadata((2, 1))
    assert lengths.tolist() == [3, 6]
    assert table[1, :2].tolist() == list(allocation.block_ids)
    gathered_key, gathered_value = cache.read(1, layer)
    torch.testing.assert_close(gathered_key[positions], key)
    torch.testing.assert_close(gathered_value[positions], value)
    prefix_key, prefix_value = cache.read(1, layer, num_tokens=5)
    assert prefix_key.shape[0] == prefix_value.shape[0] == 5
    _, prefix_lengths = cache.batch_metadata((2, 1), logical_lengths=(2, 5))
    assert prefix_lengths.tolist() == [2, 5]
    with pytest.raises(ValueError, match="exceeds"):
        cache.read(1, layer, num_tokens=7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paged_prefill_attention_preserves_chunk_history(tiny_model, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    manager = KVBlockManager(8, block_size=4)
    cache = PagedKVCache(tiny_model, manager, device=device, dtype=torch.float32)
    cache.allocate(9, 5)
    generator = torch.Generator(device=device).manual_seed(23)
    key = torch.randn(
        5,
        tiny_model.num_kv_heads,
        tiny_model.head_dim,
        generator=generator,
        device=device,
    )
    value = torch.randn(key.shape, generator=generator, device=device)
    query = torch.randn(
        1,
        3,
        tiny_model.num_attention_heads,
        tiny_model.head_dim,
        generator=generator,
        device=device,
    )
    layer = tiny_model.full_attention_layer_indices[0]
    cache.write(9, layer, torch.arange(5, device=device), key, value)
    table, _ = cache.batch_metadata((9,))
    key_pages, value_pages = cache.layer_cache(layer)
    actual = paged_prefill_attention(
        query, key_pages, value_pages, table, query_start=2
    )
    expected = causal_gqa_attention(
        query, key.unsqueeze(0), value.unsqueeze(0), query_start=2
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
