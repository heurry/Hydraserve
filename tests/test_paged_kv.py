from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache


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
