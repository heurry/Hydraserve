from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import (
    CostAwarePrefixPolicy,
    KVBlockManager,
    PagedKVCache,
    PrefixCache,
    PagedKVMemoryPlan,
)
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
    bucketed, bucketed_lengths = cache.batch_metadata(
        (2, 1), bucket_width=True
    )
    assert bucketed.shape == (2, 2)
    torch.testing.assert_close(bucketed_lengths, lengths)
    gathered_key, gathered_value = cache.read(1, layer)
    torch.testing.assert_close(gathered_key[positions], key)
    torch.testing.assert_close(gathered_value[positions], value)
    prefix_key, prefix_value = cache.read(1, layer, num_tokens=5)
    assert prefix_key.shape[0] == prefix_value.shape[0] == 5
    _, prefix_lengths = cache.batch_metadata((2, 1), logical_lengths=(2, 5))
    assert prefix_lengths.tolist() == [2, 5]
    with pytest.raises(ValueError, match="exceeds"):
        cache.read(1, layer, num_tokens=7)


def test_paged_kv_reports_memory_clamping(tiny_model) -> None:
    manager = KVBlockManager(5, block_size=4)
    plan = PagedKVMemoryPlan(20, 5, 512, 10_000, 7_000)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        memory_plan=plan,
    )
    stats = cache.stats()
    assert stats["requested_physical_blocks"] == 20
    assert stats["memory_planned_blocks"] == 5
    assert stats["memory_clamped"] == 1
    assert stats["memory_reserved_bytes"] == 7_000
    assert stats["physical_cache_bytes"] == 5 * 512


def test_batch_metadata_buckets_graph_width_to_power_of_two(tiny_model) -> None:
    manager = KVBlockManager(16, block_size=4)
    cache = PagedKVCache(tiny_model, manager, device="cpu", dtype=torch.float32)
    cache.allocate(1, 9)
    cache.allocate(2, 5)

    exact, lengths = cache.batch_metadata((1, 2))
    bucketed, bucketed_lengths = cache.batch_metadata(
        (1, 2), bucket_width=True
    )

    assert exact.shape == (2, 3)
    assert bucketed.shape == (2, 4)
    assert bucketed[:, :3].tolist() == exact.tolist()
    assert bucketed[:, 3].tolist() == [-1, -1]
    torch.testing.assert_close(bucketed_lengths, lengths)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_metadata_uses_constant_contiguous_device_builds(
    tiny_model, device: str, monkeypatch
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    manager = KVBlockManager(256, block_size=4)
    cache = PagedKVCache(tiny_model, manager, device=device, dtype=torch.float32)
    expected_lengths = []
    expected_rows = []
    for request_id in range(1, 65):
        length = 1 + request_id % 13
        expected_lengths.append(length)
        allocation = cache.allocate(request_id, length)
        expected_rows.append(allocation.block_ids)

    original_tensor = torch.tensor
    device_builds = []

    def counted_tensor(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")) == cache.device:
            device_builds.append(args[0])
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", counted_tensor)
    table, lengths = cache.batch_metadata(range(1, 65))

    assert len(device_builds) == 2
    assert table.is_contiguous()
    assert lengths.is_contiguous()
    assert lengths.tolist() == expected_lengths
    for row, block_ids in enumerate(expected_rows):
        assert table[row, : len(block_ids)].tolist() == list(block_ids)
        assert table[row, len(block_ids) :].tolist() == [-1] * (
            table.shape[1] - len(block_ids)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paged_kv_batched_decode_scatter(tiny_model, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    manager = KVBlockManager(8, block_size=4)
    cache = PagedKVCache(tiny_model, manager, device=device, dtype=torch.float32)
    cache.allocate(1, 5)
    cache.allocate(2, 3)
    table, _ = cache.batch_metadata((1, 2))
    positions = torch.tensor([4, 2], device=device, dtype=torch.int32)
    key = torch.arange(
        2 * tiny_model.num_kv_heads * tiny_model.head_dim,
        device=device,
        dtype=torch.float32,
    ).reshape(2, tiny_model.num_kv_heads, tiny_model.head_dim)
    value = -key
    layer = tiny_model.full_attention_layer_indices[0]
    cache.write_decode_batch(
        (1, 2),
        layer,
        positions,
        key,
        value,
        table,
        logical_positions=(4, 2),
    )
    first_key, first_value = cache.read(1, layer)
    second_key, second_value = cache.read(2, layer)
    torch.testing.assert_close(first_key[4], key[0])
    torch.testing.assert_close(first_value[4], value[0])
    torch.testing.assert_close(second_key[2], key[1])
    torch.testing.assert_close(second_value[2], value[1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paged_kv_batched_decode_scatter_int8(tiny_model, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    manager = KVBlockManager(8, block_size=4)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device=device,
        dtype=torch.float32,
        kv_quant="int8",
    )
    cache.allocate(1, 5)
    cache.allocate(2, 3)
    table, _ = cache.batch_metadata((1, 2))
    positions = torch.tensor([4, 2], device=device, dtype=torch.int32)
    key = torch.linspace(
        -3.0,
        5.0,
        2 * tiny_model.num_kv_heads * tiny_model.head_dim,
        device=device,
        dtype=torch.float32,
    ).reshape(2, tiny_model.num_kv_heads, tiny_model.head_dim)
    value = -0.75 * key
    layer = tiny_model.full_attention_layer_indices[0]

    cache.write_decode_batch(
        (1, 2),
        layer,
        positions,
        key,
        value,
        table,
        logical_positions=(4, 2),
    )

    first_key, first_value = cache.read(1, layer)
    second_key, second_value = cache.read(2, layer)
    torch.testing.assert_close(first_key[4], key[0], atol=4e-2, rtol=0)
    torch.testing.assert_close(first_value[4], value[0], atol=4e-2, rtol=0)
    torch.testing.assert_close(second_key[2], key[1], atol=4e-2, rtol=0)
    torch.testing.assert_close(second_value[2], value[1], atol=4e-2, rtol=0)
    raw = cache.raw_layer_cache(layer)
    assert len(raw) == 4
    assert raw[0].dtype == raw[1].dtype == torch.int8
    assert raw[2].dtype == raw[3].dtype == torch.float32


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


def test_paged_kv_prefix_lifecycle_shares_and_reclaims_physical_pages(tiny_model) -> None:
    manager = KVBlockManager(8, block_size=2)
    prefix = PrefixCache(
        block_size=2,
        max_blocks=2,
        policy=CostAwarePrefixPolicy(minimum_frequency=1),
    )
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    tokens = (1, 2, 3, 4)
    first = cache.allocate(1, 4, reserve_tokens=5, token_ids=tokens)
    published = cache.publish_prefix(1, tokens)
    assert published.inserted_block_ids == first.block_ids[:2]
    shared = first.block_ids[:2]
    cache.free(1)
    assert all(manager.block_refcount(block) == 1 for block in shared)

    second = cache.allocate(2, 4, reserve_tokens=5, token_ids=tokens)
    assert second.block_ids[:2] == shared
    assert second.prefix_blocks == 2
    assert cache.matched_prefix_tokens(2) == 4
    assert all(manager.block_refcount(block) == 2 for block in shared)
    cache.free(2)
    assert all(manager.block_refcount(block) == 1 for block in shared)

    evicted = prefix.evict(2)
    manager.release_blocks(evicted)
    assert manager.num_free_blocks == manager.num_blocks


def test_paged_kv_does_not_overwrite_shared_prefix_pages(tiny_model) -> None:
    manager = KVBlockManager(6, block_size=2)
    prefix = PrefixCache(block_size=2, max_blocks=1)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    tokens = (1, 2, 3)
    first = cache.allocate(1, 3, token_ids=tokens)
    layer = tiny_model.full_attention_layer_indices[0]
    key = torch.ones(3, tiny_model.num_kv_heads, tiny_model.head_dim)
    cache.write(1, layer, torch.arange(3), key, key)
    cache.publish_prefix(1, tokens)
    cache.free(1)

    second = cache.allocate(2, 3, token_ids=tokens)
    replacement = torch.full_like(key, 9)
    cache.write(2, layer, torch.arange(3), replacement, replacement)
    gathered, _ = cache.read(2, layer)
    torch.testing.assert_close(gathered[:2], key[:2])
    torch.testing.assert_close(gathered[2:], replacement[2:])
    cache.free(2)


def test_active_admission_evicts_unreferenced_prefix_pages_under_pressure(tiny_model) -> None:
    manager = KVBlockManager(3, block_size=2)
    prefix = PrefixCache(block_size=2, max_blocks=2)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    tokens = (1, 2, 3, 4)
    cache.allocate(1, 4, token_ids=tokens)
    cache.publish_prefix(1, tokens)
    cache.free(1)
    assert manager.num_free_blocks == 1

    allocation = cache.allocate(2, 4, token_ids=(8, 9, 10, 11))
    assert len(allocation.block_ids) == 2
    assert prefix.stats().evictions == 1
    assert dict(prefix.stats().evicted_by_reason) == {"active_pressure": 1}
    assert manager.num_free_blocks == 0
    cache.free(2)


def test_paged_cache_audit_reconciles_prefix_and_request_owners(tiny_model) -> None:
    manager = KVBlockManager(6, block_size=2, headroom_blocks=1)
    prefix = PrefixCache(block_size=2, max_blocks=2)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    tokens = (1, 2, 3, 4)
    cache.allocate(1, 4, reserve_tokens=5, token_ids=tokens)
    cache.publish_prefix(1, tokens)
    live = cache.audit()
    assert live["active_allocations"] == 1
    assert live["prefix_cached_blocks"] == 2
    assert live["headroom_blocks"] == 1
    cache.free(1)
    retained = cache.audit()
    assert retained["active_allocations"] == 0
    assert retained["allocated_blocks"] == 2
    evicted = prefix.evict(2)
    manager.release_blocks(evicted)
    final = cache.audit()
    assert final["allocated_blocks"] == 0
    assert final["physical_free_blocks"] == 6


def test_active_pressure_reclaims_prefix_without_consuming_headroom(tiny_model) -> None:
    manager = KVBlockManager(4, block_size=2, headroom_blocks=1)
    prefix = PrefixCache(block_size=2, max_blocks=2)
    cache = PagedKVCache(
        tiny_model,
        manager,
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    cached_tokens = (1, 2, 3, 4)
    cache.allocate(1, 4, token_ids=cached_tokens)
    cache.publish_prefix(1, cached_tokens)
    cache.free(1)
    assert manager.num_allocatable_blocks == 1

    allocation = cache.allocate(2, 4, token_ids=(8, 9, 10, 11))
    assert len(allocation.block_ids) == 2
    stats = cache.stats()
    assert stats["physical_free_blocks"] == 1
    assert stats["allocatable_free_blocks"] == 0
    assert stats["prefix_evicted_active_pressure"] == 1
    cache.free(2)
    cache.audit()
