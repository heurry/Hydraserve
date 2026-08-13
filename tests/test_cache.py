from __future__ import annotations

import numpy as np
import pytest

from hydraserve.cache import (
    KVBlockManager,
    LinearState,
    LinearStatePool,
    GpuLinearStatePool,
    RequestStateSlotManager,
    dequantize_int4,
    quantize_int4,
)


def test_block_manager_allocate_grow_and_free() -> None:
    manager = KVBlockManager(num_blocks=4, block_size=16)
    allocation = manager.allocate(7, 17)
    assert allocation.block_ids == (0, 1)
    assert manager.grow(7, 16).block_ids == (0, 1, 2)
    manager.free(7)
    assert manager.num_free_blocks == 4


def test_block_manager_is_transactional_when_exhausted() -> None:
    manager = KVBlockManager(num_blocks=1)
    with pytest.raises(MemoryError):
        manager.allocate(1, 17)
    assert manager.num_free_blocks == 1


def test_block_manager_reserves_decode_growth_without_exposing_tokens() -> None:
    manager = KVBlockManager(num_blocks=5, block_size=4)
    allocation = manager.allocate(1, 3, reserve_tokens=11)
    assert allocation.num_tokens == 3
    assert allocation.reserved_tokens == 11
    assert allocation.block_ids == (0, 1, 2)
    assert manager.num_free_blocks == 2

    grown = manager.grow(1, 7)
    assert grown.num_tokens == 10
    assert grown.block_ids == allocation.block_ids
    assert manager.num_free_blocks == 2

    manager.truncate(1, 4)
    assert manager.get(1).reserved_tokens == 11
    assert manager.num_free_blocks == 2


def test_block_manager_reservation_failure_is_atomic() -> None:
    manager = KVBlockManager(num_blocks=2, block_size=4)
    with pytest.raises(MemoryError):
        manager.allocate(1, 1, reserve_tokens=9)
    assert manager.capacity().free_blocks == 2


def test_decode_batch_growth_is_atomic() -> None:
    manager = KVBlockManager(num_blocks=2, block_size=4)
    manager.allocate(1, 4)
    manager.allocate(2, 4)
    with pytest.raises(MemoryError, match="decode batch"):
        manager.grow_many((1, 2))
    assert manager.get(1).num_tokens == 4
    assert manager.get(2).num_tokens == 4
    assert manager.num_free_blocks == 0


def test_shared_prefix_blocks_use_reference_counted_ownership() -> None:
    manager = KVBlockManager(num_blocks=4, block_size=2)
    first = manager.allocate(1, 4)
    shared = first.block_ids[0]
    manager.retain_blocks((shared,))
    assert manager.block_refcount(shared) == 2
    second = manager.allocate(2, 3, prefix_block_ids=(shared,))
    assert second.block_ids[0] == shared
    assert second.prefix_blocks == 1
    assert manager.block_refcount(shared) == 3
    manager.free(1)
    assert manager.block_refcount(shared) == 2
    manager.free(2)
    assert manager.block_refcount(shared) == 1
    manager.release_blocks((shared,))
    assert manager.block_refcount(shared) == 0
    assert manager.num_free_blocks == 4


def test_shared_prefix_allocation_rolls_back_without_refcount_change() -> None:
    manager = KVBlockManager(num_blocks=2, block_size=2)
    first = manager.allocate(1, 2)
    shared = first.block_ids[0]
    manager.retain_blocks((shared,))
    with pytest.raises(MemoryError):
        manager.allocate(2, 2, reserve_tokens=5, prefix_block_ids=(shared,))
    assert manager.block_refcount(shared) == 2


def test_linear_state_pool_enforces_fp32(tiny_model) -> None:
    pool = LinearStatePool(1, tiny_model.ssm_state_shape, tiny_model.conv_state_shape)
    pool.allocate(10)
    state = LinearState(
        np.ones(tiny_model.ssm_state_shape, dtype=np.float32),
        np.ones(tiny_model.conv_state_shape, dtype=np.float32),
    )
    pool.set(10, state)
    state.ssm_state.fill(9)
    assert pool.get(10).ssm_state.flat[0] == 1
    with pytest.raises(TypeError, match="float32"):
        LinearState(
            np.ones(tiny_model.ssm_state_shape, dtype=np.float16),
            np.ones(tiny_model.conv_state_shape, dtype=np.float32),
        )


def test_request_state_slots_are_bounded_and_idempotent() -> None:
    slots = RequestStateSlotManager(1)
    assert slots.allocate(1) == slots.allocate(1) == 0
    with pytest.raises(MemoryError, match="exhausted"):
        slots.allocate(2)
    slots.free(1)
    assert slots.allocate(2) == 0
    assert slots.capacity().allocated_slots == 1


def test_gpu_linear_state_pool_uses_contiguous_reusable_views(tiny_model) -> None:
    torch = pytest.importorskip("torch")
    from hydraserve.model.runtime import RuntimeState

    pool = GpuLinearStatePool(2, tiny_model, device="cpu")
    source = RuntimeState(sequence_length=7)
    for layer_index in tiny_model.linear_layer_indices:
        source.recurrent[layer_index] = torch.ones(
            (1, *tiny_model.ssm_state_shape[1:]), dtype=torch.float32
        )
        source.convolution[layer_index] = torch.full(
            (1, *tiny_model.conv_state_shape[1:]), 2.0, dtype=torch.float32
        )
    state = pool.install(10, source)
    assert state.sequence_length == 7
    first_layer = tiny_model.linear_layer_indices[0]
    state.recurrent[first_layer].fill_(3)
    assert pool.ssm_storage[0, 0].flatten()[0] == 3
    assert pool.ssm_storage.is_contiguous()
    assert pool.capacity == pool.requested_capacity == 2
    assert pool.bytes_per_slot == (
        (np.prod(tiny_model.ssm_state_shape) + np.prod(tiny_model.conv_state_shape))
        * 4
    )
    pool.free(10)
    reused = pool.allocate(11)
    assert pool.slots.get(11) == 0
    assert not bool(reused.recurrent[first_layer].any())


def test_int4_round_trip_and_actual_packing() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(7, 19)).astype(np.float32)
    encoded = quantize_int4(source, group_size=16)
    restored = dequantize_int4(encoded)
    assert restored.shape == source.shape
    assert encoded.packed.nbytes < source.nbytes / 4
    assert np.max(np.abs(restored - source)) < 0.25
