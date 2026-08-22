from __future__ import annotations

import numpy as np
import pytest
from random import Random

from hydraserve.cache import (
    KVBlockManager,
    LinearState,
    LinearStatePool,
    GpuLinearStatePool,
    RequestStateSlotManager,
    dequantize_int4,
    quantize_int4,
)
from hydraserve.cache.state_pool import cuda_state_memory_budget


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


def test_block_manager_headroom_is_not_admitted_but_remains_physically_free() -> None:
    manager = KVBlockManager(4, block_size=4, headroom_blocks=1)
    manager.allocate(1, 8)
    capacity = manager.capacity()
    assert capacity.total_blocks == 3
    assert capacity.free_blocks == 1
    assert capacity.physical_free_blocks == 2
    assert capacity.headroom_blocks == 1
    with pytest.raises(MemoryError, match="headroom"):
        manager.allocate(2, 8)
    failed = manager.audit()
    assert failed.allocation_failures == 1
    assert failed.high_watermark_blocks == 2
    manager.free(1)
    assert manager.audit().physical_free_blocks == 4


def test_block_manager_audit_tracks_fragmentation_sharing_and_no_leaks() -> None:
    manager = KVBlockManager(8, block_size=4)
    first = manager.allocate(1, 5, reserve_tokens=7)
    manager.retain_blocks((first.block_ids[0],))
    manager.allocate(2, 4, prefix_block_ids=(first.block_ids[0],))
    stats = manager.audit()
    assert stats.active_allocations == 2
    assert stats.shared_blocks == 1
    assert stats.total_references == stats.allocation_block_references + 1
    assert stats.internal_fragmentation_tokens == 1
    manager.free(1)
    manager.free(2)
    manager.release_blocks((first.block_ids[0],))
    final = manager.audit()
    assert final.allocated_blocks == 0
    assert final.total_references == 0


def test_block_manager_random_transaction_soak_preserves_invariants() -> None:
    rng = Random(17)
    manager = KVBlockManager(64, block_size=4, headroom_blocks=4)
    active: set[int] = set()
    for _ in range(2_000):
        request_id = rng.randrange(32)
        operation = rng.randrange(4)
        try:
            if request_id not in active:
                tokens = rng.randint(1, 12)
                reserve = tokens + rng.randint(0, 20)
                manager.allocate(request_id, tokens, reserve_tokens=reserve)
                active.add(request_id)
            elif operation == 0:
                manager.free(request_id)
                active.remove(request_id)
            elif operation == 1:
                current = manager.get(request_id)
                manager.truncate(request_id, rng.randint(0, current.num_tokens))
            elif operation == 2:
                current = manager.get(request_id)
                manager.reserve(
                    request_id,
                    current.num_tokens + rng.randint(0, 12),
                )
            else:
                manager.grow(request_id, rng.randint(0, 3))
        except MemoryError:
            pass
        manager.audit()
    for request_id in tuple(active):
        manager.free(request_id)
    final = manager.audit()
    assert final.physical_free_blocks == 64
    assert final.total_references == 0


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


def test_cuda_state_budget_enforces_fraction_and_hard_reserve() -> None:
    gib = 1024**3
    assert cuda_state_memory_budget(20 * gib, 0.5, 512 * 1024**2) == 10 * gib
    assert cuda_state_memory_budget(900 * 1024**2, 0.5, 512 * 1024**2) == 388 * 1024**2
    assert cuda_state_memory_budget(256 * 1024**2, 0.5, 512 * 1024**2) == 0


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
    assert pool.stats()["state_workspace_slots"] == 2
    assert pool.stats()["state_storage_bytes"] == 2 * pool.bytes_per_slot
    pool.free(10)
    reused = pool.allocate(11)
    assert pool.slots.get(11) == 0
    assert not bool(reused.recurrent[first_layer].any())


def test_gpu_state_batch_is_invisible_until_atomic_commit(tiny_model) -> None:
    torch = pytest.importorskip("torch")
    from hydraserve.model.runtime import RuntimeState

    pool = GpuLinearStatePool(
        3, tiny_model, device="cpu", workspace_capacity=2
    )
    states = {}
    for request_id, fill in ((10, 1.0), (20, 2.0)):
        source = RuntimeState(sequence_length=4)
        for layer_index in tiny_model.linear_layer_indices:
            source.recurrent[layer_index] = torch.full(
                (1, *tiny_model.ssm_state_shape[1:]), fill
            )
            source.convolution[layer_index] = torch.full(
                (1, *tiny_model.conv_state_shape[1:]), fill + 10
            )
        states[request_id] = pool.install(request_id, source)

    recurrent_workspace = pool.ssm_workspace.data_ptr()
    conv_workspace = pool.conv_workspace.data_ptr()
    first_layer = tiny_model.linear_layer_indices[0]
    with pool.batch((20, 10), (states[20], states[10])) as batch:
        slot_ids_workspace = batch.slot_ids.data_ptr()
        recurrent, convolution, next_convolution = batch.layer(first_layer)
        assert recurrent[:, 0, 0, 0].tolist() == [2.0, 1.0]
        assert convolution[:, 0, 0].tolist() == [12.0, 11.0]
        recurrent.add_(20)
        next_convolution.copy_(convolution + 30)
        batch.set_layer_result(first_layer, recurrent, next_convolution)
        assert states[20].recurrent[first_layer].flatten()[0] == 2
        assert states[10].convolution[first_layer].flatten()[0] == 11
        for layer_index in tiny_model.linear_layer_indices[1:]:
            current_recurrent, current_conv, next_conv = batch.layer(layer_index)
            next_conv.copy_(current_conv)
            batch.set_layer_result(layer_index, current_recurrent, next_conv)
        batch.commit()

    assert states[20].recurrent[first_layer].flatten()[0] == 22
    assert states[10].recurrent[first_layer].flatten()[0] == 21
    assert states[20].convolution[first_layer].flatten()[0] == 42
    assert states[10].convolution[first_layer].flatten()[0] == 41
    with pool.batch((10,), (states[10],)) as batch:
        assert pool.ssm_workspace.data_ptr() == recurrent_workspace
        assert pool.conv_workspace.data_ptr() == conv_workspace
        assert batch.slot_ids.data_ptr() == slot_ids_workspace
    with pytest.raises(MemoryError, match="workspace"):
        with pool.batch((10, 20, 30), (states[10], states[20], states[10])):
            pass


def test_int4_round_trip_and_actual_packing() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(7, 19)).astype(np.float32)
    encoded = quantize_int4(source, group_size=16)
    restored = dequantize_int4(encoded)
    assert restored.shape == source.shape
    assert encoded.packed.nbytes < source.nbytes / 4
    assert np.max(np.abs(restored - source)) < 0.25
