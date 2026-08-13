from __future__ import annotations

import numpy as np
import pytest

from hydraserve.cache import (
    KVBlockManager,
    LinearState,
    LinearStatePool,
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


def test_int4_round_trip_and_actual_packing() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(7, 19)).astype(np.float32)
    encoded = quantize_int4(source, group_size=16)
    restored = dequantize_int4(encoded)
    assert restored.shape == source.shape
    assert encoded.packed.nbytes < source.nbytes / 4
    assert np.max(np.abs(restored - source)) < 0.25
