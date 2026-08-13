from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import PagedKVMemoryPlan, plan_paged_kv_blocks


def test_cpu_kv_memory_plan_preserves_requested_capacity(tiny_model) -> None:
    plan = plan_paged_kv_blocks(
        tiny_model,
        17,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    assert plan.planned_blocks == plan.requested_blocks == 17
    assert plan.bytes_per_block == 4 * 2 * 8 * 2 * 4
    assert plan.free_bytes is None
    assert not plan.was_clamped


def test_cuda_kv_memory_plan_clamps_after_state_and_workspace_reserve(
    tiny_model,
) -> None:
    state_bytes = tiny_model.recurrent_state_bytes
    convolution_bytes = torch.empty(
        tiny_model.conv_state_shape, dtype=torch.float32
    ).numel() * 4
    bytes_per_block = 4 * 2 * 8 * 2 * 4
    hard_reserve = 1024
    state_required = state_bytes * 2 + convolution_bytes
    guaranteed = max(state_required * 2, state_required + hard_reserve)
    plan = plan_paged_kv_blocks(
        tiny_model,
        20,
        block_size=4,
        dtype=torch.float32,
        device="cuda",
        cuda_reserve_bytes=hard_reserve,
        allocation_guard_bytes=0,
        free_bytes=guaranteed + 5 * bytes_per_block,
    )
    assert plan == PagedKVMemoryPlan(
        requested_blocks=20,
        planned_blocks=5,
        bytes_per_block=bytes_per_block,
        free_bytes=guaranteed + 5 * bytes_per_block,
        reserved_bytes=guaranteed,
    )
    assert plan.was_clamped
    assert plan.planned_bytes == 5 * bytes_per_block


def test_cuda_kv_memory_plan_rejects_when_no_block_fits(tiny_model) -> None:
    with pytest.raises(MemoryError, match="one KV block"):
        plan_paged_kv_blocks(
            tiny_model,
            1,
            block_size=4,
            dtype=torch.float32,
            device="cuda",
            free_bytes=0,
        )
