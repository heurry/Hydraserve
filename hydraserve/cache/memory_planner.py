"""Memory-safe physical Paged KV sizing."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class PagedKVMemoryPlan:
    requested_blocks: int
    planned_blocks: int
    bytes_per_block: int
    free_bytes: int | None
    reserved_bytes: int

    @property
    def planned_bytes(self) -> int:
        return self.planned_blocks * self.bytes_per_block

    @property
    def was_clamped(self) -> bool:
        return self.planned_blocks < self.requested_blocks


def plan_paged_kv_blocks(
    model,
    requested_blocks: int,
    *,
    block_size: int,
    dtype,
    device,
    state_slots: int = 1,
    state_workspace_slots: int = 1,
    state_memory_fraction: float = 0.5,
    cuda_reserve_bytes: int = 512 * 1024**2,
    allocation_guard_bytes: int = 64 * 1024**2,
    free_bytes: int | None = None,
) -> PagedKVMemoryPlan:
    """Clamp a requested cache to leave guaranteed state and CUDA headroom."""
    import torch

    if min(requested_blocks, block_size) <= 0:
        raise ValueError("requested KV blocks and block size must be positive")
    if min(
        state_slots,
        state_workspace_slots,
        cuda_reserve_bytes,
        allocation_guard_bytes,
    ) < 0:
        raise ValueError("KV memory reserves cannot be negative")
    if not 0 < state_memory_fraction <= 1:
        raise ValueError("state memory fraction must be in (0, 1]")
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_block = (
        model.num_full_attention_layers
        * block_size
        * model.num_kv_heads
        * model.head_dim
        * 2
        * element_size
    )
    if bytes_per_block <= 0:
        raise ValueError("model has no physical KV bytes per block")
    state_bytes = state_slots * model.recurrent_state_bytes
    workspace_bytes = state_workspace_slots * (
        model.recurrent_state_bytes + model.conv_state_bytes
    )
    state_required_bytes = state_bytes + workspace_bytes
    reserved_bytes = max(
        ceil(state_required_bytes / state_memory_fraction),
        state_required_bytes + cuda_reserve_bytes,
    ) + allocation_guard_bytes
    target = torch.device(device)
    if target.type != "cuda":
        return PagedKVMemoryPlan(
            requested_blocks,
            requested_blocks,
            bytes_per_block,
            None,
            0,
        )
    if free_bytes is None:
        free_bytes, _ = torch.cuda.mem_get_info(target)
    free_bytes = int(free_bytes)
    if free_bytes < 0:
        raise ValueError("free CUDA memory cannot be negative")
    available = max(0, free_bytes - reserved_bytes)
    planned_blocks = min(requested_blocks, available // bytes_per_block)
    if planned_blocks <= 0:
        raise MemoryError(
            "insufficient CUDA memory for one KV block plus guaranteed recurrent state"
        )
    return PagedKVMemoryPlan(
        requested_blocks,
        planned_blocks,
        bytes_per_block,
        free_bytes,
        reserved_bytes,
    )
