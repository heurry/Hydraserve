from __future__ import annotations

from typing import Any, Iterable

from hydraserve.cache.block_manager import KVBlockManager
from hydraserve.config import ModelConfig


class PagedKVCache:
    """Physical KV tensor storage shared by all requests on one decode worker."""

    def __init__(
        self,
        model: ModelConfig,
        block_manager: KVBlockManager,
        *,
        device: str | Any,
        dtype: Any,
    ) -> None:
        import torch

        if block_manager.block_size <= 0:
            raise ValueError("invalid block manager")
        self.model = model
        self.block_manager = block_manager
        self.device = torch.device(device)
        self.dtype = dtype
        self.layer_to_slot = {
            layer_index: slot for slot, layer_index in enumerate(model.full_attention_layer_indices)
        }
        shape = (
            model.num_full_attention_layers,
            block_manager.num_blocks,
            block_manager.block_size,
            model.num_kv_heads,
            model.head_dim,
        )
        self.key = torch.empty(shape, device=self.device, dtype=dtype)
        self.value = torch.empty_like(self.key)

    def allocate(self, request_id: int, num_tokens: int):
        return self.block_manager.allocate(request_id, num_tokens)

    def reserve_append(self, request_id: int, additional_tokens: int = 1):
        return self.block_manager.grow(request_id, additional_tokens)

    def free(self, request_id: int) -> None:
        self.block_manager.free(request_id)

    def write(self, request_id: int, layer_index: int, positions, key, value) -> None:
        import torch

        slot = self._layer_slot(layer_index)
        allocation = self.block_manager.get(request_id)
        positions = torch.as_tensor(positions, device=self.device, dtype=torch.int32).contiguous()
        key = key.to(device=self.device, dtype=self.dtype).contiguous()
        value = value.to(device=self.device, dtype=self.dtype).contiguous()
        if key.shape != value.shape or key.shape != (
            positions.numel(),
            self.model.num_kv_heads,
            self.model.head_dim,
        ):
            raise ValueError("invalid projected KV shape")
        if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= allocation.num_tokens):
            raise IndexError("KV write position is outside the request allocation")
        block_ids = torch.tensor(allocation.block_ids, device=self.device, dtype=torch.int32)
        if self.device.type == "cuda":
            from hydraserve.kernels.kv_cache import write_paged_kv

            write_paged_kv(key, value, positions, block_ids, self.key[slot], self.value[slot])
            return
        logical = torch.div(positions, self.block_manager.block_size, rounding_mode="floor")
        offsets = positions.remainder(self.block_manager.block_size)
        physical = block_ids[logical.long()]
        self.key[slot, physical.long(), offsets.long()] = key
        self.value[slot, physical.long(), offsets.long()] = value

    def layer_cache(self, layer_index: int):
        slot = self._layer_slot(layer_index)
        return self.key[slot], self.value[slot]

    def batch_metadata(self, request_ids: Iterable[int]):
        import torch

        allocations = [self.block_manager.get(request_id) for request_id in request_ids]
        if not allocations:
            raise ValueError("cannot build empty KV batch metadata")
        width = max(len(allocation.block_ids) for allocation in allocations)
        table = torch.full(
            (len(allocations), width), -1, device=self.device, dtype=torch.int32
        )
        lengths = torch.empty(len(allocations), device=self.device, dtype=torch.int32)
        for row, allocation in enumerate(allocations):
            table[row, : len(allocation.block_ids)] = torch.tensor(
                allocation.block_ids, device=self.device, dtype=torch.int32
            )
            lengths[row] = allocation.num_tokens
        return table, lengths

    def _layer_slot(self, layer_index: int) -> int:
        try:
            return self.layer_to_slot[layer_index]
        except KeyError as exc:
            raise ValueError(f"layer {layer_index} is not a full-attention layer") from exc
