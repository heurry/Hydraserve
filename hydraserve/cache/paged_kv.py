from __future__ import annotations

from typing import Any, Iterable
from threading import RLock

from hydraserve.cache.block_manager import KVBlockManager
from hydraserve.cache.prefix_cache import (
    DEFAULT_NAMESPACE,
    CacheNamespace,
    PrefixCache,
    PrefixMatch,
)
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
        prefix_cache: PrefixCache | None = None,
        cache_namespace: CacheNamespace = DEFAULT_NAMESPACE,
    ) -> None:
        import torch

        if block_manager.block_size <= 0:
            raise ValueError("invalid block manager")
        self.model = model
        self.block_manager = block_manager
        self.device = torch.device(device)
        self.dtype = dtype
        if prefix_cache is not None and prefix_cache.block_size != block_manager.block_size:
            raise ValueError("prefix cache and KV allocator block sizes must match")
        self.prefix_cache = prefix_cache
        self.cache_namespace = cache_namespace
        self._prefix_matches: dict[int, tuple[tuple[int, ...], PrefixMatch]] = {}
        self._prefix_lock = RLock()
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

    def allocate(
        self,
        request_id: int,
        num_tokens: int,
        *,
        reserve_tokens: int | None = None,
        token_ids: Iterable[int] | None = None,
    ):
        tokens = None if token_ids is None else tuple(int(token) for token in token_ids)
        if tokens is not None and len(tokens) != num_tokens:
            raise ValueError("token_ids must match the logical allocation length")
        with self._prefix_lock:
            match = PrefixMatch(0, ())
            if self.prefix_cache is not None and tokens is not None:
                match = self.prefix_cache.match(
                    tokens, namespace=self.cache_namespace, acquire=True
                )
            try:
                required = self.block_manager.blocks_required(
                    num_tokens if reserve_tokens is None else reserve_tokens
                )
                new_required = max(0, required - len(match.block_ids))
                shortage = max(
                    0, new_required - self.block_manager.num_allocatable_blocks
                )
                if shortage and self.prefix_cache is not None:
                    evicted = self.prefix_cache.evict(
                        shortage, reason="active_pressure"
                    )
                    if evicted:
                        self.block_manager.release_blocks(evicted)
                allocation = self.block_manager.allocate(
                    request_id,
                    num_tokens,
                    reserve_tokens=reserve_tokens,
                    prefix_block_ids=match.block_ids,
                )
            except Exception:
                if match.matched_tokens and tokens is not None:
                    self.prefix_cache.release(
                        tokens,
                        match.matched_tokens,
                        namespace=self.cache_namespace,
                    )
                raise
            if match.matched_tokens and tokens is not None:
                self._prefix_matches[request_id] = (tokens, match)
            return allocation

    def reserve_append(self, request_id: int, additional_tokens: int = 1):
        return self.block_manager.grow(request_id, additional_tokens)

    def free(self, request_id: int) -> None:
        with self._prefix_lock:
            self.block_manager.free(request_id)
            owner = self._prefix_matches.pop(request_id, None)
            if owner is not None and self.prefix_cache is not None:
                tokens, match = owner
                self.prefix_cache.release(
                    tokens,
                    match.matched_tokens,
                    namespace=self.cache_namespace,
                )

    def publish_prefix(
        self,
        request_id: int,
        token_ids: Iterable[int],
        *,
        recompute_cost_ms: float | None = None,
    ) -> PrefixMatch:
        """Publish complete prompt pages and transfer ownership to PrefixCache."""
        if self.prefix_cache is None:
            return PrefixMatch(0, (), admitted=False, reason="prefix cache is disabled")
        tokens = tuple(int(token) for token in token_ids)
        with self._prefix_lock:
            allocation = self.block_manager.get(request_id)
            full_blocks = min(
                len(tokens) // self.block_manager.block_size,
                len(allocation.block_ids),
            )
            if full_blocks == 0:
                return PrefixMatch(0, ())
            blocks = allocation.block_ids[:full_blocks]
            self.block_manager.retain_blocks(blocks)
            try:
                result = self.prefix_cache.insert(
                    tokens,
                    blocks,
                    namespace=self.cache_namespace,
                    recompute_cost_ms=recompute_cost_ms,
                    bytes_per_block=self._bytes_per_block(),
                )
            except Exception:
                self.block_manager.release_blocks(blocks)
                raise
            inserted = set(result.inserted_block_ids)
            not_inserted = tuple(block_id for block_id in blocks if block_id not in inserted)
            if not_inserted:
                self.block_manager.release_blocks(not_inserted)
            if result.evicted_block_ids:
                self.block_manager.release_blocks(result.evicted_block_ids)
            return result

    def probe_prefix(self, token_ids: Iterable[int]) -> PrefixMatch:
        if self.prefix_cache is None:
            return PrefixMatch(0, ())
        return self.prefix_cache.match(
            token_ids, namespace=self.cache_namespace, acquire=False
        )

    def matched_prefix_tokens(self, request_id: int) -> int:
        with self._prefix_lock:
            owner = self._prefix_matches.get(request_id)
            return 0 if owner is None else owner[1].matched_tokens

    def stats(self) -> dict[str, int | float]:
        block = self.block_manager.capacity()
        prefix = None if self.prefix_cache is None else self.prefix_cache.stats()
        values: dict[str, int | float] = {
            "physical_total_blocks": block.physical_total_blocks,
            "physical_free_blocks": block.physical_free_blocks,
            "usable_total_blocks": block.total_blocks,
            "allocatable_free_blocks": block.free_blocks,
            "headroom_blocks": block.headroom_blocks,
            "allocated_blocks": block.allocated_blocks,
            "active_allocations": block.active_allocations,
            "allocation_block_references": block.allocation_block_references,
            "total_references": block.total_references,
            "shared_blocks": block.shared_blocks,
            "logical_tokens": block.logical_tokens,
            "reserved_tokens": block.reserved_tokens,
            "internal_fragmentation_tokens": block.internal_fragmentation_tokens,
            "high_watermark_blocks": block.high_watermark_blocks,
            "allocation_failures": block.allocation_failures,
            "prefix_cached_blocks": 0,
            "prefix_referenced_blocks": 0,
            "prefix_evictable_blocks": 0,
            "prefix_cached_bytes": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
            "prefix_hit_tokens": 0,
            "prefix_admissions": 0,
            "prefix_rejected_admissions": 0,
            "prefix_evictions": 0,
            "prefix_rejected_frequency": 0,
            "prefix_rejected_capacity": 0,
            "prefix_rejected_size": 0,
            "prefix_rejected_length": 0,
            "prefix_evicted_active_pressure": 0,
            "prefix_evicted_cache_capacity": 0,
            "prefix_evicted_manual": 0,
        }
        if prefix is not None:
            rejected = dict(prefix.rejected_by_reason)
            evicted = dict(prefix.evicted_by_reason)
            values.update(
                prefix_cached_blocks=prefix.cached_blocks,
                prefix_referenced_blocks=prefix.referenced_blocks,
                prefix_evictable_blocks=prefix.evictable_blocks,
                prefix_cached_bytes=prefix.cached_bytes,
                prefix_hits=prefix.hits,
                prefix_misses=prefix.misses,
                prefix_hit_tokens=prefix.hit_tokens,
                prefix_admissions=prefix.admissions,
                prefix_rejected_admissions=prefix.rejected_admissions,
                prefix_evictions=prefix.evictions,
                prefix_rejected_frequency=rejected.get(
                    "prefix has not passed the frequency doorkeeper", 0
                ),
                prefix_rejected_capacity=rejected.get(
                    "cache has no evictable capacity", 0
                ),
                prefix_rejected_size=rejected.get(
                    "prefix would consume too much of the cache", 0
                ),
                prefix_rejected_length=rejected.get(
                    "prefix is below the minimum reusable length", 0
                ),
                prefix_evicted_active_pressure=evicted.get("active_pressure", 0),
                prefix_evicted_cache_capacity=evicted.get("cache_capacity", 0),
                prefix_evicted_manual=evicted.get("manual", 0),
            )
        return values

    def audit(self) -> dict[str, int | float]:
        block = self.block_manager.audit()
        prefix = None if self.prefix_cache is None else self.prefix_cache.stats()
        prefix_owners = 0 if prefix is None else prefix.cached_blocks
        expected_references = block.allocation_block_references + prefix_owners
        if block.total_references != expected_references:
            raise RuntimeError(
                "KV reference leak: allocator references do not match request and prefix owners"
            )
        with self._prefix_lock:
            for request_id, (_, match) in self._prefix_matches.items():
                allocation = self.block_manager.get(request_id)
                if allocation.prefix_blocks != len(match.block_ids):
                    raise RuntimeError("KV prefix ownership metadata is inconsistent")
        return self.stats()

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
        matched_tokens = self.matched_prefix_tokens(request_id)
        if matched_tokens:
            writable = positions >= matched_tokens
            if not bool(writable.any()):
                return
            positions = positions[writable].contiguous()
            key = key[writable].contiguous()
            value = value[writable].contiguous()
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

    def write_decode_batch(
        self,
        request_ids,
        layer_index: int,
        positions,
        key,
        value,
        block_table,
        *,
        logical_positions,
    ) -> None:
        """Write one decode KV token per request without per-row launches."""
        import torch

        request_ids = tuple(int(request_id) for request_id in request_ids)
        logical_positions = tuple(int(position) for position in logical_positions)
        batch = len(request_ids)
        if not (
            len(logical_positions) == batch
            and positions.shape == (batch,)
            and key.shape == value.shape
            and key.shape
            == (batch, self.model.num_kv_heads, self.model.head_dim)
            and block_table.shape[0] == batch
        ):
            raise ValueError("invalid batched decode KV metadata")
        for request_id, position in zip(
            request_ids, logical_positions, strict=True
        ):
            allocation = self.block_manager.get(request_id)
            if not 0 <= position < allocation.num_tokens:
                raise IndexError("decode KV position is outside the request allocation")
            if position < self.matched_prefix_tokens(request_id):
                raise RuntimeError("decode attempted to overwrite a shared prefix page")
            required_blocks = self.block_manager.blocks_required(position + 1)
            if required_blocks > block_table.shape[1]:
                raise IndexError("decode block table does not cover the write position")
        slot = self._layer_slot(layer_index)
        positions = positions.to(device=self.device, dtype=torch.int32).contiguous()
        key = key.to(device=self.device, dtype=self.dtype).contiguous()
        value = value.to(device=self.device, dtype=self.dtype).contiguous()
        block_table = block_table.to(
            device=self.device, dtype=torch.int32
        ).contiguous()
        if self.device.type == "cuda":
            from hydraserve.kernels.kv_cache import write_paged_kv_batch

            write_paged_kv_batch(
                key,
                value,
                positions,
                block_table,
                self.key[slot],
                self.value[slot],
            )
            return
        for row, request_id in enumerate(request_ids):
            self.write(
                request_id,
                layer_index,
                positions[row : row + 1],
                key[row : row + 1],
                value[row : row + 1],
            )

    def read(self, request_id: int, layer_index: int, *, num_tokens: int | None = None):
        """Gather one request's logical K/V sequence from physical pages."""
        import torch

        slot = self._layer_slot(layer_index)
        allocation = self.block_manager.get(request_id)
        length = allocation.num_tokens if num_tokens is None else num_tokens
        if not 0 <= length <= allocation.num_tokens:
            raise ValueError("read length exceeds the request allocation")
        if not allocation.block_ids:
            empty = torch.empty(
                0,
                self.model.num_kv_heads,
                self.model.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            return empty, empty.clone()
        physical = torch.tensor(
            allocation.block_ids, device=self.device, dtype=torch.long
        )
        keys = self.key[slot, physical].reshape(
            -1, self.model.num_kv_heads, self.model.head_dim
        )[:length]
        values = self.value[slot, physical].reshape(
            -1, self.model.num_kv_heads, self.model.head_dim
        )[:length]
        return keys.contiguous(), values.contiguous()

    def batch_metadata(
        self, request_ids: Iterable[int], *, logical_lengths: Iterable[int] | None = None
    ):
        import torch

        allocations = [self.block_manager.get(request_id) for request_id in request_ids]
        if not allocations:
            raise ValueError("cannot build empty KV batch metadata")
        width = max(len(allocation.block_ids) for allocation in allocations)
        table = torch.full(
            (len(allocations), width), -1, device=self.device, dtype=torch.int32
        )
        lengths = torch.empty(len(allocations), device=self.device, dtype=torch.int32)
        lengths_override = (
            [allocation.num_tokens for allocation in allocations]
            if logical_lengths is None
            else list(logical_lengths)
        )
        if len(lengths_override) != len(allocations):
            raise ValueError("logical lengths must match request ids")
        for row, (allocation, logical_length) in enumerate(
            zip(allocations, lengths_override, strict=True)
        ):
            if not 0 <= logical_length <= allocation.num_tokens:
                raise ValueError("logical length exceeds the request allocation")
            table[row, : len(allocation.block_ids)] = torch.tensor(
                allocation.block_ids, device=self.device, dtype=torch.int32
            )
            lengths[row] = logical_length
        return table, lengths

    def _layer_slot(self, layer_index: int) -> int:
        try:
            return self.layer_to_slot[layer_index]
        except KeyError as exc:
            raise ValueError(f"layer {layer_index} is not a full-attention layer") from exc

    def _bytes_per_block(self) -> int:
        import torch

        element_size = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.model.num_full_attention_layers
            * self.block_manager.block_size
            * self.model.num_kv_heads
            * self.model.head_dim
            * 2
            * element_size
        )
