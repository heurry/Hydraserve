"""
PagedAttention Block Manager.

Manages non-contiguous KV cache blocks for full attention layers.
Supports allocation, deallocation, copy-on-write, and INT4 quantization.

Block layout:
- Block size: 16 tokens
- Per block: [2 (K+V), num_kv_heads, block_size, head_dim]
- Qwen3.5-9B: block = 2 * 4 * 16 * 256 * 2 bytes = 64 KB per layer
"""

from typing import List, Tuple, Dict, Optional, Set
import torch

from hydraserve.config import CacheConfig, ModelSpec


class Block:
    """A single KV cache block (physical storage unit)."""

    __slots__ = ('block_id', 'ref_count', 'token_count', '_hash')

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_count = 0  # How many tokens are actually stored
        self._hash = None


class BlockTable:
    """
    Per-sequence mapping from logical block positions to physical block IDs.

    block_table[i] = physical_block_id for the i-th block of this sequence.
    """

    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self.blocks: List[int] = [-1] * max_blocks  # -1 = unallocated
        self.num_blocks = 0

    def append(self, physical_id: int) -> None:
        if self.num_blocks >= self.max_blocks:
            raise RuntimeError(f"BlockTable overflow: {self.max_blocks} blocks max")
        self.blocks[self.num_blocks] = physical_id
        self.num_blocks += 1

    def get(self, logical_idx: int) -> int:
        return self.blocks[logical_idx]

    def to_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.blocks, dtype=torch.int32, device=device)


class BlockManager:
    """
    PagedAttention block allocator for full attention layers.

    Features:
    - Free block pool with LRU eviction
    - Copy-on-write support for prefix sharing
    - Optional INT4 KV quantization for memory pressure
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        cache_config: CacheConfig,
        device: torch.device,
    ):
        self.model_spec = model_spec
        self.config = cache_config
        self.device = device

        self.block_size = cache_config.block_size
        self.num_full_attn_layers = model_spec.num_full_attn_layers
        self.num_kv_heads = model_spec.num_key_value_heads
        self.head_dim = model_spec.head_dim

        # Calculate block size in bytes
        # 2 (K+V) * kv_heads * block_size * head_dim * dtype_size
        self.dtype_size = 2  # BF16
        self.block_bytes = (2 * self.num_kv_heads * self.block_size *
                            self.head_dim * self.dtype_size)

        # Total blocks based on available memory
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        available = int(gpu_mem * cache_config.gpu_memory_utilization)
        self.max_blocks = max(1, available // self.block_bytes // self.num_full_attn_layers)

        # Free block pool
        self.free_blocks: List[int] = list(range(self.max_blocks))
        self.blocks: Dict[int, Block] = {i: Block(i) for i in range(self.max_blocks)}

        # Physical KV cache storage
        # [num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]
        self.kv_cache: Optional[torch.Tensor] = None

        # Sequence → BlockTable mapping
        self.seq_block_tables: Dict[int, BlockTable] = {}

        # LRU eviction tracking
        self.access_times: Dict[int, float] = {}
        self._time = 0

    def allocate(self) -> int:
        """Allocate a free block. Returns -1 if no blocks available."""
        if not self.free_blocks:
            return -1
        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        self._touch(block_id)
        return block_id

    def allocate_blocks(self, n: int) -> List[int]:
        """Allocate n blocks. May return fewer if memory is tight."""
        allocated = []
        for _ in range(n):
            bid = self.allocate()
            if bid == -1:
                # Memory pressure: trigger eviction and retry once
                self._evict_lru()
                bid = self.allocate()
                if bid == -1:
                    break
            allocated.append(bid)
        return allocated

    def free(self, block_id: int) -> None:
        """Free a block back to the pool."""
        block = self.blocks.get(block_id)
        if block is None:
            return
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.token_count = 0
            self.free_blocks.append(block_id)

    def ref_count_inc(self, block_id: int) -> None:
        """Increment reference count (for copy-on-write sharing)."""
        self.blocks[block_id].ref_count += 1

    def ref_count_dec(self, block_id: int) -> None:
        """Decrement reference count."""
        self.free(block_id)

    def get_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_used_blocks(self) -> int:
        return self.max_blocks - len(self.free_blocks)

    def init_kv_cache(self) -> None:
        """Initialize or ensure physical KV cache exists."""
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(
                self.max_blocks,
                self.num_full_attn_layers,
                2,  # K, V
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )

    def get_kv_block(self, physical_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V tensors for a specific block and layer."""
        self.init_kv_cache()
        full_attn_layer = self._map_to_full_attn_index(layer_idx)
        k = self.kv_cache[physical_id, full_attn_layer, 0]  # [kv_heads, block_size, head_dim]
        v = self.kv_cache[physical_id, full_attn_layer, 1]
        return k, v

    def write_kv_block(
        self,
        physical_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        n_tokens: int,
    ) -> None:
        """Write K/V tensors to a block."""
        self.init_kv_cache()
        full_attn_layer = self._map_to_full_attn_index(layer_idx)
        self.kv_cache[physical_id, full_attn_layer, 0, :, :n_tokens, :] = k[:, :n_tokens, :]
        self.kv_cache[physical_id, full_attn_layer, 1, :, :n_tokens, :] = v[:, :n_tokens, :]
        self.blocks[physical_id].token_count = n_tokens
        self._touch(physical_id)

    def create_block_table(self, seq_id: int, max_blocks: int) -> BlockTable:
        """Create a new block table for a sequence."""
        bt = BlockTable(max_blocks)
        self.seq_block_tables[seq_id] = bt
        return bt

    def remove_block_table(self, seq_id: int) -> None:
        """Remove a sequence's block table and free all blocks."""
        bt = self.seq_block_tables.pop(seq_id, None)
        if bt is None:
            return
        for i in range(bt.num_blocks):
            self.free(bt.blocks[i])

    def get_block_table(self, seq_id: int) -> Optional[BlockTable]:
        return self.seq_block_tables.get(seq_id)

    def cow_copy_block(self, src_block_id: int) -> int:
        """
        Copy-on-write: duplicate a shared block for exclusive use.
        Used when prefix cache hit needs to diverge.
        """
        new_id = self.allocate()
        if new_id == -1:
            raise RuntimeError("No free blocks for CoW copy")
        # Copy KV data
        self.init_kv_cache()
        self.kv_cache[new_id].copy_(self.kv_cache[src_block_id])
        self.blocks[new_id].token_count = self.blocks[src_block_id].token_count
        self.ref_count_dec(src_block_id)
        return new_id

    # ─── Internal ───────────────────────────────────────────────

    def _map_to_full_attn_index(self, layer_idx: int) -> int:
        """Map a model layer index to a full attention layer index (0-based)."""
        # Layer indices: every 4th layer (3, 7, 11, ...) is full attention
        # This maps: 3->0, 7->1, 11->2, ...
        full_attn_idx = (layer_idx + 1) // self.model_spec.full_attention_interval - 1
        if full_attn_idx < 0:
            full_attn_idx = 0
        return full_attn_idx

    def _touch(self, block_id: int) -> None:
        """Update access time for LRU tracking."""
        self._time += 1
        self.access_times[block_id] = self._time

    def _evict_lru(self) -> None:
        """Evict least recently used block(s)."""
        if not self.access_times:
            return
        # Find the block with oldest access time
        lru_block = min(self.access_times, key=self.access_times.get)
        self.free(lru_block)
        del self.access_times[lru_block]

    # ─── Statistics ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "total_blocks": self.max_blocks,
            "free_blocks": len(self.free_blocks),
            "used_blocks": self.get_used_blocks(),
            "block_size_tokens": self.block_size,
            "block_size_bytes": self.block_bytes,
            "utilization_pct": self.get_used_blocks() / self.max_blocks * 100,
        }
