from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True, slots=True)
class BlockAllocation:
    request_id: int
    block_ids: tuple[int, ...]
    num_tokens: int


class KVBlockManager:
    """Thread-safe allocator for paged-attention block identities."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        self._allocations: dict[int, BlockAllocation] = {}
        self._lock = RLock()

    @property
    def num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free)

    def blocks_required(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate(self, request_id: int, num_tokens: int) -> BlockAllocation:
        required = self.blocks_required(num_tokens)
        with self._lock:
            if request_id in self._allocations:
                raise ValueError(f"request {request_id} already owns KV blocks")
            if required > len(self._free):
                raise MemoryError(f"need {required} KV blocks, only {len(self._free)} are free")
            block_ids = tuple(self._free[:required])
            del self._free[:required]
            allocation = BlockAllocation(request_id, block_ids, num_tokens)
            self._allocations[request_id] = allocation
            return allocation

    def grow(self, request_id: int, additional_tokens: int) -> BlockAllocation:
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        with self._lock:
            current = self.get(request_id)
            num_tokens = current.num_tokens + additional_tokens
            extra = self.blocks_required(num_tokens) - len(current.block_ids)
            if extra > len(self._free):
                raise MemoryError(f"need {extra} additional KV blocks")
            block_ids = current.block_ids + tuple(self._free[:extra])
            del self._free[:extra]
            allocation = BlockAllocation(request_id, block_ids, num_tokens)
            self._allocations[request_id] = allocation
            return allocation

    def get(self, request_id: int) -> BlockAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KeyError(f"request {request_id} has no KV allocation") from exc

    def free(self, request_id: int) -> None:
        with self._lock:
            allocation = self._allocations.pop(request_id, None)
            if allocation is None:
                return
            self._free.extend(allocation.block_ids)
            self._free.sort()
