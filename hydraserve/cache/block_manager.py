from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True, slots=True)
class BlockAllocation:
    request_id: int
    block_ids: tuple[int, ...]
    num_tokens: int
    reserved_tokens: int


@dataclass(frozen=True, slots=True)
class BlockCapacity:
    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    block_size: int

    @property
    def total_tokens(self) -> int:
        return self.total_blocks * self.block_size

    @property
    def free_tokens(self) -> int:
        return self.free_blocks * self.block_size


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

    def can_allocate(self, num_tokens: int) -> bool:
        required = self.blocks_required(num_tokens)
        with self._lock:
            return required <= len(self._free)

    def allocate(
        self,
        request_id: int,
        num_tokens: int,
        *,
        reserve_tokens: int | None = None,
    ) -> BlockAllocation:
        """Atomically allocate logical tokens and optional future capacity.

        ``num_tokens`` is the readable logical length. ``reserve_tokens`` may
        reserve pages for future decode growth without exposing unwritten KV
        positions to attention.
        """
        reservation = num_tokens if reserve_tokens is None else reserve_tokens
        if reservation < num_tokens:
            raise ValueError("reserve_tokens cannot be smaller than num_tokens")
        required = self.blocks_required(reservation)
        with self._lock:
            if request_id in self._allocations:
                raise ValueError(f"request {request_id} already owns KV blocks")
            if required > len(self._free):
                raise MemoryError(f"need {required} KV blocks, only {len(self._free)} are free")
            block_ids = tuple(self._free[:required])
            del self._free[:required]
            allocation = BlockAllocation(request_id, block_ids, num_tokens, reservation)
            self._allocations[request_id] = allocation
            return allocation

    def reserve(self, request_id: int, total_tokens: int) -> BlockAllocation:
        """Increase a request's guaranteed capacity without changing length."""
        if total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")
        with self._lock:
            current = self.get(request_id)
            if total_tokens < current.num_tokens:
                raise ValueError("reservation cannot be smaller than logical length")
            target_blocks = self.blocks_required(total_tokens)
            extra = max(0, target_blocks - len(current.block_ids))
            if extra > len(self._free):
                raise MemoryError(f"need {extra} additional KV blocks")
            block_ids = current.block_ids + tuple(self._free[:extra])
            del self._free[:extra]
            allocation = BlockAllocation(
                request_id,
                block_ids,
                current.num_tokens,
                max(current.reserved_tokens, total_tokens),
            )
            self._allocations[request_id] = allocation
            return allocation

    def grow(self, request_id: int, additional_tokens: int) -> BlockAllocation:
        return self.grow_many((request_id,), additional_tokens=additional_tokens)[0]

    def grow_many(
        self,
        request_ids,
        *,
        additional_tokens: int = 1,
    ) -> tuple[BlockAllocation, ...]:
        """Atomically advance a decode batch's logical KV lengths."""
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        ids = tuple(int(request_id) for request_id in request_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique")
        with self._lock:
            planned = []
            total_extra = 0
            for request_id in ids:
                current = self.get(request_id)
                num_tokens = current.num_tokens + additional_tokens
                extra = max(
                    0, self.blocks_required(num_tokens) - len(current.block_ids)
                )
                planned.append((current, num_tokens, extra))
                total_extra += extra
            if total_extra > len(self._free):
                raise MemoryError(
                    f"decode batch needs {total_extra} additional KV blocks, "
                    f"only {len(self._free)} are free"
                )
            cursor = 0
            results = []
            for current, num_tokens, extra in planned:
                appended = tuple(self._free[cursor : cursor + extra])
                cursor += extra
                allocation = BlockAllocation(
                    current.request_id,
                    current.block_ids + appended,
                    num_tokens,
                    max(current.reserved_tokens, num_tokens),
                )
                self._allocations[current.request_id] = allocation
                results.append(allocation)
            del self._free[:cursor]
            return tuple(results)

    def truncate(self, request_id: int, num_tokens: int) -> BlockAllocation:
        """Roll logical length back while retaining the capacity reservation."""
        with self._lock:
            current = self.get(request_id)
            if not 0 <= num_tokens <= current.num_tokens:
                raise ValueError("truncate length must be within the allocation")
            allocation = BlockAllocation(
                request_id,
                current.block_ids,
                num_tokens,
                current.reserved_tokens,
            )
            self._allocations[request_id] = allocation
            return allocation

    def get(self, request_id: int) -> BlockAllocation:
        with self._lock:
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

    def capacity(self) -> BlockCapacity:
        with self._lock:
            free = len(self._free)
            return BlockCapacity(
                total_blocks=self.num_blocks,
                free_blocks=free,
                allocated_blocks=self.num_blocks - free,
                block_size=self.block_size,
            )
