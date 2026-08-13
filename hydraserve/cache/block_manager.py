from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True, slots=True)
class BlockAllocation:
    request_id: int
    block_ids: tuple[int, ...]
    num_tokens: int
    reserved_tokens: int
    prefix_blocks: int = 0


@dataclass(frozen=True, slots=True)
class BlockCapacity:
    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    block_size: int
    physical_total_blocks: int
    physical_free_blocks: int
    headroom_blocks: int
    active_allocations: int
    allocation_block_references: int
    total_references: int
    shared_blocks: int
    logical_tokens: int
    reserved_tokens: int
    internal_fragmentation_tokens: int
    high_watermark_blocks: int
    allocation_failures: int

    @property
    def total_tokens(self) -> int:
        return self.total_blocks * self.block_size

    @property
    def free_tokens(self) -> int:
        return self.free_blocks * self.block_size


class KVBlockManager:
    """Thread-safe allocator for paged-attention block identities."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        *,
        headroom_blocks: int = 0,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        if not 0 <= headroom_blocks < num_blocks:
            raise ValueError("headroom_blocks must be below num_blocks")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.headroom_blocks = headroom_blocks
        self._free = list(range(num_blocks))
        self._refcounts = [0] * num_blocks
        self._allocations: dict[int, BlockAllocation] = {}
        self._high_watermark_blocks = 0
        self._allocation_failures = 0
        self._lock = RLock()

    @property
    def num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def num_allocatable_blocks(self) -> int:
        with self._lock:
            return max(0, len(self._free) - self.headroom_blocks)

    @property
    def usable_blocks(self) -> int:
        return self.num_blocks - self.headroom_blocks

    def blocks_required(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        required = self.blocks_required(num_tokens)
        with self._lock:
            return required <= max(0, len(self._free) - self.headroom_blocks)

    def allocate(
        self,
        request_id: int,
        num_tokens: int,
        *,
        reserve_tokens: int | None = None,
        prefix_block_ids=(),
    ) -> BlockAllocation:
        """Atomically allocate logical tokens and optional future capacity.

        ``num_tokens`` is the readable logical length. ``reserve_tokens`` may
        reserve pages for future decode growth without exposing unwritten KV
        positions to attention.
        """
        reservation = num_tokens if reserve_tokens is None else reserve_tokens
        prefix = tuple(int(block_id) for block_id in prefix_block_ids)
        if reservation < num_tokens:
            raise ValueError("reserve_tokens cannot be smaller than num_tokens")
        if len(set(prefix)) != len(prefix):
            raise ValueError("prefix block ids must be unique")
        if len(prefix) > num_tokens // self.block_size:
            raise ValueError("only complete logical prefix blocks can be shared")
        required = self.blocks_required(reservation)
        with self._lock:
            if request_id in self._allocations:
                raise ValueError(f"request {request_id} already owns KV blocks")
            if any(block_id < 0 or block_id >= self.num_blocks for block_id in prefix):
                raise ValueError("prefix block id is outside the allocator")
            if any(self._refcounts[block_id] <= 0 for block_id in prefix):
                raise ValueError("prefix block is not retained by the cache")
            new_required = required - len(prefix)
            if new_required < 0:
                raise ValueError("prefix contains more blocks than the reservation")
            allocatable = max(0, len(self._free) - self.headroom_blocks)
            if new_required > allocatable:
                self._allocation_failures += 1
                raise MemoryError(
                    f"need {new_required} KV blocks, only {allocatable} are allocatable "
                    f"({self.headroom_blocks} headroom)"
                )
            new_blocks = tuple(self._free[:new_required])
            del self._free[:new_required]
            for block_id in prefix:
                self._refcounts[block_id] += 1
            for block_id in new_blocks:
                self._refcounts[block_id] = 1
            block_ids = prefix + new_blocks
            allocation = BlockAllocation(
                request_id, block_ids, num_tokens, reservation, len(prefix)
            )
            self._allocations[request_id] = allocation
            self._record_high_watermark_locked()
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
            allocatable = max(0, len(self._free) - self.headroom_blocks)
            if extra > allocatable:
                self._allocation_failures += 1
                raise MemoryError(f"need {extra} additional KV blocks")
            block_ids = current.block_ids + tuple(self._free[:extra])
            del self._free[:extra]
            for block_id in block_ids[len(current.block_ids) :]:
                self._refcounts[block_id] = 1
            allocation = BlockAllocation(
                request_id,
                block_ids,
                current.num_tokens,
                max(current.reserved_tokens, total_tokens),
                current.prefix_blocks,
            )
            self._allocations[request_id] = allocation
            self._record_high_watermark_locked()
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
            allocatable = max(0, len(self._free) - self.headroom_blocks)
            if total_extra > allocatable:
                self._allocation_failures += 1
                raise MemoryError(
                    f"decode batch needs {total_extra} additional KV blocks, "
                    f"only {allocatable} are allocatable"
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
                    current.prefix_blocks,
                )
                self._allocations[current.request_id] = allocation
                results.append(allocation)
            del self._free[:cursor]
            for allocation in results:
                for block_id in allocation.block_ids:
                    if self._refcounts[block_id] == 0:
                        self._refcounts[block_id] = 1
            self._record_high_watermark_locked()
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
                current.prefix_blocks,
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
            self._release_blocks_locked(allocation.block_ids)

    def retain_blocks(self, block_ids) -> None:
        """Add an external owner, such as PrefixCache, to physical pages."""
        blocks = tuple(int(block_id) for block_id in block_ids)
        if len(set(blocks)) != len(blocks):
            raise ValueError("retained block ids must be unique")
        with self._lock:
            for block_id in blocks:
                if not 0 <= block_id < self.num_blocks:
                    raise ValueError("retained block id is outside the allocator")
                if self._refcounts[block_id] <= 0:
                    raise ValueError("cannot retain an unallocated block")
            for block_id in blocks:
                self._refcounts[block_id] += 1

    def release_blocks(self, block_ids) -> None:
        with self._lock:
            self._release_blocks_locked(tuple(int(block_id) for block_id in block_ids))

    def block_refcount(self, block_id: int) -> int:
        with self._lock:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError("block id is outside the allocator")
            return self._refcounts[block_id]

    def _release_blocks_locked(self, block_ids: tuple[int, ...]) -> None:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("released block ids must be unique")
        for block_id in block_ids:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError("released block id is outside the allocator")
            if self._refcounts[block_id] <= 0:
                raise RuntimeError(f"KV block {block_id} reference count would underflow")
        for block_id in block_ids:
            self._refcounts[block_id] -= 1
            if self._refcounts[block_id] == 0:
                self._free.append(block_id)
        self._free.sort()

    def capacity(self) -> BlockCapacity:
        with self._lock:
            physical_free = len(self._free)
            free = max(0, physical_free - self.headroom_blocks)
            allocations = tuple(self._allocations.values())
            allocation_references = sum(
                len(allocation.block_ids) for allocation in allocations
            )
            total_references = sum(self._refcounts)
            reserved_tokens = sum(
                allocation.reserved_tokens for allocation in allocations
            )
            allocated_blocks = self.num_blocks - physical_free
            return BlockCapacity(
                total_blocks=self.usable_blocks,
                free_blocks=free,
                allocated_blocks=allocated_blocks,
                block_size=self.block_size,
                physical_total_blocks=self.num_blocks,
                physical_free_blocks=physical_free,
                headroom_blocks=self.headroom_blocks,
                active_allocations=len(allocations),
                allocation_block_references=allocation_references,
                total_references=total_references,
                shared_blocks=sum(refcount > 1 for refcount in self._refcounts),
                logical_tokens=sum(
                    allocation.num_tokens for allocation in allocations
                ),
                reserved_tokens=reserved_tokens,
                internal_fragmentation_tokens=(
                    allocation_references * self.block_size - reserved_tokens
                ),
                high_watermark_blocks=self._high_watermark_blocks,
                allocation_failures=self._allocation_failures,
            )

    def audit(self) -> BlockCapacity:
        """Validate free-list, ownership, and reference-count invariants."""

        with self._lock:
            free = set(self._free)
            if len(free) != len(self._free):
                raise RuntimeError("KV free list contains duplicate block ids")
            if any(not 0 <= block_id < self.num_blocks for block_id in free):
                raise RuntimeError("KV free list contains an invalid block id")
            for block_id, refcount in enumerate(self._refcounts):
                if refcount < 0:
                    raise RuntimeError("KV block reference count is negative")
                if (block_id in free) != (refcount == 0):
                    raise RuntimeError("KV free list and reference counts disagree")
            for request_id, allocation in self._allocations.items():
                if request_id != allocation.request_id:
                    raise RuntimeError("KV allocation request id is inconsistent")
                if len(set(allocation.block_ids)) != len(allocation.block_ids):
                    raise RuntimeError("KV allocation contains duplicate blocks")
                if any(self._refcounts[block_id] <= 0 for block_id in allocation.block_ids):
                    raise RuntimeError("KV allocation references a free block")
                if not 0 <= allocation.num_tokens <= allocation.reserved_tokens:
                    raise RuntimeError("KV allocation token lengths are inconsistent")
                if self.blocks_required(allocation.reserved_tokens) > len(
                    allocation.block_ids
                ):
                    raise RuntimeError("KV allocation reservation exceeds its blocks")
        return self.capacity()

    def _record_high_watermark_locked(self) -> None:
        allocated = self.num_blocks - len(self._free)
        self._high_watermark_blocks = max(self._high_watermark_blocks, allocated)
