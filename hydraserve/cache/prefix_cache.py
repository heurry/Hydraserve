from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from typing import Iterable, Protocol


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Identity boundary for KV compatibility.

    KV pages must never be reused across a model, tokenizer, model revision, or
    adapter boundary even when the numerical token ids happen to be equal.
    """

    model: str = "default"
    tokenizer_revision: str = "default"
    model_revision: str = "default"
    adapter: str = "none"


DEFAULT_NAMESPACE = CacheNamespace()


@dataclass(frozen=True, slots=True)
class PrefixCandidate:
    namespace: CacheNamespace
    token_ids: tuple[int, ...]
    blocks: int
    block_size: int
    frequency: int
    recompute_cost_ms: float
    bytes_per_block: int
    cache_capacity_blocks: int | None


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    namespace: CacheNamespace
    prefix_tokens: int
    frequency: int
    hits: int
    last_access: int
    recompute_cost_ms: float
    bytes_per_block: int


class PrefixCachePolicy(Protocol):
    def admit(self, candidate: PrefixCandidate) -> tuple[bool, str | None]: ...

    def eviction_score(self, entry: PrefixEntry, now: int) -> float: ...


@dataclass(frozen=True, slots=True)
class CostAwarePrefixPolicy:
    """Frequency/cost/size-aware baseline suitable for online replacement.

    ``minimum_frequency=2`` acts as a small doorkeeper against one-hit scans.
    The default remains one for backwards compatibility and small deployments.
    """

    minimum_frequency: int = 1
    minimum_prefix_tokens: int = 0
    maximum_entry_fraction: float = 1.0
    recency_half_life: float = 1024.0

    def __post_init__(self) -> None:
        if self.minimum_frequency <= 0 or self.minimum_prefix_tokens < 0:
            raise ValueError("invalid prefix admission thresholds")
        if not 0 < self.maximum_entry_fraction <= 1:
            raise ValueError("maximum_entry_fraction must be in (0, 1]")
        if self.recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")

    def admit(self, candidate: PrefixCandidate) -> tuple[bool, str | None]:
        prefix_tokens = candidate.blocks * candidate.block_size
        if prefix_tokens < self.minimum_prefix_tokens:
            return False, "prefix is below the minimum reusable length"
        if candidate.frequency < self.minimum_frequency:
            return False, "prefix has not passed the frequency doorkeeper"
        capacity = candidate.cache_capacity_blocks
        if capacity is not None and candidate.blocks > max(
            1, int(capacity * self.maximum_entry_fraction)
        ):
            return False, "prefix would consume too much of the cache"
        return True, None

    def eviction_score(self, entry: PrefixEntry, now: int) -> float:
        age = max(0, now - entry.last_access)
        freshness = 1.0 / (1.0 + age / self.recency_half_life)
        reuse = max(1, entry.frequency + entry.hits)
        cost = max(entry.recompute_cost_ms, float(entry.prefix_tokens), 1e-9)
        size = max(entry.bytes_per_block, 1)
        return reuse * cost * freshness / size


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    matched_tokens: int
    block_ids: tuple[int, ...]
    admitted: bool = True
    evicted_block_ids: tuple[int, ...] = ()
    inserted_block_ids: tuple[int, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PrefixCacheStats:
    cached_blocks: int
    hits: int
    misses: int
    admissions: int
    rejected_admissions: int
    evictions: int
    namespaces: int
    frequency_entries: int
    referenced_blocks: int
    evictable_blocks: int
    cached_bytes: int
    hit_tokens: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    evicted_by_reason: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class _RadixNode:
    token_block: tuple[int, ...] | None
    block_id: int | None = None
    parent: "_RadixNode | None" = None
    children: dict[tuple[int, ...], "_RadixNode"] = field(default_factory=dict)
    references: int = 0
    last_access: int = 0
    frequency: int = 0
    hits: int = 0
    prefix_tokens: int = 0
    recompute_cost_ms: float = 0.0
    bytes_per_block: int = 1


class PrefixCache:
    """Namespace-safe block radix cache for full-attention KV only.

    GDN recurrent and convolution state is intentionally absent from this API;
    callers must obtain it from prefill/transfer for every request. Evicted
    physical block ids are returned to the caller for explicit allocator
    reclamation; this class never silently frees external storage.
    """

    def __init__(
        self,
        block_size: int = 16,
        *,
        max_blocks: int | None = None,
        policy: PrefixCachePolicy | None = None,
        max_frequency_entries: int = 65_536,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_blocks is not None and max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if max_frequency_entries <= 0:
            raise ValueError("max_frequency_entries must be positive")
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.policy = policy or CostAwarePrefixPolicy()
        self.max_frequency_entries = max_frequency_entries
        self._roots: dict[CacheNamespace, _RadixNode] = {}
        self._clock = count(1)
        self._frequency: dict[tuple[CacheNamespace, tuple[int, ...]], int] = {}
        self._cached_blocks = 0
        self._hits = 0
        self._misses = 0
        self._admissions = 0
        self._rejected_admissions = 0
        self._evictions = 0
        self._hit_tokens = 0
        self._rejection_reasons: dict[str, int] = {}
        self._eviction_reasons: dict[str, int] = {}
        self._lock = RLock()

    def insert(
        self,
        token_ids: Iterable[int],
        block_ids: Iterable[int],
        *,
        namespace: CacheNamespace = DEFAULT_NAMESPACE,
        recompute_cost_ms: float | None = None,
        bytes_per_block: int = 1,
    ) -> PrefixMatch:
        tokens = tuple(int(token) for token in token_ids)
        blocks = tuple(int(block) for block in block_ids)
        full_blocks = min(len(tokens) // self.block_size, len(blocks))
        if recompute_cost_ms is not None and recompute_cost_ms < 0:
            raise ValueError("recompute_cost_ms cannot be negative")
        if bytes_per_block <= 0:
            raise ValueError("bytes_per_block must be positive")
        if full_blocks == 0:
            return PrefixMatch(0, ())
        canonical_tokens = tokens[: full_blocks * self.block_size]
        with self._lock:
            root = self._root(namespace)
            node = root
            existing_path: list[_RadixNode] = []
            new_blocks = 0
            for index in range(full_blocks):
                token_block = canonical_tokens[
                    index * self.block_size : (index + 1) * self.block_size
                ]
                child = node.children.get(token_block)
                if child is None:
                    new_blocks += 1
                    child = _RadixNode(token_block, parent=node)
                    node.children[token_block] = child
                    node = child
                    continue
                if child.block_id is not None and child.block_id != blocks[index]:
                    raise ValueError("the same token prefix is already mapped to another KV block")
                existing_path.append(child)
                node = child
            shared_blocks = len(existing_path)
            # Frequency is counted on the shared prefix, not the full prompt
            # sequence: distinct tails must not stop a reused prefix from
            # passing the doorkeeper. With no shared path yet, record the full
            # sequence so an identical second sighting is admitted. A fresh
            # shared-prefix key gets +1 because the tree path itself proves
            # one earlier sighting.
            if shared_blocks:
                shared_key = (
                    namespace,
                    canonical_tokens[: shared_blocks * self.block_size],
                )
                first_recording = self._frequency.get(shared_key) is None
                frequency = self._record_frequency(shared_key)
                if first_recording:
                    frequency += 1
            else:
                frequency = self._record_frequency((namespace, canonical_tokens))

            def candidate(blocks_count: int, freq: int) -> PrefixCandidate:
                return PrefixCandidate(
                    namespace=namespace,
                    token_ids=canonical_tokens[: blocks_count * self.block_size],
                    blocks=blocks_count,
                    block_size=self.block_size,
                    frequency=freq,
                    recompute_cost_ms=float(
                        recompute_cost_ms
                        if recompute_cost_ms is not None
                        else blocks_count * self.block_size
                    ),
                    bytes_per_block=bytes_per_block,
                    cache_capacity_blocks=self.max_blocks,
                )

            # Admit the full entry first; fall back to the shared prefix so a
            # fresh tail never hides a reusable prefix behind the doorkeeper.
            full_frequency = frequency if shared_blocks == full_blocks else 1
            admitted_blocks = 0
            admitted, reason = self.policy.admit(
                candidate(full_blocks, full_frequency)
            )
            if admitted:
                admitted_blocks = full_blocks
            elif shared_blocks:
                admitted, reason = self.policy.admit(
                    candidate(shared_blocks, frequency)
                )
                if admitted:
                    admitted_blocks = shared_blocks
            if not admitted_blocks:
                self._rejected_admissions += 1
                self._record_rejection(reason)
                self._prune_blockless_leaves()
                return PrefixMatch(0, (), admitted=False, reason=reason)

            admitted_candidate = candidate(admitted_blocks, frequency)
            evicted: tuple[int, ...] = ()
            if self.max_blocks is not None:
                refillable = sum(
                    1 for path_node in existing_path if path_node.block_id is None
                )
                new_cached = refillable + (
                    new_blocks if admitted_blocks > shared_blocks else 0
                )
                required = max(0, self._cached_blocks + new_cached - self.max_blocks)
                protected = {id(path_node) for path_node in existing_path}
                available = self._count_evictable(protected)
                if required > available:
                    self._rejected_admissions += 1
                    self._record_rejection("cache has no evictable capacity")
                    return PrefixMatch(
                        0,
                        (),
                        admitted=False,
                        reason="cache has no evictable capacity",
                    )
                evicted = self._evict_locked(required, protected, "cache_capacity")

            node = root
            cost_per_block = admitted_candidate.recompute_cost_ms / admitted_blocks
            inserted_indices: list[int] = []
            for index in range(admitted_blocks):
                token_block = canonical_tokens[
                    index * self.block_size : (index + 1) * self.block_size
                ]
                child = node.children.get(token_block)
                if child is None:
                    child = _RadixNode(token_block, parent=node)
                    node.children[token_block] = child
                if child.block_id is None:
                    inserted_indices.append(index)
                    self._cached_blocks += 1
                child.block_id = blocks[index]
                child.last_access = next(self._clock)
                child.frequency = max(child.frequency, frequency)
                child.prefix_tokens = (index + 1) * self.block_size
                child.recompute_cost_ms = cost_per_block * (index + 1)
                child.bytes_per_block = bytes_per_block
                node = child
            self._admissions += 1
            inserted = tuple(blocks[index] for index in inserted_indices)
            return PrefixMatch(
                admitted_blocks * self.block_size,
                blocks[:admitted_blocks],
                evicted_block_ids=evicted,
                inserted_block_ids=inserted,
            )

    def match(
        self,
        token_ids: Iterable[int],
        *,
        namespace: CacheNamespace = DEFAULT_NAMESPACE,
        acquire: bool = False,
    ) -> PrefixMatch:
        tokens = tuple(int(token) for token in token_ids)
        matched: list[int] = []
        with self._lock:
            node = self._roots.get(namespace)
            if node is None:
                self._misses += 1
                return PrefixMatch(0, ())
            for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
                token_block = tokens[start : start + self.block_size]
                child = node.children.get(token_block)
                if child is None or child.block_id is None:
                    break
                child.last_access = next(self._clock)
                child.hits += 1
                matched.append(child.block_id)
                node = child
            if matched:
                self._hits += 1
                self._hit_tokens += len(matched) * self.block_size
                if acquire:
                    self._adjust_references(namespace, tokens, len(matched), 1)
            else:
                self._misses += 1
        return PrefixMatch(len(matched) * self.block_size, tuple(matched))

    def release(
        self,
        token_ids: Iterable[int],
        matched_tokens: int,
        *,
        namespace: CacheNamespace = DEFAULT_NAMESPACE,
    ) -> None:
        if matched_tokens % self.block_size:
            raise ValueError("matched_tokens must be block aligned")
        tokens = tuple(int(token) for token in token_ids)
        with self._lock:
            self._adjust_references(
                namespace, tokens, matched_tokens // self.block_size, -1
            )

    def evict(
        self, max_blocks: int = 1, *, reason: str = "manual"
    ) -> tuple[int, ...]:
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if not reason:
            raise ValueError("eviction reason cannot be empty")
        with self._lock:
            return self._evict_locked(max_blocks, set(), reason)

    def stats(self) -> PrefixCacheStats:
        with self._lock:
            nodes = [
                node
                for root in self._roots.values()
                for node in self._nodes(root)
                if node.block_id is not None
            ]
            return PrefixCacheStats(
                cached_blocks=self._cached_blocks,
                hits=self._hits,
                misses=self._misses,
                admissions=self._admissions,
                rejected_admissions=self._rejected_admissions,
                evictions=self._evictions,
                namespaces=len(self._roots),
                frequency_entries=len(self._frequency),
                referenced_blocks=sum(node.references > 0 for node in nodes),
                evictable_blocks=self._count_evictable(set()),
                cached_bytes=sum(node.bytes_per_block for node in nodes),
                hit_tokens=self._hit_tokens,
                rejected_by_reason=tuple(sorted(self._rejection_reasons.items())),
                evicted_by_reason=tuple(sorted(self._eviction_reasons.items())),
            )

    def _record_rejection(self, reason: str | None) -> None:
        key = reason or "unspecified"
        self._rejection_reasons[key] = self._rejection_reasons.get(key, 0) + 1

    def _prune_blockless_leaves(self) -> None:
        """Bound metadata growth from rejected scans.

        Rejected inserts leave block-less radix nodes behind so a second
        sighting of the same prefix can refill them; prune the leaf excess
        once the total exceeds the frequency-metadata budget.
        """
        blockless = sum(
            1
            for root in self._roots.values()
            for node in self._nodes(root)
            if node.block_id is None
        )
        excess = blockless - self.max_frequency_entries
        if excess <= 0:
            return
        victims = [
            (namespace, node)
            for namespace, root in self._roots.items()
            for node in self._leaves(root)
            if node.block_id is None
        ][:excess]
        for _, victim in victims:
            parent = victim.parent
            if parent is not None and victim.token_block is not None:
                parent.children.pop(victim.token_block, None)
        for namespace, root in tuple(self._roots.items()):
            if not root.children:
                self._roots.pop(namespace, None)

    def _record_frequency(
        self, key: tuple[CacheNamespace, tuple[int, ...]]
    ) -> int:
        current = self._frequency.get(key)
        if current is not None:
            current += 1
            self._frequency[key] = current
            return current
        if len(self._frequency) >= self.max_frequency_entries:
            # TinyLFU-style periodic aging bounds metadata and lets old scans
            # disappear instead of retaining every observed prompt forever.
            self._frequency = {
                existing: value // 2
                for existing, value in self._frequency.items()
                if value // 2 > 0
            }
            if len(self._frequency) >= self.max_frequency_entries:
                least_frequent = min(
                    self._frequency, key=self._frequency.__getitem__
                )
                self._frequency.pop(least_frequent)
        self._frequency[key] = 1
        return 1

    def _root(self, namespace: CacheNamespace) -> _RadixNode:
        root = self._roots.get(namespace)
        if root is None:
            root = _RadixNode(None)
            self._roots[namespace] = root
        return root

    def _adjust_references(
        self,
        namespace: CacheNamespace,
        tokens: tuple[int, ...],
        blocks: int,
        delta: int,
    ) -> None:
        try:
            node = self._roots[namespace]
        except KeyError as exc:
            raise KeyError("prefix namespace is no longer present") from exc
        path: list[_RadixNode] = []
        for index in range(blocks):
            token_block = tokens[
                index * self.block_size : (index + 1) * self.block_size
            ]
            try:
                node = node.children[token_block]
            except KeyError as exc:
                raise KeyError("prefix is no longer present in the cache") from exc
            path.append(node)
        if delta < 0 and any(path_node.references == 0 for path_node in path):
            raise RuntimeError("prefix reference count would become negative")
        for path_node in path:
            path_node.references += delta

    def _evict_locked(
        self, max_blocks: int, protected: set[int], reason: str
    ) -> tuple[int, ...]:
        evicted: list[int] = []
        while len(evicted) < max_blocks:
            candidates: list[tuple[CacheNamespace, _RadixNode]] = []
            for namespace, root in self._roots.items():
                candidates.extend(
                    (namespace, node)
                    for node in self._leaves(root)
                    if node.references == 0 and id(node) not in protected
                )
            if not candidates:
                break
            now = next(self._clock)
            namespace, victim = min(
                candidates,
                key=lambda item: self.policy.eviction_score(
                    self._entry(item[0], item[1]), now
                ),
            )
            if victim.block_id is not None:
                evicted.append(victim.block_id)
                self._cached_blocks -= 1
                self._evictions += 1
                self._eviction_reasons[reason] = (
                    self._eviction_reasons.get(reason, 0) + 1
                )
            parent = victim.parent
            if parent is not None and victim.token_block is not None:
                parent.children.pop(victim.token_block, None)
            root = self._roots.get(namespace)
            if root is not None and not root.children:
                self._roots.pop(namespace, None)
        return tuple(evicted)

    def _count_evictable(self, protected: set[int]) -> int:
        # Iterative post-order so a 128K-token prefix (thousands of blocks) does
        # not overflow the Python recursion limit.
        total = 0
        for root in self._roots.values():
            nodes: list[_RadixNode] = []
            stack = list(root.children.values())
            while stack:
                node = stack.pop()
                nodes.append(node)
                stack.extend(node.children.values())
            blocked: dict[int, bool] = {}
            count_blocks = 0
            for node in reversed(nodes):
                is_blocked = id(node) in protected or node.references > 0
                if not is_blocked:
                    is_blocked = any(
                        blocked.get(id(child), False) for child in node.children.values()
                    )
                blocked[id(node)] = is_blocked
                if not is_blocked and node.block_id is not None:
                    count_blocks += 1
            total += count_blocks
        return total

    @staticmethod
    def _entry(namespace: CacheNamespace, node: _RadixNode) -> PrefixEntry:
        return PrefixEntry(
            namespace=namespace,
            prefix_tokens=node.prefix_tokens,
            frequency=node.frequency,
            hits=node.hits,
            last_access=node.last_access,
            recompute_cost_ms=node.recompute_cost_ms,
            bytes_per_block=node.bytes_per_block,
        )

    @classmethod
    def _leaves(cls, node: _RadixNode) -> list[_RadixNode]:
        leaves: list[_RadixNode] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if not current.children:
                if current.parent is not None:
                    leaves.append(current)
            else:
                stack.extend(current.children.values())
        return leaves

    @classmethod
    def _nodes(cls, node: _RadixNode) -> list[_RadixNode]:
        nodes: list[_RadixNode] = []
        if node.parent is not None:
            nodes.append(node)
        stack = list(node.children.values())
        while stack:
            current = stack.pop()
            nodes.append(current)
            stack.extend(current.children.values())
        return nodes
