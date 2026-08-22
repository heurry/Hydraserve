"""Host-resident block-radix L2 cache for transferred prefix KV state."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock

import numpy as np

from hydraserve.cache.kv_quantizer import Int4Tensor, Int8Tensor


@dataclass(frozen=True, slots=True)
class HostPrefixCacheStats:
    entries: int
    bytes_used: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True, slots=True)
class HostPrefixMatch:
    """Deepest page-aligned host prefix and its owned KV payload."""

    matched_tokens: int
    payload: object | None
    entry_key: tuple[str, tuple[int, ...]] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(slots=True)
class _HostRadixNode:
    parent: "_HostRadixNode | None" = None
    edge: tuple[int, ...] = ()
    children: dict[tuple[int, ...], "_HostRadixNode"] = field(default_factory=dict)
    entry_key: tuple[str, tuple[int, ...]] | None = None
    source_keys: set[tuple[str, tuple[int, ...]]] = field(default_factory=set)


class HostPrefixCache:
    """Bounded LRU backed by a namespace-safe block radix index.

    Payloads are owned by the cache and returned by reference. Transfer
    backends already materialize independent NumPy arrays, so an additional
    full-prompt ``deepcopy`` on every admission and hit only doubles memory
    traffic. Callers must treat returned payloads as immutable.
    """

    def __init__(self, max_bytes: int, block_size: int = 1) -> None:
        if max_bytes <= 0:
            raise ValueError("host prefix cache capacity must be positive")
        if block_size <= 0:
            raise ValueError("host prefix cache block_size must be positive")
        self.max_bytes = max_bytes
        self.block_size = block_size
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], object] = OrderedDict()
        self._sizes: dict[tuple[str, tuple[int, ...]], int] = {}
        self._nodes: dict[tuple[str, tuple[int, ...]], _HostRadixNode] = {}
        self._pins: dict[tuple[str, tuple[int, ...]], int] = {}
        self._roots: dict[str, _HostRadixNode] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    @staticmethod
    def _nbytes(payload) -> int:
        if isinstance(payload, np.ndarray):
            return payload.nbytes
        if isinstance(payload, Int4Tensor):
            return payload.packed.nbytes + payload.scales.nbytes
        if isinstance(payload, Int8Tensor):
            return payload.quantized.nbytes + payload.scales.nbytes
        raise TypeError("host prefix cache only accepts ndarray or quantized KV")

    def _canonical(self, token_ids) -> tuple[int, ...]:
        tokens = tuple(int(token) for token in token_ids)
        length = len(tokens) - len(tokens) % self.block_size
        return tokens[:length]

    @staticmethod
    def _slice_payload(payload, tokens: int):
        if isinstance(payload, np.ndarray):
            if payload.ndim < 3:
                return np.ascontiguousarray(payload)
            if payload.shape[2] < tokens:
                raise ValueError("host KV payload does not cover the token prefix")
            # A contiguous owned slice makes eviction accounting exact and
            # avoids retaining a larger transfer buffer through a NumPy view.
            return np.ascontiguousarray(payload[:, :, :tokens])
        if isinstance(payload, Int4Tensor):
            if payload.shape[2] != tokens:
                raise ValueError("quantized host KV must already be block aligned")
            return payload
        if isinstance(payload, Int8Tensor):
            if payload.shape[2] != tokens:
                raise ValueError("quantized host KV must already be block aligned")
            return payload
        raise TypeError("host prefix cache only accepts ndarray or quantized KV")

    def put(self, namespace: str, token_ids, payload) -> bool:
        namespace = str(namespace)
        all_tokens = tuple(int(token) for token in token_ids)
        tokens = self._canonical(all_tokens)
        if not namespace or not all_tokens:
            return False
        owned = self._slice_payload(payload, len(all_tokens))
        key = (namespace, all_tokens)
        size = self._nbytes(owned)
        if size > self.max_bytes:
            return False
        with self._lock:
            if self._pins.get(key, 0):
                # The existing immutable payload is already valid for this
                # token sequence and cannot be replaced while being restored.
                return True
            old_size = self._sizes.pop(key, 0)
            if old_size:
                self._entries.pop(key)
                self._bytes -= old_size
            reclaimable = sum(
                self._sizes[candidate]
                for candidate in self._entries
                if not self._pins.get(candidate, 0)
            )
            if self._bytes + size - reclaimable > self.max_bytes:
                return False
            while self._entries and self._bytes + size > self.max_bytes:
                victim = next(
                    (candidate for candidate in self._entries if not self._pins.get(candidate, 0)),
                    None,
                )
                if victim is None:
                    return False
                self._entries.pop(victim)
                self._bytes -= self._sizes.pop(victim)
                self._remove_index(victim)
                self._evictions += 1
            node = self._roots.setdefault(namespace, _HostRadixNode())
            for start in range(0, len(tokens), self.block_size):
                edge = tokens[start : start + self.block_size]
                node = node.children.setdefault(
                    edge, _HostRadixNode(parent=node, edge=edge)
                )
                node.source_keys.add(key)
            if tokens == all_tokens:
                node.entry_key = key
            self._nodes[key] = node
            self._entries[key] = owned
            self._sizes[key] = size
            self._bytes += size
            return True

    def _remove_index(self, key: tuple[str, tuple[int, ...]]) -> None:
        node = self._nodes.pop(key, None)
        if node is None:
            return
        node.entry_key = None
        while node.parent is not None:
            node.source_keys.discard(key)
            parent = node.parent
            if not node.children and node.entry_key is None and not node.source_keys:
                parent.children.pop(node.edge, None)
            node = parent
        namespace = key[0]
        root = self._roots.get(namespace)
        if root is not None and not root.children:
            self._roots.pop(namespace, None)

    def _match_key(
        self, namespace: str, token_ids
    ) -> tuple[tuple[str, tuple[int, ...]], int] | None:
        tokens = tuple(int(token) for token in token_ids)
        exact = (str(namespace), tokens)
        if exact in self._entries:
            return exact, len(tokens)
        node = self._roots.get(str(namespace))
        if node is None:
            return None
        matched = None
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            edge = tokens[start : start + self.block_size]
            node = node.children.get(edge)
            if node is None:
                break
            if node.source_keys:
                # The shortest backing entry retains the least excess host
                # memory behind the returned prefix view.
                source = min(node.source_keys, key=lambda item: len(item[1]))
                matched = source, start + self.block_size
        return matched

    def match(self, namespace: str, token_ids) -> HostPrefixMatch:
        """Return and LRU-touch the longest complete-block cached prefix."""
        with self._lock:
            matched = self._match_key(namespace, token_ids)
            if matched is None:
                self._misses += 1
                return HostPrefixMatch(0, None)
            key, matched_tokens = matched
            payload = self._entries[key]
            self._entries.move_to_end(key)
            self._hits += 1
            if (
                isinstance(payload, np.ndarray)
                and payload.ndim >= 3
                and matched_tokens < payload.shape[2]
            ):
                payload = payload[:, :, :matched_tokens]
            return HostPrefixMatch(matched_tokens, payload, key)

    def pin(self, namespace: str, token_ids) -> HostPrefixMatch:
        """Pin the longest prefix so admission and async restore cannot race eviction."""
        with self._lock:
            matched = self._match_key(namespace, token_ids)
            if matched is None:
                self._misses += 1
                return HostPrefixMatch(0, None)
            key, matched_tokens = matched
            self._pins[key] = self._pins.get(key, 0) + 1
            payload = self._entries[key]
            self._entries.move_to_end(key)
            self._hits += 1
            if (
                isinstance(payload, np.ndarray)
                and payload.ndim >= 3
                and matched_tokens < payload.shape[2]
            ):
                payload = payload[:, :, :matched_tokens]
            return HostPrefixMatch(matched_tokens, payload, key)

    def unpin(self, match: HostPrefixMatch) -> None:
        key = match.entry_key
        if key is None:
            return
        with self._lock:
            count = self._pins.get(key, 0)
            if count <= 1:
                self._pins.pop(key, None)
            else:
                self._pins[key] = count - 1

    def longest_prefix_tokens(self, namespace: str, token_ids) -> int:
        """Probe admission affinity without changing cache hit statistics."""
        with self._lock:
            matched = self._match_key(namespace, token_ids)
            return 0 if matched is None else matched[1]

    def get(self, namespace: str, token_ids):
        """Compatibility exact lookup; returns the owned immutable-by-contract array."""
        key = (str(namespace), tuple(int(token) for token in token_ids))
        with self._lock:
            payload = self._entries.get(key)
            if payload is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return payload

    def contains(self, namespace: str, token_ids) -> bool:
        key = (str(namespace), tuple(int(token) for token in token_ids))
        with self._lock:
            return key in self._entries

    def stats(self) -> HostPrefixCacheStats:
        with self._lock:
            return HostPrefixCacheStats(
                len(self._entries),
                self._bytes,
                self._hits,
                self._misses,
                self._evictions,
            )
