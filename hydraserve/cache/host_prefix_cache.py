"""Host-resident L2 cache for transferred prefix KV state."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock

import numpy as np

from hydraserve.cache.kv_quantizer import Int4Tensor


@dataclass(frozen=True, slots=True)
class HostPrefixCacheStats:
    entries: int
    bytes_used: int
    hits: int
    misses: int
    evictions: int


class HostPrefixCache:
    """Bounded LRU of CPU KV bundles keyed by model namespace and token prefix."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("host prefix cache capacity must be positive")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], object] = OrderedDict()
        self._sizes: dict[tuple[str, tuple[int, ...]], int] = {}
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
            return payload.packed.nbytes + payload.scale.nbytes
        raise TypeError("host prefix cache only accepts ndarray or Int4Tensor KV")

    def put(self, namespace: str, token_ids, payload) -> bool:
        key = (str(namespace), tuple(int(token) for token in token_ids))
        if not key[0] or not key[1]:
            raise ValueError("host prefix cache key cannot be empty")
        size = self._nbytes(payload)
        if size > self.max_bytes:
            return False
        with self._lock:
            old_size = self._sizes.pop(key, 0)
            if old_size:
                self._entries.pop(key)
                self._bytes -= old_size
            while self._entries and self._bytes + size > self.max_bytes:
                victim, _ = self._entries.popitem(last=False)
                self._bytes -= self._sizes.pop(victim)
                self._evictions += 1
            self._entries[key] = deepcopy(payload)
            self._sizes[key] = size
            self._bytes += size
            return True

    def get(self, namespace: str, token_ids):
        key = (str(namespace), tuple(int(token) for token in token_ids))
        with self._lock:
            payload = self._entries.get(key)
            if payload is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return deepcopy(payload)

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
