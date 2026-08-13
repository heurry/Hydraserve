from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    matched_tokens: int
    block_ids: tuple[int, ...]


@dataclass(slots=True)
class _RadixNode:
    token_block: tuple[int, ...] | None
    block_id: int | None = None
    parent: "_RadixNode | None" = None
    children: dict[tuple[int, ...], "_RadixNode"] = field(default_factory=dict)
    references: int = 0
    last_access: int = 0


class PrefixCache:
    """Block-aligned radix tree for full-attention KV only.

    GDN recurrent and convolution state is intentionally absent from this API;
    callers must obtain it from prefill/transfer for every request.
    """

    def __init__(self, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._root = _RadixNode(None)
        self._clock = count(1)
        self._lock = RLock()

    def insert(self, token_ids: Iterable[int], block_ids: Iterable[int]) -> PrefixMatch:
        tokens = tuple(int(token) for token in token_ids)
        blocks = tuple(int(block) for block in block_ids)
        full_blocks = min(len(tokens) // self.block_size, len(blocks))
        if full_blocks == 0:
            return PrefixMatch(0, ())
        node = self._root
        with self._lock:
            for index in range(full_blocks):
                token_block = tokens[
                    index * self.block_size : (index + 1) * self.block_size
                ]
                child = node.children.get(token_block)
                if child is None:
                    child = _RadixNode(token_block, parent=node)
                    node.children[token_block] = child
                if child.block_id is not None and child.block_id != blocks[index]:
                    raise ValueError("the same token prefix is already mapped to another KV block")
                child.block_id = blocks[index]
                child.last_access = next(self._clock)
                node = child
        return PrefixMatch(full_blocks * self.block_size, blocks[:full_blocks])

    def match(self, token_ids: Iterable[int], *, acquire: bool = False) -> PrefixMatch:
        tokens = tuple(int(token) for token in token_ids)
        node = self._root
        matched: list[int] = []
        with self._lock:
            for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
                token_block = tokens[start : start + self.block_size]
                child = node.children.get(token_block)
                if child is None or child.block_id is None:
                    break
                child.last_access = next(self._clock)
                matched.append(child.block_id)
                node = child
            if acquire and matched:
                self._adjust_references(tokens, len(matched), 1)
        return PrefixMatch(len(matched) * self.block_size, tuple(matched))

    def release(self, token_ids: Iterable[int], matched_tokens: int) -> None:
        if matched_tokens % self.block_size:
            raise ValueError("matched_tokens must be block aligned")
        tokens = tuple(int(token) for token in token_ids)
        with self._lock:
            self._adjust_references(tokens, matched_tokens // self.block_size, -1)

    def evict(self, max_blocks: int = 1) -> tuple[int, ...]:
        """Evict least-recently-used unreferenced leaves and return physical blocks."""
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        evicted: list[int] = []
        with self._lock:
            while len(evicted) < max_blocks:
                leaves = self._leaves(self._root)
                candidates = [node for node in leaves if node.references == 0]
                if not candidates:
                    break
                victim = min(candidates, key=lambda node: node.last_access)
                if victim.block_id is not None:
                    evicted.append(victim.block_id)
                parent = victim.parent
                if parent is not None and victim.token_block is not None:
                    parent.children.pop(victim.token_block, None)
        return tuple(evicted)

    def _adjust_references(
        self, tokens: tuple[int, ...], blocks: int, delta: int
    ) -> None:
        node = self._root
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
        if delta < 0 and any(node.references == 0 for node in path):
            raise RuntimeError("prefix reference count would become negative")
        for node in path:
            node.references += delta

    @classmethod
    def _leaves(cls, node: _RadixNode) -> list[_RadixNode]:
        if not node.children:
            return [] if node.parent is None else [node]
        leaves: list[_RadixNode] = []
        for child in node.children.values():
            leaves.extend(cls._leaves(child))
        return leaves
