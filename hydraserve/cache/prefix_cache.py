"""
Prefix Cache Tree (Radix Tree).

Caches KV Cache blocks for shared prompt prefixes to avoid redundant prefill.

Key constraint for hybrid attention models (design doc §5.6):
- Prefix matching only considers full attention KV Cache
- Linear attention states are NOT prefix-cacheable (they're recurrent, not positional)
- skip_mamba_match strategy: match on KV, skip linear state in radix tree
- cow_mamba=False: don't copy-on-write from radix tree; state arrives via network

This aligns with SGLang Issue #32732 approach for mamba/SSM models.
"""

from typing import Dict, List, Optional, Tuple, Set
import torch
import hashlib


class RadixNode:
    """Node in the prefix cache radix tree."""

    __slots__ = ('token_ids', 'children', 'block_ids', 'ref_count',
                 'last_access_time', 'parent')

    def __init__(self, token_ids: Tuple[int, ...] = ()):
        self.token_ids: Tuple[int, ...] = token_ids
        self.children: Dict[int, RadixNode] = {}  # next_token → child
        self.block_ids: List[int] = []             # physical KV block IDs
        self.ref_count: int = 1
        self.last_access_time: float = 0.0
        self.parent: Optional[RadixNode] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_tokens(self) -> int:
        """Total tokens from root to this node."""
        count = len(self.token_ids)
        node = self.parent
        while node is not None:
            count += len(node.token_ids)
            node = node.parent
        return count


class PrefixCache:
    """
    Radix tree prefix cache for full attention KV Cache.

    Only KV Cache is cached; linear attention states are never cached
    because they are path-dependent (recurrent accumulation).
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = RadixNode()
        self._access_time = 0.0
        self.total_nodes = 1
        self.total_cached_tokens = 0

    def insert(
        self,
        token_ids: List[int],
        block_ids: List[int],
    ) -> None:
        """
        Insert a token sequence and its KV block IDs into the radix tree.

        Args:
            token_ids: Full token sequence
            block_ids: Physical block IDs for this sequence's KV cache
        """
        node = self.root
        pos = 0

        while pos < len(token_ids):
            token = token_ids[pos]

            if token in node.children:
                child = node.children[token]
                # Match common prefix
                common_len = 0
                min_len = min(len(child.token_ids), len(token_ids) - pos)
                for i in range(min_len):
                    if child.token_ids[i] == token_ids[pos + i]:
                        common_len += 1
                    else:
                        break

                if common_len == len(child.token_ids):
                    # Full match, descend
                    node = child
                    pos += common_len
                    # Update block_ids if we have more tokens
                    if pos < len(token_ids):
                        blk_idx = pos // self.block_size
                        if blk_idx < len(block_ids):
                            # Extend or replace block_ids
                            pass
                else:
                    # Partial match: split the child
                    self._split_node(child, common_len, token_ids, pos, block_ids)
                    return
            else:
                # No match: create new child
                remaining = tuple(token_ids[pos:])
                new_node = RadixNode(remaining)
                new_node.parent = node
                new_node.block_ids = list(block_ids[pos // self.block_size:])
                node.children[token] = new_node
                self.total_nodes += 1
                self.total_cached_tokens += len(remaining)
                return

        # Exact match of existing prefix: update block_ids
        node.block_ids = list(block_ids)
        node.ref_count += 1
        self._touch(node)

    def match(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """
        Match a prefix in the cache.

        Args:
            token_ids: Token sequence to match

        Returns:
            (matched_tokens, matched_block_ids):
              matched_tokens: number of prefix tokens matched
              matched_block_ids: physical block IDs for the matched prefix
        """
        node = self.root
        pos = 0
        matched_blocks = []

        while pos < len(token_ids):
            token = token_ids[pos]
            if token not in node.children:
                break

            child = node.children[token]
            common_len = 0
            min_len = min(len(child.token_ids), len(token_ids) - pos)
            for i in range(min_len):
                if child.token_ids[i] == token_ids[pos + i]:
                    common_len += 1
                else:
                    break

            if common_len == 0:
                break

            matched_blocks.extend(child.block_ids[:common_len // self.block_size])
            pos += common_len

            if common_len < len(child.token_ids):
                break

            node = child

        self._touch(node)
        return pos, matched_blocks

    def evict_oldest(self, n: int = 1) -> int:
        """Evict the n oldest-accessed leaf nodes. Returns number evicted."""
        evicted = 0
        for _ in range(n):
            leaves = self._collect_leaves(self.root)
            if not leaves:
                break
            oldest = min(leaves, key=lambda n: n.last_access_time)
            self._remove_node(oldest)
            evicted += 1
        return evicted

    def get_stats(self) -> Dict:
        return {
            "total_nodes": self.total_nodes,
            "total_cached_tokens": self.total_cached_tokens,
            "block_size": self.block_size,
        }

    # ─── Internal ───────────────────────────────────────────────

    def _split_node(
        self,
        node: RadixNode,
        common_len: int,
        new_tokens: List[int],
        new_pos: int,
        new_blocks: List[int],
    ) -> None:
        """Split a node at common_len, inserting new content."""
        # Split existing node
        old_remaining = node.token_ids[common_len:]
        old_children = dict(node.children)

        # Update existing node to be the common prefix
        node.token_ids = node.token_ids[:common_len]
        node.block_ids = node.block_ids[:common_len // self.block_size]
        node.children = {}

        # Create child for old remaining
        if old_remaining:
            old_child = RadixNode(old_remaining)
            old_child.block_ids = node.block_ids[common_len // self.block_size:]
            old_child.children = old_children
            old_child.parent = node
            old_child.last_access_time = self._access_time
            for c in old_children.values():
                c.parent = old_child
            node.children[old_remaining[0]] = old_child
            self.total_nodes += 1

        # Add new tokens
        remaining_new = tuple(new_tokens[new_pos + common_len:])
        if remaining_new:
            new_child = RadixNode(remaining_new)
            new_child.block_ids = list(new_blocks[(new_pos + common_len) // self.block_size:])
            new_child.parent = node
            new_child.last_access_time = self._access_time
            node.children[remaining_new[0]] = new_child
            self.total_nodes += 1
            self.total_cached_tokens += len(remaining_new)

    def _remove_node(self, node: RadixNode) -> None:
        """Remove a node from the tree."""
        if node.parent is None:
            return  # Can't remove root

        parent = node.parent
        if node.token_ids:
            first_token = node.token_ids[0]
            if first_token in parent.children:
                del parent.children[first_token]

        self.total_nodes -= 1
        self.total_cached_tokens -= len(node.token_ids)

        # Clean up parent if it became a leaf with no block_ids
        if parent.is_leaf and not parent.block_ids and parent.parent is not None:
            self._remove_node(parent)

    def _collect_leaves(self, node: RadixNode) -> List[RadixNode]:
        """Collect leaves recursively."""
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children.values():
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _touch(self, node: RadixNode) -> None:
        """Update access time."""
        self._access_time += 1
        node.last_access_time = self._access_time
