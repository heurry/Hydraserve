from __future__ import annotations

import pytest

from hydraserve.cache import PrefixCache


def test_prefix_cache_matches_only_complete_blocks() -> None:
    cache = PrefixCache(block_size=4)
    inserted = cache.insert(range(10), (7, 9, 11))
    assert inserted.matched_tokens == 8
    assert inserted.block_ids == (7, 9)
    matched = cache.match(tuple(range(8)) + (99, 100))
    assert matched.matched_tokens == 8
    assert matched.block_ids == (7, 9)


def test_prefix_cache_branches_at_block_boundary() -> None:
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2, 3, 4), (10, 11))
    cache.insert((1, 2, 8, 9), (10, 12))
    assert cache.match((1, 2, 3, 4)).block_ids == (10, 11)
    assert cache.match((1, 2, 8, 9)).block_ids == (10, 12)
    with pytest.raises(ValueError, match="another KV block"):
        cache.insert((1, 2), (99,))


def test_referenced_prefix_is_not_evicted() -> None:
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2, 3, 4), (20, 21))
    cache.insert((8, 9), (30,))
    match = cache.match((1, 2, 3, 4), acquire=True)
    assert cache.evict(2) == (30,)
    cache.release((1, 2, 3, 4), match.matched_tokens)
    # Leaves are evicted from the end of a shared path; the parent becomes a
    # candidate on the next call.
    assert cache.evict(2) == (21, 20)


def test_release_cannot_underflow_reference_count() -> None:
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2), (3,))
    with pytest.raises(RuntimeError, match="negative"):
        cache.release((1, 2), 2)
