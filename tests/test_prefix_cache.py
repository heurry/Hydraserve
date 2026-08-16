from __future__ import annotations

import pytest

from hydraserve.cache import CacheNamespace, CostAwarePrefixPolicy, PrefixCache


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


def test_same_prefix_remap_keeps_first_mapping() -> None:
    """A later request publishing the same prefix into fresh blocks must not
    raise nor overwrite the existing mapping; its blocks are not inserted."""
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2, 3, 4), (10, 11))
    remapped = cache.insert((1, 2), (99,))
    assert remapped.inserted_block_ids == ()
    # The tree still maps the shared prefix to the original blocks.
    assert cache.match((1, 2, 3, 4)).block_ids == (10, 11)
    assert cache.stats().cached_blocks == 2


def test_referenced_prefix_is_not_evicted() -> None:
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2, 3, 4), (20, 21))
    cache.insert((8, 9), (30,))
    match = cache.match((1, 2, 3, 4), acquire=True)
    stats = cache.stats()
    assert stats.referenced_blocks == 2
    assert stats.evictable_blocks == 1
    assert stats.hit_tokens == 4
    assert cache.evict(2) == (30,)
    assert dict(cache.stats().evicted_by_reason) == {"manual": 1}
    cache.release((1, 2, 3, 4), match.matched_tokens)
    # Leaves are evicted from the end of a shared path; the parent becomes a
    # candidate on the next call.
    assert cache.evict(2) == (21, 20)


def test_release_cannot_underflow_reference_count() -> None:
    cache = PrefixCache(block_size=2)
    cache.insert((1, 2), (3,))
    with pytest.raises(RuntimeError, match="negative"):
        cache.release((1, 2), 2)


def test_namespace_prevents_cross_model_kv_reuse() -> None:
    cache = PrefixCache(block_size=2)
    first = CacheNamespace(model="qwen", model_revision="a")
    second = CacheNamespace(model="qwen", model_revision="b")
    cache.insert((1, 2), (10,), namespace=first)
    cache.insert((1, 2), (20,), namespace=second)
    assert cache.match((1, 2), namespace=first).block_ids == (10,)
    assert cache.match((1, 2), namespace=second).block_ids == (20,)
    assert cache.stats().namespaces == 2


def test_frequency_doorkeeper_rejects_one_hit_scan() -> None:
    cache = PrefixCache(
        block_size=2,
        policy=CostAwarePrefixPolicy(minimum_frequency=2),
    )
    first = cache.insert((1, 2, 3, 4), (10, 11))
    assert not first.admitted
    assert "frequency" in first.reason
    second = cache.insert((1, 2, 3, 4), (10, 11))
    assert second.admitted
    assert cache.match((1, 2, 3, 4)).block_ids == (10, 11)
    stats = cache.stats()
    assert stats.rejected_admissions == 1
    assert stats.admissions == 1
    assert dict(stats.rejected_by_reason) == {
        "prefix has not passed the frequency doorkeeper": 1
    }


def test_cost_aware_eviction_returns_physical_blocks_to_reclaim() -> None:
    cache = PrefixCache(block_size=2, max_blocks=2)
    cache.insert((1, 2), (10,), recompute_cost_ms=1)
    cache.insert((3, 4), (20,), recompute_cost_ms=100)
    inserted = cache.insert((5, 6), (30,), recompute_cost_ms=10)
    assert inserted.admitted
    assert inserted.evicted_block_ids == (10,)
    assert inserted.inserted_block_ids == (30,)
    assert cache.match((1, 2)).matched_tokens == 0
    assert cache.match((3, 4)).block_ids == (20,)
    assert cache.match((5, 6)).block_ids == (30,)
    assert cache.stats().cached_blocks == 2
    assert dict(cache.stats().evicted_by_reason) == {"cache_capacity": 1}


def test_capacity_rejects_when_every_victim_is_referenced() -> None:
    cache = PrefixCache(block_size=2, max_blocks=1)
    cache.insert((1, 2), (10,))
    match = cache.match((1, 2), acquire=True)
    rejected = cache.insert((3, 4), (20,))
    assert not rejected.admitted
    assert rejected.reason == "cache has no evictable capacity"
    assert rejected.evicted_block_ids == ()
    assert cache.match((1, 2)).block_ids == (10,)
    cache.release((1, 2), match.matched_tokens)


def test_large_prefix_admission_limit_prevents_cache_pollution() -> None:
    cache = PrefixCache(
        block_size=2,
        max_blocks=10,
        policy=CostAwarePrefixPolicy(maximum_entry_fraction=0.2),
    )
    rejected = cache.insert(range(6), (1, 2, 3))
    assert not rejected.admitted
    assert "too much" in rejected.reason


def test_frequency_doorkeeper_metadata_is_bounded() -> None:
    cache = PrefixCache(
        block_size=1,
        policy=CostAwarePrefixPolicy(minimum_frequency=2),
        max_frequency_entries=2,
    )
    for token in range(20):
        cache.insert((token,), (token,))
    assert cache.stats().frequency_entries <= 2


def test_doorkeeper_admits_shared_prefix_across_distinct_tails() -> None:
    """Bug B regression: a reused prefix must pass the frequency doorkeeper
    even when each full prompt has a distinct tail."""
    cache = PrefixCache(
        block_size=2,
        policy=CostAwarePrefixPolicy(minimum_frequency=2),
    )
    first = cache.insert((1, 2, 5, 6), (10, 11))
    assert not first.admitted
    second = cache.insert((1, 2, 7, 8), (20, 21))
    assert second.admitted
    assert second.matched_tokens == 2
    assert second.inserted_block_ids == (20,)
    third = cache.match((1, 2, 9, 10))
    assert third.matched_tokens == 2
    assert third.block_ids == (20,)
    stats = cache.stats()
    assert stats.cached_blocks == 1
    assert stats.admissions == 1
    assert stats.rejected_admissions == 1


def test_doorkeeper_still_rejects_a_single_distinct_tail() -> None:
    cache = PrefixCache(
        block_size=2,
        policy=CostAwarePrefixPolicy(minimum_frequency=2),
    )
    result = cache.insert((1, 2, 5, 6), (10, 11))
    assert not result.admitted
    assert "frequency" in result.reason
    assert cache.stats().cached_blocks == 0
