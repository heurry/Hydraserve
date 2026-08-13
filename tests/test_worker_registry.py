from __future__ import annotations

import pytest

from hydraserve.engine import BackendCapacity
from hydraserve.router import (
    DecodeWorkerRegistry,
    DecodeWorkerSnapshot,
    WorkerTopology,
)


def _worker(worker_id, *, free_blocks=100, free_slots=8, bandwidth=10, hops=1):
    return DecodeWorkerSnapshot(
        worker_id,
        f"cuda:{worker_id + 1}",
        BackendCapacity(100, free_blocks, 8, free_slots),
        WorkerTopology(bandwidth, hops),
    )


def test_registry_prefers_cache_affinity_then_load_and_topology() -> None:
    registry = DecodeWorkerRegistry((_worker(0), _worker(1, free_blocks=90)))
    selected = registry.candidates(
        required_blocks=4,
        prompt_tokens=100,
        prefix_matches={1: 100},
    )
    assert selected[0].worker_id == 1
    assert selected[0].prefix_match_tokens == 100


def test_registry_filters_unhealthy_or_insufficient_workers() -> None:
    registry = DecodeWorkerRegistry(
        (_worker(0, free_blocks=2), _worker(1, free_slots=0), _worker(2))
    )
    registry.set_health(2, False)
    assert registry.candidates(required_blocks=3, prompt_tokens=10) == ()
    registry.set_health(2, True)
    assert registry.candidates(required_blocks=3, prompt_tokens=10)[0].worker_id == 2


def test_request_worker_binding_is_immutable_and_accounted() -> None:
    registry = DecodeWorkerRegistry((_worker(0), _worker(1)))
    registry.bind(10, 1)
    registry.bind(10, 1)
    assert registry.worker_for(10) == 1
    assert registry.snapshots()[1].active_requests == 1
    with pytest.raises(RuntimeError, match="already bound"):
        registry.bind(10, 0)
    assert registry.release(10) == 1
    assert registry.release(10) is None
    assert registry.snapshots()[1].active_requests == 0
