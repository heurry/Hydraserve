from __future__ import annotations

from types import SimpleNamespace
from threading import RLock

import pytest

from hydraserve.engine import (
    AdmissionDecision,
    BackendCapacity,
    MultiWorkerGenerationBackend,
    PDClusterConfig,
    ServingRequest,
)
from hydraserve.router import (
    AdaptiveRouter,
    DecodeWorkerRegistry,
    DecodeWorkerSnapshot,
    RouterConfig,
)


class FakeMultiWorkerBackend(MultiWorkerGenerationBackend):
    def __init__(self, *, prefix_affinity=None):
        self.config = SimpleNamespace(
            block_size=4,
            cache_tokens_per_worker=40,
            max_state_slots_per_worker=4,
            prefix_cache_blocks=0,
        )
        self.router = AdaptiveRouter(
            RouterConfig(short_prompt_tokens=4, long_prompt_tokens=8, force_pd_tokens=16)
        )
        self.prefix_affinity = prefix_affinity
        self.registry = DecodeWorkerRegistry(
            tuple(
                DecodeWorkerSnapshot(
                    index,
                    f"cuda:{index + 1}",
                    BackendCapacity(10, 10, 4, 4),
                )
                for index in range(2)
            )
        )
        self._reserved_blocks = [dict(), dict()]
        self._route_decisions = {}
        self._state_lock = RLock()
        self._prefill_healthy = True
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
        self.rpc_calls = []

    def _reserve_on(self, worker_id, request):
        self.rpc_calls.append(("reserve", worker_id, request.request_id))
        self._reserved_blocks[worker_id][request.request_id] = self._required_blocks(request)
        return AdmissionDecision.accept()

    def _collocated_prefill(self, worker_id, request):
        self.rpc_calls.append(("collocated", worker_id, request.request_id))
        return request.request_id + 100

    def _decode_rpc(self, worker_id, command, expected_op, request_id=None):
        self.rpc_calls.append((expected_op, worker_id, tuple(command.get("request_ids", ()))))
        if expected_op == "decode":
            ids = tuple(command["request_ids"])
            return {"op": "decode", "request_ids": ids, "token_ids": ids}
        return {"op": expected_op, "request_id": request_id}


def test_cluster_config_rejects_duplicate_or_overlapping_devices() -> None:
    with pytest.raises(ValueError, match="unique"):
        PDClusterConfig("model", ("cuda:1", "cuda:1"))
    with pytest.raises(ValueError, match="distinct"):
        PDClusterConfig("model", ("cuda:0",))


def test_multi_worker_admission_uses_prefix_affinity_and_binds_route() -> None:
    backend = FakeMultiWorkerBackend(
        prefix_affinity=lambda request, worker_id: len(request.token_ids)
        if worker_id == 1
        else 0
    )
    request = ServingRequest(5, tuple(range(3)), 2)
    assert backend.prefill(request) == 105
    assert backend.worker_for(5) == 1
    assert request.route == "collocated"
    assert ("reserve", 1, 5) in backend.rpc_calls


def test_multi_worker_decode_groups_by_worker_and_preserves_input_order() -> None:
    backend = FakeMultiWorkerBackend()
    requests = tuple(ServingRequest(index, (index + 1,), 2) for index in range(4))
    for request, worker_id in zip(requests, (0, 1, 0, 1), strict=True):
        backend.registry.bind(request.request_id, worker_id)
    assert backend.decode(requests) == (0, 1, 2, 3)
    decode_calls = [call for call in backend.rpc_calls if call[0] == "decode"]
    assert set(decode_calls) == {
        ("decode", 0, (0, 2)),
        ("decode", 1, (1, 3)),
    }


def test_multi_worker_admission_defers_when_cluster_is_full() -> None:
    backend = FakeMultiWorkerBackend()
    for worker in backend.registry.snapshots():
        backend.registry.update_capacity(
            worker.worker_id, BackendCapacity(10, 0, 4, 0)
        )
    decision = backend.admit(ServingRequest(99, (1,), 1))
    assert not decision.admitted and decision.retryable
