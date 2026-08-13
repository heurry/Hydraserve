from __future__ import annotations

from types import SimpleNamespace
from threading import Event, Lock, RLock
from queue import Queue

import pytest

from hydraserve.engine import (
    AdmissionDecision,
    BackendCapacity,
    MultiWorkerGenerationBackend,
    PartialDecodeError,
    PDClusterConfig,
    ServingRequest,
    WorkerUnavailableError,
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
        self._closed = False
        self._recovery_stop = Event()
        self._recovering_workers = set()
        self._recovery_threads = {}
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._recovery_failures = 0
        self.max_worker_restarts = 3
        self.worker_restart_backoff_s = 0
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


def test_multi_worker_decode_isolates_failed_worker_group() -> None:
    class OneWorkerFails(FakeMultiWorkerBackend):
        def _decode_rpc(self, worker_id, command, expected_op, request_id=None):
            if expected_op == "decode" and worker_id == 1:
                raise RuntimeError("worker 1 failed")
            return super()._decode_rpc(worker_id, command, expected_op, request_id)

    backend = OneWorkerFails()
    requests = tuple(ServingRequest(index, (index + 1,), 2) for index in range(4))
    for request, worker_id in zip(requests, (0, 1, 0, 1), strict=True):
        backend.registry.bind(request.request_id, worker_id)
    with pytest.raises(PartialDecodeError) as raised:
        backend.decode(requests)
    assert raised.value.token_ids == {0: 0, 2: 2}
    assert set(raised.value.errors) == {1, 3}
    assert all("worker 1 failed" in str(error) for error in raised.value.errors.values())


def test_multi_worker_admission_defers_when_cluster_is_full() -> None:
    backend = FakeMultiWorkerBackend()
    for worker in backend.registry.snapshots():
        backend.registry.update_capacity(
            worker.worker_id, BackendCapacity(10, 0, 4, 0)
        )
    decision = backend.admit(ServingRequest(99, (1,), 1))
    assert not decision.admitted and decision.retryable


def test_dead_decode_worker_is_removed_and_recovery_is_scheduled() -> None:
    class DeadProcess:
        def is_alive(self):
            return False

    backend = FakeMultiWorkerBackend()
    backend._decode_processes = [DeadProcess(), DeadProcess()]
    backend._decode_locks = [Lock(), Lock()]
    backend._decode_commands = [Queue(), Queue()]
    backend._decode_responses = [Queue(), Queue()]
    scheduled = []
    backend._schedule_decode_recovery = scheduled.append

    with pytest.raises(WorkerUnavailableError, match="not running"):
        MultiWorkerGenerationBackend._decode_rpc(
            backend, 0, {"op": "decode", "request_ids": (1,)}, "decode"
        )
    assert not backend.registry.snapshots()[0].healthy
    assert scheduled == [0]


def test_worker_recovery_retries_with_backoff_and_restores_health() -> None:
    class RecoveringBackend(FakeMultiWorkerBackend):
        def __init__(self):
            super().__init__()
            self.restart_calls = 0

        def _restart_decode_worker_once(self, worker_id):
            self.restart_calls += 1
            if self.restart_calls == 1:
                raise RuntimeError("startup failed")
            self.registry.set_health(worker_id, True)

    backend = RecoveringBackend()
    backend.registry.set_health(0, False)
    backend._recovering_workers.add(0)
    backend._recover_decode_worker(0)
    stats = backend.recovery_stats()
    assert backend.restart_calls == 2
    assert stats.attempts == 2
    assert stats.successes == 1
    assert stats.failures == 1
    assert stats.healthy_workers == 2
    assert stats.recovering_workers == ()
