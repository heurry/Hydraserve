"""CPU-only coordinator tests for the engine-only multi-GPU DP backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import count
from queue import Queue
from types import SimpleNamespace
from threading import Barrier, Lock, RLock

from hydraserve.engine import (
    BackendCapacity,
    MultiGPUCollocatedBackend,
    ServingRequest,
)


class _AliveProcess:
    @staticmethod
    def is_alive() -> bool:
        return True


def _coordinator(worker_count: int) -> MultiGPUCollocatedBackend:
    backend = object.__new__(MultiGPUCollocatedBackend)
    backend._processes = [_AliveProcess() for _ in range(worker_count)]
    backend._commands = [Queue() for _ in range(worker_count)]
    backend._responses = [Queue() for _ in range(worker_count)]
    backend._locks = [Lock() for _ in range(worker_count)]
    backend._response_locks = [Lock() for _ in range(worker_count)]
    backend._waiters = [{} for _ in range(worker_count)]
    backend._rpc_ids = count(1)
    backend.operation_timeout = 2.0
    backend._healthy = [True] * worker_count
    backend._pending = [0] * worker_count
    backend._assigned = [0] * worker_count
    backend._assigned_work = [0] * worker_count
    backend._prefill_tokens = [0] * worker_count
    backend._request_loads = {}
    backend._round_robin = 0
    backend._state_lock = RLock()
    backend._bound = {}
    backend._capacity = [BackendCapacity(100, 100, 64, 64)] * worker_count
    backend._capacity_versions = [-1] * worker_count
    backend.config = SimpleNamespace(
        devices=tuple(f"cuda:{index}" for index in range(worker_count)),
        block_size=16,
    )
    backend._decode_executor = ThreadPoolExecutor(max_workers=worker_count)
    return backend


def test_worker_selection_counts_persistent_assignments() -> None:
    backend = _coordinator(4)
    try:
        assert [backend._pick_worker() for _ in range(8)] == [0, 1, 2, 3] * 2
        assert backend._assigned == [2, 2, 2, 2]
    finally:
        backend._decode_executor.shutdown()


def test_worker_selection_avoids_outstanding_prefill_and_checks_capacity() -> None:
    backend = _coordinator(3)
    request = ServingRequest(1, tuple(range(32)), 8)
    backend._prefill_tokens = [4096, 0, 0]
    backend._assigned_work = [1, 100, 10]
    backend._capacity[2] = BackendCapacity(100, 1, 64, 64)
    try:
        assert backend._pick_worker(request) == 1
    finally:
        backend._decode_executor.shutdown()


def test_capacity_update_accepts_top_level_worker_payload() -> None:
    backend = _coordinator(1)
    try:
        backend._update_capacity(
            0,
            {
                "kv_total_blocks": 32,
                "kv_free_blocks": 7,
                "state_total_slots": 8,
                "state_free_slots": 3,
            },
        )
        assert backend._capacity[0] == BackendCapacity(32, 7, 8, 3)
    finally:
        backend._decode_executor.shutdown()


def test_capacity_update_does_not_regress_on_late_rpc_caller() -> None:
    backend = _coordinator(1)
    try:
        backend._update_capacity(
            0,
            {
                "response_sequence": 2,
                "kv_total_blocks": 32,
                "kv_free_blocks": 20,
                "state_total_slots": 8,
                "state_free_slots": 5,
            },
        )
        backend._update_capacity(
            0,
            {
                "response_sequence": 1,
                "kv_total_blocks": 32,
                "kv_free_blocks": 7,
                "state_total_slots": 8,
                "state_free_slots": 3,
            },
        )
        assert backend._capacity[0] == BackendCapacity(32, 20, 8, 5)
        assert backend._capacity_versions == [2]
    finally:
        backend._decode_executor.shutdown()


def test_admission_rejects_request_larger_than_every_worker() -> None:
    backend = _coordinator(2)
    backend._capacity = [
        BackendCapacity(2, 2, 64, 64),
        BackendCapacity(3, 3, 64, 64),
    ]
    request = ServingRequest(1, tuple(range(64)), 2)
    try:
        decision = backend.admit(request)
        assert not decision.admitted
        assert not decision.retryable
        assert "largest worker capacity is 3" in decision.reason
    finally:
        backend._decode_executor.shutdown()


def test_collocated_backend_opts_into_worker_independent_decode() -> None:
    backend = _coordinator(2)
    request = ServingRequest(7, (1,), 2)
    backend._bound[7] = 1
    try:
        assert backend.supports_independent_decode
        assert backend.decode_executor_parallelism == 2
        assert backend.decode_executor_group(request) == ("collocated", 1)
    finally:
        backend._decode_executor.shutdown()


def test_collocated_prefill_executor_has_one_chunk_preemption_slot_per_worker(
    monkeypatch,
) -> None:
    backend = _coordinator(2)
    first = ServingRequest(10, tuple(range(1024)), 16)
    second = ServingRequest(11, tuple(range(128)), 16)
    try:
        assert backend.prefill_executor_limits == {
            "collocated:0": 2,
            "collocated:1": 2,
        }
        monkeypatch.setenv("HYDRASERVE_DP_PREFILL_QUEUE_DEPTH", "1")
        assert backend.prefill_executor_limits == {
            "collocated:0": 1,
            "collocated:1": 1,
        }
        monkeypatch.setenv("HYDRASERVE_DP_PREFILL_QUEUE_DEPTH", "99")
        assert backend.prefill_executor_limits["collocated:0"] == 2
        # Hints do not claim capacity or advance the round-robin cursor.
        assert backend.prefill_executor_group_hint(first) == "collocated:0"
        assert backend._assigned == [0, 0]
        backend._bound[second.request_id] = 1
        assert backend.prefill_executor_group(second) == "collocated:1"
    finally:
        backend._decode_executor.shutdown()


def test_admission_retries_another_worker_and_releases_routing_load() -> None:
    backend = _coordinator(2)
    request = ServingRequest(9, tuple(range(32)), 8)

    def rpc(index, command, expected_op, request_id=None):
        assert expected_op == "admission"
        if index == 0:
            return {
                "op": "admission",
                "request_id": request_id,
                "admitted": False,
                "retryable": True,
                "reason": "worker 0 filled concurrently",
            }
        return {
            "op": "admission",
            "request_id": request_id,
            "admitted": True,
        }

    backend._rpc = rpc
    try:
        assert backend.admit(request).admitted
        assert request.worker_id == 1
        assert backend._assigned == [0, 1]
        assert backend._assigned_work == [0, 40]
        assert backend._prefill_tokens == [0, 32]

        backend._mark_prefill_complete(request.request_id, 1)
        assert backend._prefill_tokens == [0, 0]
        assert backend._assigned_work == [0, 8]

        backend.release(request.request_id)
        assert backend._assigned == [0, 0]
        assert backend._assigned_work == [0, 0]
        assert backend._request_loads == {}
    finally:
        backend._decode_executor.shutdown()


def test_correlated_rpc_allows_multiple_inflight_calls_on_one_worker() -> None:
    backend = _coordinator(1)
    calls = (
        ("decode", {"op": "decode", "request_ids": (1,)}, None),
        ("release", {"op": "release", "request_id": 2}, 2),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(backend._rpc, 0, command, expected, request_id)
                for expected, command, request_id in calls
            ]
            commands = [backend._commands[0].get(timeout=1) for _ in calls]
            assert backend._pending == [2]
            # Return responses in reverse order. rpc_id routing must still wake
            # the matching caller rather than relying on FIFO response order.
            for command in reversed(commands):
                if command["op"] == "decode":
                    backend._responses[0].put(
                        {
                            "op": "decode",
                            "request_ids": command["request_ids"],
                            "token_ids": (11,),
                            "rpc_id": command["rpc_id"],
                        }
                    )
                else:
                    backend._responses[0].put(
                        {
                            "op": "release",
                            "request_id": command["request_id"],
                            "rpc_id": command["rpc_id"],
                        }
                    )
            results = [future.result(timeout=1) for future in futures]
        assert [result["op"] for result in results] == ["decode", "release"]
        assert backend._pending == [0]
        assert backend._waiters == [{}]
    finally:
        backend._decode_executor.shutdown()


def test_decode_dispatches_workers_in_parallel_and_preserves_request_mapping() -> None:
    backend = _coordinator(2)
    barrier = Barrier(2)
    requests = (
        ServingRequest(11, (1,), 2),
        ServingRequest(10, (2,), 2),
        ServingRequest(13, (3,), 2),
        ServingRequest(12, (4,), 2),
    )
    backend._bound = {10: 0, 11: 1, 12: 0, 13: 1}

    def rpc(index, command, expected_op, request_id=None):
        assert expected_op == "decode"
        barrier.wait(timeout=1)
        request_ids = tuple(command["request_ids"])
        return {
            "op": "decode",
            "request_ids": request_ids,
            "token_ids": tuple(1000 + request_id for request_id in request_ids),
        }

    backend._rpc = rpc
    try:
        assert backend.decode(requests) == (1011, 1010, 1013, 1012)
    finally:
        backend._decode_executor.shutdown()
