"""CPU-only coordinator tests for the engine-only multi-GPU DP backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, RLock

from hydraserve.engine import MultiGPUCollocatedBackend, ServingRequest


def _coordinator(worker_count: int) -> MultiGPUCollocatedBackend:
    backend = object.__new__(MultiGPUCollocatedBackend)
    backend._processes = [object()] * worker_count
    backend._healthy = [True] * worker_count
    backend._pending = [0] * worker_count
    backend._assigned = [0] * worker_count
    backend._round_robin = 0
    backend._state_lock = RLock()
    backend._bound = {}
    backend._decode_executor = ThreadPoolExecutor(max_workers=worker_count)
    return backend


def test_worker_selection_counts_persistent_assignments() -> None:
    backend = _coordinator(4)
    try:
        assert [backend._pick_worker() for _ in range(8)] == [0, 1, 2, 3] * 2
        assert backend._assigned == [2, 2, 2, 2]
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
