from __future__ import annotations

import pytest

from queue import Queue
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace

from hydraserve.engine import (
    AdmissionDecision,
    DisaggregatedGenerationBackend,
    PDWorkerConfig,
    ServingRequest,
)
from hydraserve.engine.pd_service import _DeferredCudaCacheFree


def test_deferred_cuda_cache_free_waits_for_event_before_reuse() -> None:
    class FakeEvent:
        def __init__(self):
            self.ready = False
            self.recorded = False
            self.synchronized = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.ready

        def synchronize(self):
            self.synchronized = True
            self.ready = True

    class FakeCache:
        device = SimpleNamespace(type="cuda")

        def __init__(self):
            self.freed = []

        def free(self, request_id):
            self.freed.append(request_id)

    events: list[FakeEvent] = []

    def make_event():
        event = FakeEvent()
        events.append(event)
        return event

    cache = FakeCache()
    freer = _DeferredCudaCacheFree(cache, event_factory=make_event)

    freer.free(7)
    assert events[0].recorded
    assert cache.freed == []

    freer.collect()
    assert cache.freed == []

    events[0].ready = True
    freer.collect()
    assert cache.freed == [7]

    freer.free(8)
    freer.collect(blocking=True)
    assert events[1].synchronized
    assert cache.freed == [7, 8]


def test_pd_worker_config_rejects_same_device_without_spawning() -> None:
    with pytest.raises(ValueError, match="distinct"):
        DisaggregatedGenerationBackend(
            PDWorkerConfig("unused", prefill_device="cuda:0", decode_device="cuda:0")
        )


def test_pd_worker_config_rejects_invalid_cache_without_spawning() -> None:
    with pytest.raises(ValueError, match="cache limits"):
        DisaggregatedGenerationBackend(PDWorkerConfig("unused", cache_tokens=0))


def test_pd_worker_config_rejects_invalid_state_capacity_without_spawning() -> None:
    with pytest.raises(ValueError, match="cache limits"):
        DisaggregatedGenerationBackend(PDWorkerConfig("unused", max_state_slots=0))


def test_pd_worker_config_rejects_invalid_decode_workspace_without_spawning() -> None:
    with pytest.raises(ValueError, match="cache limits"):
        DisaggregatedGenerationBackend(
            PDWorkerConfig("unused", max_decode_batch_size=0)
        )


def test_pd_recovery_rpc_carries_exact_history_without_prefill_sampling() -> None:
    class FakePDBackend(DisaggregatedGenerationBackend):
        def __init__(self):
            self._decode_lock = Lock()
            self._decode_commands = Queue()
            self._decode_responses = Queue()
            self._decode_responses.put({"op": "recover", "request_id": 4})
            self.operation_timeout = 1.0
            self.config = SimpleNamespace(block_size=4)
            self.released = []

        def _reserve_decode(self, request, *, force_rpc=False):
            return AdmissionDecision.accept()

        def _update_capacity(self, result):
            return None

        def _decode_rpc(self, command, expected_op, request_id=None):
            self._decode_commands.put(command)
            result = self._decode_responses.get(timeout=1)
            self._check(result, expected_op, request_id)
            return result

        def release(self, request_id):
            self.released.append(request_id)

    backend = FakePDBackend()
    request = ServingRequest(4, (1, 2), 5, generated_token_ids=[8, 9, 10])
    decision = backend.recover(request)
    command = backend._decode_commands.get_nowait()

    assert decision.admitted
    assert command["op"] == "recover"
    assert command["token_ids"] == (1, 2)
    assert command["generated_token_ids"] == (8, 9, 10)
    assert command["replay_token_ids"] == (1, 2, 8, 9)
    assert backend.released == []


def test_fixed_pd_prepare_rpc_waits_for_final_response_after_receiver_armed() -> None:
    class AliveProcess:
        @staticmethod
        def is_alive():
            return True

    backend = object.__new__(DisaggregatedGenerationBackend)
    backend._decode_lock = Lock()
    backend._decode = AliveProcess()
    backend._decode_commands = Queue()
    backend._decode_responses = Queue()
    backend.operation_timeout = 2.0
    receiver_armed = Event()
    results = {}

    thread = Thread(
        target=lambda: results.setdefault(
            "prepare",
            backend._decode_rpc(
                {"op": "prepare", "request_id": 7},
                "prepare",
                7,
                receiver_armed=receiver_armed,
            ),
        )
    )
    thread.start()
    backend._decode_commands.get(timeout=1)
    backend._decode_responses.put({"op": "prepare_armed", "request_id": 7})

    assert receiver_armed.wait(1)
    assert thread.is_alive()

    backend._decode_responses.put(
        {"op": "prepare", "request_id": 7, "token_id": 11}
    )
    thread.join(1)
    assert not thread.is_alive()
    assert results["prepare"]["token_id"] == 11


def test_fixed_pd_detects_dead_prefill_before_waiting_for_rpc_timeout() -> None:
    class DeadProcess:
        def is_alive(self):
            return False

    backend = object.__new__(DisaggregatedGenerationBackend)
    backend._recovery_lock = RLock()
    backend._closed = False
    backend._prefill_healthy = True
    backend._prefill = DeadProcess()
    scheduled = []
    backend._schedule_recovery = scheduled.append

    assert not backend._prefill_available()
    assert not backend._prefill_healthy
    assert scheduled == ["prefill"]


def test_fixed_pd_worker_recovery_retries_with_backoff() -> None:
    class RecoveringBackend:
        max_worker_restarts = 3
        worker_restart_backoff_s = 0

        def __init__(self):
            self._recovery_stop = Event()
            self._recovery_lock = RLock()
            self._prefill_healthy = False
            self._prefill_recovering = True
            self._prefill_recovery_thread = None
            self._prefill_recovery_attempts = 0
            self._prefill_recovery_successes = 0
            self._prefill_recovery_failures = 0
            self.calls = 0

        def _restart_worker_once(self, kind):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("startup failed")

    backend = RecoveringBackend()
    DisaggregatedGenerationBackend._recover_worker(backend, "prefill")

    assert backend.calls == 2
    assert backend._prefill_healthy
    assert backend._prefill_recovery_attempts == 2
    assert backend._prefill_recovery_successes == 1
    assert backend._prefill_recovery_failures == 1
    assert not backend._prefill_recovering
