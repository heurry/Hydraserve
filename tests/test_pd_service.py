from __future__ import annotations

import pytest

from queue import Queue
from threading import Lock
from types import SimpleNamespace

from hydraserve.engine import (
    AdmissionDecision,
    DisaggregatedGenerationBackend,
    PDWorkerConfig,
    ServingRequest,
)


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
