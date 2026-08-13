from __future__ import annotations

import pytest

from hydraserve.engine import DisaggregatedGenerationBackend, PDWorkerConfig


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
