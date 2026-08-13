from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.engine import (
    ContinuousGenerationLoop,
    DisaggregatedGenerationBackend,
    PDWorkerConfig,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_PD_SERVICE") != "1",
        reason="set HYDRASERVE_RUN_PD_SERVICE=1 to load one 4B model per GPU",
    ),
]


def test_persistent_real_two_gpu_pd_generation() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = DisaggregatedGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=128,
            use_flash_attention=False,
        )
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    first = loop.submit([1, 42, 17, 9], max_new_tokens=2)
    second = loop.submit([1, 12, 23], max_new_tokens=2)
    first_events = list(first)
    second_events = list(second)
    loop.close()
    assert [event.token_id for event in first_events[:-1]]
    assert [event.token_id for event in second_events[:-1]]
    assert first_events[-1].finish_reason == "length"
    assert second_events[-1].finish_reason == "length"
