from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.engine import (
    AdaptiveGenerationBackend,
    ContinuousGenerationLoop,
    DisaggregatedGenerationBackend,
    MultiWorkerGenerationBackend,
    PDClusterConfig,
    PDWorkerConfig,
)
from hydraserve.router import AdaptiveRouter, Route, RouterConfig


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


def test_real_adaptive_service_executes_both_routes() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = AdaptiveGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=128,
            use_flash_attention=False,
        ),
        router=AdaptiveRouter(
            RouterConfig(
                short_prompt_tokens=4,
                long_prompt_tokens=8,
                force_pd_tokens=16,
            )
        ),
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    short = loop.submit([1, 42, 17], max_new_tokens=2)
    long = loop.submit([1, 12, 23, 4, 5, 6, 7, 8, 9, 10], max_new_tokens=2)
    short_events = list(short)
    long_events = list(long)
    assert short.request.route == Route.COLLOCATED.value
    assert long.request.route == Route.PD_DISAGGREGATED.value
    stats = backend.routing_stats()
    loop.close()
    assert short_events[-1].finish_reason == "length"
    assert long_events[-1].finish_reason == "length"
    assert stats.collocated == 1
    assert stats.pd_disaggregated == 1


def test_real_cluster_backend_single_decode_worker_vertical_slice() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = MultiWorkerGenerationBackend(
        PDClusterConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            ("cuda:1",),
            cache_tokens_per_worker=128,
            use_flash_attention=False,
        ),
        router=AdaptiveRouter(
            RouterConfig(
                short_prompt_tokens=4,
                long_prompt_tokens=8,
                force_pd_tokens=16,
            )
        ),
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    short = loop.submit([1, 2, 3], max_new_tokens=2)
    long = loop.submit([1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=2)
    assert list(short)[-1].finish_reason == "length"
    assert list(long)[-1].finish_reason == "length"
    assert short.request.route == Route.COLLOCATED.value
    assert long.request.route == Route.PD_DISAGGREGATED.value
    assert short.request.worker_id == 0
    loop.close()
