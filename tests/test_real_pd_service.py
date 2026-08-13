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
    SamplingParams,
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


def test_real_pd_preemption_recovery_matches_uninterrupted_generation() -> None:
    from threading import Event

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    inner = DisaggregatedGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=128,
            max_state_slots=1,
            use_flash_attention=False,
        )
    )
    decode_entered = Event()
    allow_decode = Event()

    class BlockingBackend:
        supports_async_prefill = True

        def __init__(self):
            self.blocked = False

        def __getattr__(self, name):
            return getattr(inner, name)

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(30)
            return inner.decode(requests)

    loop = ContinuousGenerationLoop(BlockingBackend(), max_batch_size=1)
    prompt = [1, 42, 17, 9]
    sampling = SamplingParams(temperature=0.8, top_k=16, seed=20260814)
    try:
        background = loop.submit(
            prompt, max_new_tokens=3, sampling_params=sampling
        )
        first = background.get(timeout=120)
        assert first.token_id is not None
        assert decode_entered.wait(120)
        urgent = loop.submit([1, 7], max_new_tokens=1, priority=7)
        allow_decode.set()
        urgent_events = list(urgent)
        background_events = [first, *list(background)]
        reference_events = list(
            loop.submit(prompt, max_new_tokens=3, sampling_params=sampling)
        )
    finally:
        loop.close(timeout=120)

    generated = [event.token_id for event in background_events if event.token_id is not None]
    reference = [event.token_id for event in reference_events if event.token_id is not None]
    assert generated == reference
    assert urgent_events[-1].finish_reason == "length"
    assert background_events[-1].finish_reason == "length"
    assert loop.preemptions_total == 1
    assert loop.recoveries_total == 1
    stats = inner.cache_stats()
    assert stats["active_allocations"] == 0
    assert stats["physical_free_blocks"] == stats["physical_total_blocks"]


def test_real_multi_worker_coordinator_recovers_preempted_request() -> None:
    from threading import Event

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    inner = MultiWorkerGenerationBackend(
        PDClusterConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            ("cuda:1",),
            prefill_device="cuda:0",
            cache_tokens_per_worker=128,
            max_state_slots_per_worker=1,
            use_flash_attention=False,
        ),
        router=AdaptiveRouter(
            RouterConfig(short_prompt_tokens=32, long_prompt_tokens=64)
        ),
    )
    decode_entered = Event()
    allow_decode = Event()

    class BlockingBackend:
        supports_async_prefill = True

        def __init__(self):
            self.blocked = False

        def __getattr__(self, name):
            return getattr(inner, name)

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(30)
            return inner.decode(requests)

    loop = ContinuousGenerationLoop(BlockingBackend(), max_batch_size=1)
    prompt = [1, 42, 17, 9]
    try:
        background = loop.submit(prompt, max_new_tokens=3)
        first = background.get(timeout=120)
        assert first.token_id is not None
        assert decode_entered.wait(120)
        urgent = loop.submit([1, 7], max_new_tokens=1, priority=7)
        allow_decode.set()
        urgent_events = list(urgent)
        background_events = [first, *list(background)]
        reference_events = list(loop.submit(prompt, max_new_tokens=3))
    finally:
        loop.close(timeout=120)

    generated = [event.token_id for event in background_events if event.token_id is not None]
    reference = [event.token_id for event in reference_events if event.token_id is not None]
    assert generated == reference
    assert urgent_events[-1].finish_reason == "length"
    assert loop.preemptions_total == 1
    assert loop.recoveries_total == 1
    stats = inner.cache_stats()
    assert stats["active_allocations"] == 0
    assert stats["physical_free_blocks"] == stats["physical_total_blocks"]


def test_real_pd_kv_headroom_stats_and_release_are_end_to_end() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = DisaggregatedGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=128,
            block_size=16,
            max_state_slots=2,
            kv_headroom_blocks=2,
            use_flash_attention=False,
        )
    )
    loop = ContinuousGenerationLoop(
        backend, max_batch_size=2, max_active_requests=2
    )
    try:
        handles = [
            loop.submit([1, 42, 17, 9 + index], max_new_tokens=2)
            for index in range(2)
        ]
        assert all(list(handle)[-1].finish_reason == "length" for handle in handles)
        stats = backend.cache_stats()
        assert stats["physical_total_blocks"] == 8
        assert stats["physical_free_blocks"] == 8
        assert stats["usable_total_blocks"] == 6
        assert stats["allocatable_free_blocks"] == 6
        assert stats["headroom_blocks"] == 2
        assert stats["active_allocations"] == 0
        assert stats["total_references"] == 0
        assert stats["high_watermark_blocks"] > 0
    finally:
        loop.close()


def test_real_pd_seeded_sampling_and_logprobs_are_reproducible() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = DisaggregatedGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=128,
            use_flash_attention=False,
        )
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    sampling = SamplingParams(
        temperature=0.8,
        top_k=16,
        seed=12345,
        logprobs=3,
    )
    try:
        first = list(
            loop.submit([1, 42, 17, 9], max_new_tokens=3, sampling_params=sampling)
        )
        second = list(
            loop.submit([1, 42, 17, 9], max_new_tokens=3, sampling_params=sampling)
        )
        first_tokens = [event.token_id for event in first if event.token_id is not None]
        second_tokens = [event.token_id for event in second if event.token_id is not None]
        assert first_tokens == second_tokens
        assert len(first_tokens) == 3
        for event in first[:-1]:
            assert event.logprob is not None
            assert len(event.top_logprobs) == 3
    finally:
        loop.close()


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


def test_real_cluster_recovers_a_crashed_decode_worker() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = MultiWorkerGenerationBackend(
        PDClusterConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            ("cuda:1",),
            cache_tokens_per_worker=64,
            use_flash_attention=False,
        ),
        startup_timeout=180,
        operation_timeout=30,
        worker_restart_backoff_s=0.1,
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    try:
        crashed = backend._decode_processes[0]
        crashed.terminate()
        crashed.join(10)
        assert not crashed.is_alive()

        handle = loop.submit([1, 42, 17], max_new_tokens=1)
        events = []
        while True:
            event = handle.get(timeout=180)
            events.append(event)
            if event.finished:
                break
        assert events[-1].finish_reason == "length"
        stats = backend.recovery_stats()
        assert stats.successes == 1
        assert stats.healthy_workers == 1
    finally:
        loop.close()


def test_real_active_request_survives_decode_worker_crash_and_rebinds() -> None:
    from threading import Event

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    inner = MultiWorkerGenerationBackend(
        PDClusterConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            ("cuda:1",),
            cache_tokens_per_worker=128,
            max_state_slots_per_worker=1,
            use_flash_attention=False,
        ),
        startup_timeout=180,
        operation_timeout=30,
        worker_restart_backoff_s=0.1,
    )
    decode_entered = Event()
    allow_decode = Event()

    class CrashBoundaryBackend:
        supports_async_prefill = True

        def __init__(self):
            self.blocked = False

        def __getattr__(self, name):
            return getattr(inner, name)

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(30)
            return inner.decode(requests)

    loop = ContinuousGenerationLoop(CrashBoundaryBackend(), max_batch_size=1)
    prompt = [1, 42, 17, 9]
    sampling = SamplingParams(temperature=0.8, top_k=16, seed=314159)
    try:
        handle = loop.submit(prompt, max_new_tokens=3, sampling_params=sampling)
        first = handle.get(timeout=120)
        assert first.token_id is not None
        assert decode_entered.wait(120)
        crashed = inner._decode_processes[0]
        crashed.terminate()
        crashed.join(10)
        assert not crashed.is_alive()
        allow_decode.set()

        recovered_events = [first, *list(handle)]
        reference_events = list(
            loop.submit(prompt, max_new_tokens=3, sampling_params=sampling)
        )
        stats = inner.recovery_stats()
    finally:
        loop.close(timeout=120)

    recovered = [event.token_id for event in recovered_events if event.token_id is not None]
    reference = [event.token_id for event in reference_events if event.token_id is not None]
    assert recovered == reference
    assert recovered_events[-1].finish_reason == "length"
    assert loop.fault_suspensions_total == 1
    assert loop.recoveries_total == 1
    assert stats.successes == 1
    assert stats.healthy_workers == 1


def test_real_prefill_worker_crash_fails_closed_then_restores_pd_route() -> None:
    from time import monotonic, sleep

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = MultiWorkerGenerationBackend(
        PDClusterConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            ("cuda:1",),
            prefill_device="cuda:0",
            cache_tokens_per_worker=128,
            max_state_slots_per_worker=1,
            use_flash_attention=False,
        ),
        router=AdaptiveRouter(
            RouterConfig(
                short_prompt_tokens=4,
                long_prompt_tokens=8,
                force_pd_tokens=16,
            )
        ),
        startup_timeout=180,
        operation_timeout=30,
        worker_restart_backoff_s=0.1,
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    long_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        crashed = backend._prefill
        crashed.terminate()
        crashed.join(10)
        assert not crashed.is_alive()

        degraded = loop.submit(long_prompt, max_new_tokens=1)
        degraded_events = list(degraded)
        assert degraded.request.route == Route.COLLOCATED.value
        assert degraded.request.route_reason == "prefill_unavailable"
        assert degraded_events[-1].finish_reason == "length"

        deadline = monotonic() + 180
        while not backend.prefill_recovery_stats().healthy:
            assert monotonic() < deadline, "prefill worker did not recover"
            sleep(0.1)
        restored = loop.submit(long_prompt, max_new_tokens=1)
        restored_events = list(restored)
        stats = backend.prefill_recovery_stats()
    finally:
        loop.close(timeout=120)

    assert restored.request.route == Route.PD_DISAGGREGATED.value
    assert restored_events[-1].finish_reason == "length"
    assert stats.attempts == 1
    assert stats.successes == 1
    assert stats.failures == 0
    assert stats.healthy


def test_real_decode_worker_retains_and_reuses_full_attention_prefix_pages() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    backend = AdaptiveGenerationBackend(
        PDWorkerConfig(
            "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
            cache_tokens=64,
            block_size=4,
            prefix_cache_blocks=4,
            prefix_cache_min_frequency=1,
            use_flash_attention=False,
        ),
        router=AdaptiveRouter(
            RouterConfig(short_prompt_tokens=16, long_prompt_tokens=32, force_pd_tokens=64)
        ),
    )
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    prompt = [1, 42, 17, 9, 5, 6, 7, 8]
    first = loop.submit(prompt, max_new_tokens=1)
    first_tokens = [event.token_id for event in list(first) if event.token_id is not None]
    assert backend.prefix_match_tokens(prompt) == 8
    second = loop.submit(prompt, max_new_tokens=1)
    second_tokens = [event.token_id for event in list(second) if event.token_id is not None]
    assert first_tokens == second_tokens
    assert backend.prefix_match_tokens(prompt) == 8
    loop.close()
