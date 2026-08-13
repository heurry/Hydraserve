from __future__ import annotations

from hydraserve.benchmark import BenchmarkSample, run_benchmark
from hydraserve.engine import ContinuousGenerationLoop


class Tokenizer:
    def encode(self, text):
        return tuple(ord(character) for character in text)


class Backend:
    def __init__(self):
        self.live = set()

    def prefill(self, request):
        self.live.add(request.request_id)
        return request.token_ids[-1] + 1

    def decode(self, requests):
        return tuple(request.generated_token_ids[-1] + 1 for request in requests)

    def release(self, request_id):
        self.live.remove(request_id)


def test_benchmark_collects_concurrent_latency_and_throughput() -> None:
    backend = Backend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=4)
    samples = [
        BenchmarkSample("toy", str(index), f"prompt {index}") for index in range(6)
    ]
    summary = run_benchmark(
        loop,
        Tokenizer(),
        samples,
        max_new_tokens=3,
        concurrency=3,
        max_prompt_tokens=5,
        warmup_requests=1,
    )
    loop.close()
    assert summary.requests == summary.succeeded == 5
    assert summary.failed == 0
    assert all(result.prompt_tokens == 5 for result in summary.results)
    assert all(result.completion_tokens == 3 for result in summary.results)
    assert set(summary.ttft_ms) == {"p50", "p95", "p99"}
    assert set(summary.tpot_ms) == {"p50", "p95", "p99"}
    assert summary.output_token_throughput > 0
    assert summary.route_counts == {"collocated": 5}
    assert summary.warmup_requests == 1
    assert summary.to_dict()["results"][0]["sample_id"] == "1"
    assert "route_estimated_savings_ms" in summary.to_dict()["results"][0]
    assert "route_decode_load" in summary.to_dict()["results"][0]
    assert "route_prefill_queue_ahead_ms" in summary.to_dict()["results"][0]
    assert "route_observed_prefill_service_ms" in summary.to_dict()["results"][0]
    assert "admission_wait_ms" in summary.to_dict()["results"][0]
    assert "observed_prefill_queue_wait_ms" in summary.to_dict()["results"][0]
    assert "request_id" in summary.to_dict()["results"][0]
    assert backend.live == set()


def test_benchmark_resets_online_router_state_after_warmup() -> None:
    class CalibratedBackend(Backend):
        def __init__(self):
            super().__init__()
            self.reset_count = 0

        def reset_routing_calibration(self):
            self.reset_count += 1

    backend = CalibratedBackend()
    loop = ContinuousGenerationLoop(backend)
    run_benchmark(
        loop,
        Tokenizer(),
        [BenchmarkSample("toy", str(index), "x") for index in range(2)],
        max_new_tokens=1,
        warmup_requests=1,
    )
    loop.close()
    assert backend.reset_count == 1


def test_benchmark_records_request_error() -> None:
    class FailingBackend(Backend):
        def prefill(self, request):
            raise RuntimeError("cannot prefill")

        def release(self, request_id):
            return

    loop = ContinuousGenerationLoop(FailingBackend())
    summary = run_benchmark(
        loop, Tokenizer(), [BenchmarkSample("toy", "bad", "x")]
    )
    loop.close()
    assert summary.succeeded == 0
    assert summary.failed == 1
    assert summary.ttft_ms == {}
    assert summary.results[0].error == "cannot prefill"
    assert summary.route_counts == {}


def test_fixed_and_seeded_poisson_arrival_configuration() -> None:
    samples = [BenchmarkSample("toy", str(index), "x") for index in range(3)]
    for pattern in ("fixed", "poisson"):
        loop = ContinuousGenerationLoop(Backend())
        summary = run_benchmark(
            loop,
            Tokenizer(),
            samples,
            max_new_tokens=1,
            concurrency=2,
            request_rate=1000,
            arrival_pattern=pattern,
            seed=7,
        )
        loop.close()
        assert summary.succeeded == 3
        assert summary.arrival_pattern == pattern
        assert summary.offered_request_rate == 1000


def test_arrival_configuration_validation() -> None:
    import pytest

    loop = ContinuousGenerationLoop(Backend())
    with pytest.raises(ValueError, match="require request_rate"):
        run_benchmark(
            loop,
            Tokenizer(),
            [],
            arrival_pattern="fixed",
        )
    loop.close()
