from __future__ import annotations

from time import sleep

import pytest

from hydraserve.benchmark import BenchmarkSample, run_benchmark
from hydraserve.engine import AdmissionDecision, ContinuousGenerationLoop


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


def test_benchmark_marks_short_trace_records_as_higher_priority() -> None:
    class PriorityBackend(Backend):
        def __init__(self):
            super().__init__()
            self.priorities: dict[tuple[int, ...], int] = {}

        def prefill(self, request):
            self.priorities[tuple(request.token_ids)] = request.priority
            return super().prefill(request)

    backend = PriorityBackend()
    loop = ContinuousGenerationLoop(backend)
    run_benchmark(
        loop,
        Tokenizer(),
        [
            BenchmarkSample("toy", "s", "s", klass="short"),
            BenchmarkSample("toy", "l", "l", klass="long"),
            BenchmarkSample("toy", "d", "d"),
        ],
        max_new_tokens=1,
    )
    loop.close()

    assert backend.priorities[tuple(Tokenizer().encode("s"))] == 1
    assert backend.priorities[tuple(Tokenizer().encode("l"))] == 0
    assert backend.priorities[tuple(Tokenizer().encode("d"))] == 0


def test_benchmark_records_request_error() -> None:
    class FailingBackend(Backend):
        def prefill(self, request):
            raise RuntimeError("cannot prefill")

        def release(self, request_id):
            return

    loop = ContinuousGenerationLoop(FailingBackend())
    summary = run_benchmark(loop, Tokenizer(), [BenchmarkSample("toy", "bad", "x")])
    loop.close()
    assert summary.succeeded == 0
    assert summary.failed == 1
    assert summary.ttft_ms == {}
    assert summary.results[0].error == "cannot prefill"
    assert summary.route_counts == {}
    assert summary.route_failure_counts == {"collocated": 1}


def test_benchmark_keeps_pre_admission_failure_route_unknown() -> None:
    class RejectingBackend(Backend):
        def admit(self, request):
            return AdmissionDecision.reject("request exceeds KV capacity")

    loop = ContinuousGenerationLoop(RejectingBackend())
    summary = run_benchmark(
        loop, Tokenizer(), [BenchmarkSample("toy", "rejected", "x")]
    )
    loop.close()

    assert summary.failed == 1
    assert summary.results[0].route is None
    assert summary.route_failure_counts == {"unknown": 1}


def test_tpot_excludes_release_tail_and_records_decode_batch_size() -> None:
    class SlowReleaseBackend(Backend):
        def release(self, request_id):
            sleep(0.03)
            super().release(request_id)

    loop = ContinuousGenerationLoop(SlowReleaseBackend(), max_batch_size=4)
    summary = run_benchmark(
        loop,
        Tokenizer(),
        [BenchmarkSample("toy", "timing", "x")],
        max_new_tokens=3,
    )
    loop.close()

    result = summary.results[0]
    assert len(result.itl_ms) == 2
    assert result.decode_batch_sizes == (1, 1)
    assert result.tpot_ms == pytest.approx(sum(result.itl_ms) / 2)
    assert result.release_tail_ms is not None
    assert result.release_tail_ms >= 20
    assert summary.release_tail_ms["p50"] == result.release_tail_ms
    assert summary.itl_ms


def test_benchmark_reports_client_queue_and_arrival_based_ttft() -> None:
    class SlowPrefillBackend(Backend):
        def prefill(self, request):
            sleep(0.03)
            return super().prefill(request)

    loop = ContinuousGenerationLoop(SlowPrefillBackend())
    summary = run_benchmark(
        loop,
        Tokenizer(),
        [
            BenchmarkSample("toy", "first", "x"),
            BenchmarkSample("toy", "second", "y"),
        ],
        max_new_tokens=1,
        concurrency=1,
        arrival_pattern="burst",
        closed_loop_clients=True,
    )
    loop.close()

    second = summary.results[1]
    assert second.client_queue_ms >= 20
    assert second.e2e_ttft_ms is not None
    assert second.ttft_ms is not None
    assert second.e2e_ttft_ms > second.ttft_ms
    assert summary.client_queue_ms["p99"] >= second.client_queue_ms * 0.9


def test_open_loop_benchmark_dispatch_does_not_queue_behind_slow_request() -> None:
    class SlowFirstPrefillBackend(Backend):
        def prefill(self, request):
            if request.token_ids[-1] == ord("x"):
                sleep(0.03)
            return super().prefill(request)

    loop = ContinuousGenerationLoop(SlowFirstPrefillBackend(), max_batch_size=1)
    summary = run_benchmark(
        loop,
        Tokenizer(),
        [
            BenchmarkSample("toy", "first", "x"),
            BenchmarkSample("toy", "second", "y"),
        ],
        max_new_tokens=1,
        concurrency=1,
        arrival_pattern="burst",
    )
    loop.close()

    second = summary.results[1]
    assert second.client_queue_ms < 20
    assert second.e2e_ttft_ms is not None
    assert second.ttft_ms is not None
    assert second.e2e_ttft_ms == pytest.approx(
        second.ttft_ms + second.client_queue_ms,
        abs=5,
    )


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
    loop = ContinuousGenerationLoop(Backend())
    with pytest.raises(ValueError, match="require request_rate"):
        run_benchmark(
            loop,
            Tokenizer(),
            [],
            arrival_pattern="fixed",
        )
    loop.close()
