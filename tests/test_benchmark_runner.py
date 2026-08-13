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
    )
    loop.close()
    assert summary.requests == summary.succeeded == 6
    assert summary.failed == 0
    assert all(result.prompt_tokens == 5 for result in summary.results)
    assert all(result.completion_tokens == 3 for result in summary.results)
    assert set(summary.ttft_ms) == {"p50", "p95", "p99"}
    assert set(summary.tpot_ms) == {"p50", "p95", "p99"}
    assert summary.output_token_throughput > 0
    assert summary.to_dict()["results"][0]["sample_id"] == "0"
    assert backend.live == set()


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
