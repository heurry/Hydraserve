"""P0 harness tests: JSONL trace adapter, by-class metrics, SLO goodput."""

from __future__ import annotations

import json

from hydraserve.benchmark.datasets import (
    BenchmarkSample,
    TraceSpec,
    iter_trace,
    write_trace,
)
from hydraserve.benchmark.runner import (
    RequestMetrics,
    run_benchmark,
    _by_class_stats,
    _slo_stats,
)
from hydraserve.engine import ContinuousGenerationLoop


class Tokenizer:
    """Deterministic 1-char-per-token tokenizer so trace lengths are exact."""

    base_vocab_size = 26

    def encode(self, text):
        return tuple(ord(c) for c in text)

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


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


def test_trace_roundtrip(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    entries = [
        TraceSpec(id="short-0", klass="short", prompt_tokens=8, max_new_tokens=4, arrival_offset_ms=500, ignore_eos=True),
        TraceSpec(id="long-0", klass="long", prompt_tokens=32, max_new_tokens=2, seed=7),
    ]
    meta = write_trace(Tokenizer(), entries, path, seed=42)
    assert meta["entries"] == 2
    assert meta["trace_sha256"]

    samples = list(iter_trace(Tokenizer(), path, seed=42))
    assert len(samples) == 2
    first = samples[0]
    assert first.klass in {"short", "long"}
    assert first.max_new_tokens in {4, 2}
    # re-encode is exact because the tokenizer is 1 char == 1 token
    assert first.metadata["reencode_tokens"] == first.metadata["target_tokens"]
    assert first.metadata["trace_seed"] == 42


def test_trace_written_prompts_are_reproducible(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    write_trace(Tokenizer(), [TraceSpec(id="a", klass="short", prompt_tokens=10, max_new_tokens=4)], path, seed=5)
    with path.open() as fh:
        line = json.loads(fh.readline())
    assert len(line["prompt"]) == 10  # 1 char per token
    assert line["prompt_sha256"]


def test_by_class_and_slo_stats() -> None:
    def metric(klass, ttft, tpot, tokens, target, error=None):
        return RequestMetrics(
            sample_id="x", request_id=1, prompt_tokens=8, completion_tokens=tokens,
            ttft_ms=ttft, tpot_ms=tpot, latency_ms=0.0, finish_reason="stop",
            error=error, klass=klass, target_new_tokens=target,
        )

    results = [
        # short, SLO met
        metric("short", ttft=1000, tpot=50, tokens=128, target=128),
        # short, SLO failed (TTFT too slow)
        metric("short", ttft=9000, tpot=50, tokens=128, target=128),
        # short, SLO failed (error; no ttft/tpot produced)
        metric("short", ttft=None, tpot=None, tokens=10, target=128, error="boom"),
        # long, out of SLO scope
        metric("long", ttft=2000, tpot=500, tokens=16, target=16),
    ]
    by_class = _by_class_stats(results, wall_time=10.0)
    assert by_class["short"]["requests"] == 3
    assert by_class["short"]["succeeded"] == 2
    assert by_class["long"]["requests"] == 1
    assert "ttft_ms" in by_class["short"]

    slo = _slo_stats(results, wall_time=10.0)
    assert slo["short_requests"] == 3
    assert slo["met_requests"] == 1  # only the first short met every gate
    assert slo["met_rate"] == round(1 / 3, 4)
    assert slo["ttft_ok"] == 1  # only #1 (1000ms) within 5s; #2 is 9000ms
    assert slo["tpot_ok"] == 2  # #1 and #2 both 50ms
    assert slo["no_error"] == 2
    assert slo["goodput_tokens_s"] == 128 / 10.0


def test_run_benchmark_uses_per_sample_max_new_tokens() -> None:
    loop = ContinuousGenerationLoop(Backend(), max_batch_size=4)
    samples = [
        BenchmarkSample("trace", "short-0", "a" * 4, klass="short", max_new_tokens=3, ignore_eos=True),
        BenchmarkSample("trace", "long-0", "b" * 4, klass="long", max_new_tokens=5),
    ]
    summary = run_benchmark(
        loop, Tokenizer(), samples, max_new_tokens=2, concurrency=2
    )
    loop.close()
    assert summary.succeeded == 2
    assert summary.by_class["short"]["requests"] == 1
    assert summary.by_class["long"]["requests"] == 1
    short = [r for r in summary.results if r.klass == "short"][0]
    assert short.target_new_tokens == 3  # per-sample override, not global 2
    assert short.completion_tokens == 3
    long = [r for r in summary.results if r.klass == "long"][0]
    assert long.target_new_tokens == 5
    assert long.completion_tokens == 5
