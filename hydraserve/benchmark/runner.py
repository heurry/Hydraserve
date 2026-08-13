"""Tokenizer-aware benchmark execution and latency aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from math import ceil
from time import perf_counter
from typing import Iterable

from hydraserve.benchmark.datasets import BenchmarkSample


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    sample_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    finish_reason: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    requests: int
    succeeded: int
    failed: int
    wall_time_s: float
    request_throughput: float
    output_token_throughput: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    latency_ms: dict[str, float]
    results: tuple[RequestMetrics, ...]

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "wall_time_s": self.wall_time_s,
            "request_throughput": self.request_throughput,
            "output_token_throughput": self.output_token_throughput,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "latency_ms": self.latency_ms,
            "results": [asdict(result) for result in self.results],
        }


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def value(percentile: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {name: value(percentile) for name, percentile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}


def run_benchmark(
    generation_loop,
    tokenizer,
    samples: Iterable[BenchmarkSample],
    *,
    max_new_tokens: int = 32,
    concurrency: int = 1,
    max_prompt_tokens: int | None = None,
) -> BenchmarkSummary:
    if max_new_tokens <= 0 or concurrency <= 0:
        raise ValueError("max_new_tokens and concurrency must be positive")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    indexed = tuple(enumerate(samples))

    def run_one(item) -> tuple[int, RequestMetrics]:
        index, sample = item
        token_ids = tokenizer.encode(sample.prompt)
        if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[-max_prompt_tokens:]
        started = perf_counter()
        first_token_at = None
        completion_tokens = 0
        finish_reason = "error"
        error = None
        try:
            handle = generation_loop.submit(token_ids, max_new_tokens)
            for event in handle:
                now = perf_counter()
                if event.token_id is not None:
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = now
                if event.finished:
                    finish_reason = event.finish_reason or "unknown"
                    error = event.error
        except Exception as exc:
            error = str(exc)
        ended = perf_counter()
        latency_ms = (ended - started) * 1000
        ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000
        tpot_ms = None
        if first_token_at is not None and completion_tokens > 1:
            tpot_ms = (ended - first_token_at) * 1000 / (completion_tokens - 1)
        return index, RequestMetrics(
            sample.sample_id,
            len(token_ids),
            completion_tokens,
            ttft_ms,
            tpot_ms,
            latency_ms,
            finish_reason,
            error,
        )

    wall_started = perf_counter()
    ordered_results: list[RequestMetrics | None] = [None] * len(indexed)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one, item) for item in indexed]
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result
    wall_time = perf_counter() - wall_started
    results = tuple(result for result in ordered_results if result is not None)
    succeeded = tuple(result for result in results if result.error is None)
    divisor = wall_time if wall_time > 0 else float("inf")
    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / divisor,
        output_token_throughput=sum(result.completion_tokens for result in succeeded) / divisor,
        ttft_ms=_percentiles(
            result.ttft_ms for result in succeeded if result.ttft_ms is not None
        ),
        tpot_ms=_percentiles(
            result.tpot_ms for result in succeeded if result.tpot_ms is not None
        ),
        latency_ms=_percentiles(result.latency_ms for result in succeeded),
        results=results,
    )
