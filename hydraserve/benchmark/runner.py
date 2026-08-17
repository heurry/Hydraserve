"""Tokenizer-aware benchmark execution and latency aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from math import ceil
from random import Random
from time import perf_counter, sleep
from typing import Iterable
from urllib import error, request

from hydraserve.benchmark.datasets import BenchmarkSample


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    sample_id: str
    request_id: int | None
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    finish_reason: str
    error: str | None = None
    route: str | None = None
    route_reason: str | None = None
    worker_id: int | None = None
    route_collocated_cost_ms: float | None = None
    route_pd_cost_ms: float | None = None
    route_estimated_savings_ms: float | None = None
    route_cost_confidence: float | None = None
    route_decode_load: float | None = None
    route_prefill_load: float | None = None
    route_prefill_queue_ahead_ms: float = 0.0
    route_observed_prefill_service_ms: float | None = None
    admission_wait_ms: float | None = None
    observed_prefill_queue_wait_ms: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    requests: int
    succeeded: int
    failed: int
    wall_time_s: float
    request_throughput: float
    output_token_throughput: float
    warmup_requests: int
    offered_request_rate: float | None
    arrival_pattern: str
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    latency_ms: dict[str, float]
    route_counts: dict[str, int]
    prefix_cache_stats: dict[str, int | float]
    per_worker: dict[str, dict[str, int | float]]
    results: tuple[RequestMetrics, ...]

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "wall_time_s": self.wall_time_s,
            "request_throughput": self.request_throughput,
            "output_token_throughput": self.output_token_throughput,
            "warmup_requests": self.warmup_requests,
            "offered_request_rate": self.offered_request_rate,
            "arrival_pattern": self.arrival_pattern,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "latency_ms": self.latency_ms,
            "route_counts": self.route_counts,
            "prefix_cache_stats": self.prefix_cache_stats,
            "per_worker": self.per_worker,
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


def _prefix_cache_stats(generation_loop) -> dict[str, int | float]:
    """Collect prefix-cache hit/miss counters from the serving backend."""
    backend = getattr(generation_loop, "backend", None)
    stats_fn = getattr(backend, "cache_stats", None)
    if stats_fn is None:
        return {}
    try:
        stats = stats_fn()
    except Exception:
        return {}
    hits = int(stats.get("prefix_hits", 0))
    misses = int(stats.get("prefix_misses", 0))
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_tokens": int(stats.get("prefix_hit_tokens", 0)),
        "hit_rate": round(hits / total, 6) if total else 0.0,
        "cached_blocks": int(stats.get("prefix_cached_blocks", 0)),
        "referenced_blocks": int(stats.get("prefix_referenced_blocks", 0)),
        "evictable_blocks": int(stats.get("prefix_evictable_blocks", 0)),
    }


def _per_worker_stats(results: Iterable[RequestMetrics]) -> dict[str, dict[str, int | float]]:
    """Aggregate per-worker distribution across succeeded and failed requests."""
    workers: dict[int, dict[str, int | float]] = {}
    for result in results:
        if result.worker_id is None:
            continue
        bucket = workers.setdefault(
            result.worker_id,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "max_tpot_ms": 0.0,
                "max_ttft_ms": 0.0,
            },
        )
        bucket["requests"] = int(bucket["requests"]) + 1
        bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + result.prompt_tokens
        bucket["output_tokens"] = int(bucket["output_tokens"]) + result.completion_tokens
        if result.tpot_ms is not None:
            bucket["max_tpot_ms"] = max(float(bucket["max_tpot_ms"]), result.tpot_ms)
        if result.ttft_ms is not None:
            bucket["max_ttft_ms"] = max(float(bucket["max_ttft_ms"]), result.ttft_ms)
    return {str(worker_id): bucket for worker_id, bucket in sorted(workers.items())}


def run_benchmark(
    generation_loop,
    tokenizer,
    samples: Iterable[BenchmarkSample],
    *,
    max_new_tokens: int = 32,
    concurrency: int = 1,
    max_prompt_tokens: int | None = None,
    warmup_requests: int = 0,
    request_rate: float | None = None,
    arrival_pattern: str = "burst",
    seed: int = 0,
) -> BenchmarkSummary:
    if max_new_tokens <= 0 or concurrency <= 0:
        raise ValueError("max_new_tokens and concurrency must be positive")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    if request_rate is not None and request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if arrival_pattern not in {"burst", "fixed", "poisson"}:
        raise ValueError("arrival_pattern must be burst, fixed, or poisson")
    if arrival_pattern != "burst" and request_rate is None:
        raise ValueError("fixed/poisson arrivals require request_rate")
    all_samples = tuple(samples)
    warmups = all_samples[:warmup_requests]
    indexed = tuple(enumerate(all_samples[warmup_requests:]))

    def encode(sample: BenchmarkSample):
        token_ids = tokenizer.encode(sample.prompt)
        if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[-max_prompt_tokens:]
        return token_ids

    for sample in warmups:
        handle = generation_loop.submit(encode(sample), max_new_tokens)
        terminal = None
        for terminal in handle:
            pass
        if terminal is None or terminal.error:
            raise RuntimeError(
                f"warmup request {sample.sample_id} failed: "
                f"{None if terminal is None else terminal.error}"
            )
    if warmups:
        reset_calibration = getattr(
            generation_loop.backend, "reset_routing_calibration", None
        )
        if reset_calibration is not None:
            reset_calibration()

    def run_one(item) -> tuple[int, RequestMetrics]:
        index, sample = item
        token_ids = encode(sample)
        started = perf_counter()
        first_token_at = None
        completion_tokens = 0
        finish_reason = "error"
        error = None
        handle = None
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
        admission_wait_ms = (
            None if handle is None else handle.request.admission_wait_ms
        )
        observed_service_ms = (
            None
            if handle is None
            else handle.request.route_observed_prefill_service_ms
        )
        observed_queue_ms = (
            None
            if handle is None
            else handle.request.observed_prefill_queue_wait_ms
        )
        return index, RequestMetrics(
            sample_id=sample.sample_id,
            request_id=None if handle is None else handle.request_id,
            prompt_tokens=len(token_ids),
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            error=error,
            route=None if handle is None else (handle.request.route or "collocated"),
            route_reason=None if handle is None else handle.request.route_reason,
            worker_id=None if handle is None else handle.request.worker_id,
            route_collocated_cost_ms=(
                None if handle is None else handle.request.route_collocated_cost_ms
            ),
            route_pd_cost_ms=(
                None if handle is None else handle.request.route_pd_cost_ms
            ),
            route_estimated_savings_ms=(
                None
                if handle is None
                else handle.request.route_estimated_savings_ms
            ),
            route_cost_confidence=(
                None if handle is None else handle.request.route_cost_confidence
            ),
            route_decode_load=(
                None if handle is None else handle.request.route_decode_load
            ),
            route_prefill_load=(
                None if handle is None else getattr(handle.request, "route_prefill_load", None)
            ),
            route_prefill_queue_ahead_ms=(
                0.0
                if handle is None
                else handle.request.route_prefill_queue_ahead_ms
            ),
            route_observed_prefill_service_ms=observed_service_ms,
            admission_wait_ms=admission_wait_ms,
            observed_prefill_queue_wait_ms=observed_queue_ms,
        )

    offsets: list[float] = []
    if arrival_pattern == "burst":
        offsets = [0.0] * len(indexed)
    elif arrival_pattern == "fixed":
        offsets = [index / request_rate for index in range(len(indexed))]
    else:
        random = Random(seed)
        elapsed = 0.0
        for index in range(len(indexed)):
            if index:
                elapsed += random.expovariate(request_rate)
            offsets.append(elapsed)

    wall_started = perf_counter()
    ordered_results: list[RequestMetrics | None] = [None] * len(indexed)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for item, offset in zip(indexed, offsets, strict=True):
            delay = wall_started + offset - perf_counter()
            if delay > 0:
                sleep(delay)
            futures.append(executor.submit(run_one, item))
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result
    wall_time = perf_counter() - wall_started
    results = tuple(result for result in ordered_results if result is not None)
    succeeded = tuple(result for result in results if result.error is None)
    divisor = wall_time if wall_time > 0 else float("inf")
    route_counts: dict[str, int] = {}
    for result in succeeded:
        route = result.route or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1
    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / divisor,
        output_token_throughput=sum(result.completion_tokens for result in succeeded) / divisor,
        warmup_requests=len(warmups),
        offered_request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        ttft_ms=_percentiles(
            result.ttft_ms for result in succeeded if result.ttft_ms is not None
        ),
        tpot_ms=_percentiles(
            result.tpot_ms for result in succeeded if result.tpot_ms is not None
        ),
        latency_ms=_percentiles(result.latency_ms for result in succeeded),
        route_counts=route_counts,
        prefix_cache_stats=_prefix_cache_stats(generation_loop),
        per_worker=_per_worker_stats(results),
        results=results,
    )


def _iter_completion_sse(
    endpoint: str,
    prompt: str,
    max_new_tokens: int,
    *,
    timeout: float = 600.0,
) -> Iterable[tuple[str, str | None]]:
    """Stream one completion from a HydraServe endpoint as SSE ``(kind, value)``.

    ``kind`` is ``token``, ``finish``, or ``error``; ``value`` carries the finish
    reason or error message. The response body is the server's ``text/event-stream``.
    """
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_new_tokens, "stream": True}
    ).encode("utf-8")
    req = request.Request(
        f"{endpoint.rstrip('/')}/v1/completions", data=body, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    with request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                message = chunk["error"].get("message", "server error")
                yield "error", str(message)
                return
            choices = chunk.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                yield "finish", choices[0]["finish_reason"]
                continue
            yield "token", None


def run_http_benchmark(
    endpoint: str,
    tokenizer,
    samples: Iterable[BenchmarkSample],
    *,
    max_new_tokens: int = 32,
    concurrency: int = 1,
    warmup_requests: int = 0,
    request_rate: float | None = None,
    arrival_pattern: str = "burst",
    seed: int = 0,
    timeout_s: float = 600.0,
) -> BenchmarkSummary:
    """Benchmark a remote HydraServe endpoint (e.g. a load-aware DP proxy).

    Unlike :func:`run_benchmark`, requests are issued over HTTP to ``endpoint``
    (an OpenAI-style ``/v1/completions`` server) rather than an in-process
    generation loop, so multi-process DP deployments can be driven directly.
    """
    if max_new_tokens <= 0 or concurrency <= 0:
        raise ValueError("max_new_tokens and concurrency must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    if request_rate is not None and request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if arrival_pattern not in {"burst", "fixed", "poisson"}:
        raise ValueError("arrival_pattern must be burst, fixed, or poisson")
    if arrival_pattern != "burst" and request_rate is None:
        raise ValueError("fixed/poisson arrivals require request_rate")

    all_samples = tuple(samples)
    warmups = all_samples[:warmup_requests]
    indexed = tuple(enumerate(all_samples[warmup_requests:]))

    for sample in warmups:
        for kind, value in _iter_completion_sse(
            endpoint, sample.prompt, max_new_tokens, timeout=timeout_s
        ):
            if kind == "error":
                raise RuntimeError(f"warmup request {sample.sample_id} failed: {value}")

    def run_one(item) -> tuple[int, RequestMetrics]:
        index, sample = item
        prompt_tokens = len(tokenizer.encode(sample.prompt))
        started = perf_counter()
        first_token_at = None
        completion_tokens = 0
        finish_reason = "error"
        error = None
        try:
            for kind, value in _iter_completion_sse(
                endpoint, sample.prompt, max_new_tokens, timeout=timeout_s
            ):
                now = perf_counter()
                if kind == "token":
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = now
                elif kind == "finish":
                    finish_reason = value or "stop"
                elif kind == "error":
                    error = value
        except Exception as exc:  # network error / HTTP status / timeout
            error = str(exc)
        ended = perf_counter()
        latency_ms = (ended - started) * 1000
        ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000
        tpot_ms = None
        if first_token_at is not None and completion_tokens > 1:
            tpot_ms = (ended - first_token_at) * 1000 / (completion_tokens - 1)
        return index, RequestMetrics(
            sample_id=sample.sample_id,
            request_id=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            error=error,
            route=None,
            route_reason=None,
            worker_id=None,
        )

    if arrival_pattern == "burst":
        offsets = [0.0] * len(indexed)
    elif arrival_pattern == "fixed":
        offsets = [index / request_rate for index in range(len(indexed))]
    else:
        random = Random(seed)
        elapsed = 0.0
        offsets = []
        for index in range(len(indexed)):
            if index:
                elapsed += random.expovariate(request_rate)
            offsets.append(elapsed)

    wall_started = perf_counter()
    ordered_results: list[RequestMetrics | None] = [None] * len(indexed)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for item, offset in zip(indexed, offsets, strict=True):
            delay = wall_started + offset - perf_counter()
            if delay > 0:
                sleep(delay)
            futures.append(executor.submit(run_one, item))
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result
    wall_time = perf_counter() - wall_started
    results = tuple(result for result in ordered_results if result is not None)
    succeeded = tuple(result for result in results if result.error is None)
    divisor = wall_time if wall_time > 0 else float("inf")
    route_counts: dict[str, int] = {}
    for result in succeeded:
        route = result.route or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1
    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / divisor,
        output_token_throughput=sum(result.completion_tokens for result in succeeded) / divisor,
        warmup_requests=len(warmups),
        offered_request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        ttft_ms=_percentiles(
            result.ttft_ms for result in succeeded if result.ttft_ms is not None
        ),
        tpot_ms=_percentiles(
            result.tpot_ms for result in succeeded if result.tpot_ms is not None
        ),
        latency_ms=_percentiles(result.latency_ms for result in succeeded),
        route_counts=route_counts,
        prefix_cache_stats={},
        per_worker={},
        results=results,
    )
