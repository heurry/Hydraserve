"""Tokenizer-aware benchmark execution and latency aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
from math import ceil
from random import Random
from time import perf_counter, sleep
from typing import Any, Iterable
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
    # ``ttft_ms`` starts when the client worker actually submits to the engine.
    # ``e2e_ttft_ms`` starts at the trace's intended arrival time and therefore
    # includes any client-side executor queueing.
    e2e_ttft_ms: float | None = None
    client_queue_ms: float = 0.0
    # Time between the last visible token and the terminal event.  Keep this
    # separate from TPOT because engine cleanup/release may be synchronous.
    release_tail_ms: float | None = None
    itl_ms: tuple[float, ...] = ()
    decode_batch_sizes: tuple[int, ...] = ()
    route: str | None = None
    route_reason: str | None = None
    worker_id: int | None = None
    worker_pool: str | None = None
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
    # P0 trace fields (V3 plan).
    klass: str = "default"
    ignore_eos: bool = False
    target_new_tokens: int | None = None


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
    # P0 additions (V3 plan): per-class metrics, SLO goodput, reproducibility.
    by_class: dict[str, dict[str, int | float]] = field(default_factory=dict)
    slo: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    e2e_ttft_ms: dict[str, float] = field(default_factory=dict)
    itl_ms: dict[str, float] = field(default_factory=dict)
    release_tail_ms: dict[str, float] = field(default_factory=dict)
    client_queue_ms: dict[str, float] = field(default_factory=dict)
    route_counts_all: dict[str, int] = field(default_factory=dict)
    route_failure_counts: dict[str, int] = field(default_factory=dict)
    throughput_valid: bool = True

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
            "by_class": self.by_class,
            "slo": self.slo,
            "metadata": self.metadata,
            "e2e_ttft_ms": self.e2e_ttft_ms,
            "itl_ms": self.itl_ms,
            "release_tail_ms": self.release_tail_ms,
            "client_queue_ms": self.client_queue_ms,
            "route_counts_all": self.route_counts_all,
            "route_failure_counts": self.route_failure_counts,
            "throughput_valid": self.throughput_valid,
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

    return {
        name: value(percentile)
        for name, percentile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }


def _count_routes(results: Iterable[RequestMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        route = result.route or "unknown"
        counts[route] = counts.get(route, 0) + 1
    return counts


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


def _per_worker_stats(
    results: Iterable[RequestMetrics],
) -> dict[str, dict[str, int | float]]:
    """Aggregate per-worker distribution across succeeded and failed requests."""
    workers: dict[str, dict[str, int | float]] = {}
    for result in results:
        if result.worker_id is None:
            continue
        worker_key = (
            str(result.worker_id)
            if result.worker_pool is None
            else f"{result.worker_pool}:{result.worker_id}"
        )
        bucket = workers.setdefault(
            worker_key,
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
        bucket["output_tokens"] = (
            int(bucket["output_tokens"]) + result.completion_tokens
        )
        if result.tpot_ms is not None:
            bucket["max_tpot_ms"] = max(float(bucket["max_tpot_ms"]), result.tpot_ms)
        if result.ttft_ms is not None:
            bucket["max_ttft_ms"] = max(float(bucket["max_ttft_ms"]), result.ttft_ms)
    return dict(sorted(workers.items()))


@dataclass(frozen=True, slots=True)
class SLOConfig:
    """Service-level-objective thresholds (V5 plan section 4).

    Defaults are the V5 service targets and are not meant to be tuned to a
    particular run. TTFT thresholds are applied end-to-end: they compare
    against ``e2e_ttft_ms`` (measured from the trace's intended arrival time,
    including client executor queueing) when it is available.
    """

    short_ttft_ms: float = 1000.0
    short_tpot_ms: float = 100.0
    long_ttft_ms: float = 10_000.0
    long_tpot_ms: float = 150.0
    long_max_admission_wait_ms: float = 30_000.0
    # A request whose admission wait exceeds this is reported as starved.
    starvation_admission_wait_ms: float = 30_000.0


def _by_class_stats(
    results: Iterable[RequestMetrics], wall_time: float
) -> dict[str, dict[str, int | float]]:
    """Per-request-class aggregation (P0.3): throughput, percentiles, failures."""
    buckets: dict[str, list[RequestMetrics]] = {}
    for result in results:
        buckets.setdefault(result.klass, []).append(result)
    out: dict[str, dict[str, int | float]] = {}
    for klass, rs in sorted(buckets.items()):
        succeeded = [r for r in rs if r.error is None]
        divisor = wall_time if wall_time > 0 else float("inf")
        out[klass] = {
            "requests": len(rs),
            "succeeded": len(succeeded),
            "failed": len(rs) - len(succeeded),
            "throughput_valid": len(succeeded) == len(rs),
            "output_tokens": sum(r.completion_tokens for r in succeeded),
            "request_throughput": round(len(succeeded) / divisor, 4),
            "output_token_throughput": round(
                sum(r.completion_tokens for r in succeeded) / divisor, 4
            ),
            "ttft_ms": _percentiles(
                r.ttft_ms for r in succeeded if r.ttft_ms is not None
            ),
            "e2e_ttft_ms": _percentiles(
                r.e2e_ttft_ms for r in succeeded if r.e2e_ttft_ms is not None
            ),
            "tpot_ms": _percentiles(
                r.tpot_ms for r in succeeded if r.tpot_ms is not None
            ),
            "itl_ms": _percentiles(value for r in succeeded for value in r.itl_ms),
            "release_tail_ms": _percentiles(
                r.release_tail_ms for r in succeeded if r.release_tail_ms is not None
            ),
            "client_queue_ms": _percentiles(r.client_queue_ms for r in rs),
            "latency_ms": _percentiles(r.latency_ms for r in succeeded),
            "error_reasons": _count_errors(rs),
        }
    return out


def _count_errors(results: Iterable[RequestMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        if result.error:
            counts[result.error] = counts.get(result.error, 0) + 1
    return counts


def _full_output_ok(result: RequestMetrics, target: int | None) -> bool:
    """Whether a request produced its full intended output (V5 plan gate 5).

    V5 real mode runs ``ignore_eos=false``, so a legitimate early EOS must
    count as a complete success instead of failing the ``completion_tokens >=
    target`` check. Reaching ``target`` or hitting a normal ``stop`` both
    qualify; only truncation/error paths are penalised.
    """
    if result.error:
        return False
    if result.finish_reason == "stop":
        return True
    if target is None:
        return True
    return result.completion_tokens >= target


def _class_slo_stats(
    results: list[RequestMetrics],
    klass: str,
    *,
    ttft_threshold_ms: float,
    tpot_threshold_ms: float,
    max_admission_wait_ms: float | None,
    wall_time: float,
) -> dict[str, int | float]:
    """SLO goodput for one request class (V5 plan section 4/7).

    A request meets SLO when it did not error, its end-to-end TTFT and TPOT are
    within threshold, it produced its full output (target reached or normal
    EOS), and -- when the class defines one -- its admission wait is bounded.
    """
    members = [result for result in results if result.klass == klass]
    met = 0
    met_tokens = 0
    ttft_ok = tpot_ok = full_ok = admission_ok = no_error = 0
    for result in members:
        ok = True
        if result.error:
            ok = False
        else:
            no_error += 1
        slo_ttft_ms = (
            result.e2e_ttft_ms if result.e2e_ttft_ms is not None else result.ttft_ms
        )
        if slo_ttft_ms is not None and slo_ttft_ms <= ttft_threshold_ms:
            ttft_ok += 1
        else:
            ok = False
        if result.tpot_ms is not None and result.tpot_ms <= tpot_threshold_ms:
            tpot_ok += 1
        else:
            ok = False
        if max_admission_wait_ms is not None:
            admission_ms = result.admission_wait_ms or 0.0
            if admission_ms <= max_admission_wait_ms:
                admission_ok += 1
            else:
                ok = False
        if _full_output_ok(result, result.target_new_tokens):
            full_ok += 1
        else:
            ok = False
        if ok:
            met += 1
            met_tokens += result.completion_tokens
    divisor = wall_time if wall_time > 0 else float("inf")
    return {
        "requests": len(members),
        "met_requests": met,
        "met_rate": round(met / len(members), 4) if members else 0.0,
        "goodput_requests_s": round(met / divisor, 4),
        "goodput_tokens_s": round(met_tokens / divisor, 4),
        "ttft_ok": ttft_ok,
        "tpot_ok": tpot_ok,
        "full_output_ok": full_ok,
        "admission_ok": admission_ok,
        "no_error": no_error,
    }


def _route_reason_counts(results: Iterable[RequestMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        reason = result.route_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _admission_wait_percentiles(
    results: Iterable[RequestMetrics],
) -> dict[str, float]:
    values = [
        result.admission_wait_ms
        for result in results
        if result.admission_wait_ms is not None
    ]
    if not values:
        return {}
    percentiles = _percentiles(values)
    percentiles["max"] = max(values)
    return percentiles


def _slo_stats(
    results: Iterable[RequestMetrics],
    wall_time: float,
    slo: SLOConfig | None = None,
) -> dict[str, int | float]:
    """Per-class SLO goodput and V5 fairness/observability stats.

    ``slo`` defaults to the V5 service targets. Short keys mirror the V3 flat
    layout for backward compatibility; Long and fairness keys are V5 additions.
    """
    slo = slo or SLOConfig()
    members = list(results)
    short = _class_slo_stats(
        members,
        "short",
        ttft_threshold_ms=slo.short_ttft_ms,
        tpot_threshold_ms=slo.short_tpot_ms,
        max_admission_wait_ms=None,
        wall_time=wall_time,
    )
    long = _class_slo_stats(
        members,
        "long",
        ttft_threshold_ms=slo.long_ttft_ms,
        tpot_threshold_ms=slo.long_tpot_ms,
        max_admission_wait_ms=slo.long_max_admission_wait_ms,
        wall_time=wall_time,
    )
    reason_counts = _route_reason_counts(members)
    # "Fallback" = a Long request that would have used the Hybrid prefill slot
    # but was admitted to a D-bound worker instead because the Hybrid was
    # saturated (queue overflow) or unavailable.
    long_fallbacks = sum(
        1
        for result in members
        if result.klass == "long"
        and result.route_reason in {"hybrid_queue_overflow", "prefill_unavailable"}
    )
    starved = sum(
        1
        for result in members
        if result.admission_wait_ms is not None
        and result.admission_wait_ms > slo.starvation_admission_wait_ms
    )
    return {
        # short (V3 flat layout, kept for backward compatibility).
        "short_requests": short["requests"],
        "met_requests": short["met_requests"],
        "met_rate": short["met_rate"],
        "goodput_requests_s": short["goodput_requests_s"],
        "goodput_tokens_s": short["goodput_tokens_s"],
        "ttft_ok": short["ttft_ok"],
        "tpot_ok": short["tpot_ok"],
        "full_output_ok": short["full_output_ok"],
        "no_error": short["no_error"],
        "ttft_threshold_ms": slo.short_ttft_ms,
        "tpot_threshold_ms": slo.short_tpot_ms,
        "require_full_output": True,
        # long (V5).
        "long_requests": long["requests"],
        "long_met_requests": long["met_requests"],
        "long_met_rate": long["met_rate"],
        "long_goodput_requests_s": long["goodput_requests_s"],
        "long_goodput_tokens_s": long["goodput_tokens_s"],
        "long_ttft_ok": long["ttft_ok"],
        "long_tpot_ok": long["tpot_ok"],
        "long_full_output_ok": long["full_output_ok"],
        "long_admission_ok": long["admission_ok"],
        "long_no_error": long["no_error"],
        "long_ttft_threshold_ms": slo.long_ttft_ms,
        "long_tpot_threshold_ms": slo.long_tpot_ms,
        "long_max_admission_wait_ms": slo.long_max_admission_wait_ms,
        # fairness / observability (V5).
        "admission_wait_ms": _admission_wait_percentiles(members),
        "route_reason_counts": reason_counts,
        "overflow_count": reason_counts.get("hybrid_queue_overflow", 0),
        "long_fallback_count": long_fallbacks,
        "starved_requests": starved,
        "starvation_admission_wait_ms": slo.starvation_admission_wait_ms,
    }


def run_benchmark(
    generation_loop,
    tokenizer,
    samples: Iterable[BenchmarkSample],
    *,
    max_new_tokens: int = 32,
    concurrency: int = 1,
    max_prompt_tokens: int | None = None,
    warmup_requests: int = 0,
    warmup_samples: Iterable[BenchmarkSample] | None = None,
    request_rate: float | None = None,
    arrival_pattern: str = "burst",
    seed: int = 0,
    slo: SLOConfig | None = None,
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
    # A frozen trace is the measured workload: CLI callers provide separate
    # prompts so warmup never consumes official records. Keep the historical
    # prefix-consuming behavior only when no independent pool is supplied.
    warmup_pool = tuple(warmup_samples) if warmup_samples is not None else all_samples
    warmups = warmup_pool[:warmup_requests]
    if len(warmups) != warmup_requests:
        raise ValueError("not enough independent warmup samples")
    measured_samples = (
        all_samples if warmup_samples is not None else all_samples[warmup_requests:]
    )
    indexed = tuple(enumerate(measured_samples))

    def encode(sample: BenchmarkSample):
        token_ids = tokenizer.encode(sample.prompt)
        if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[-max_prompt_tokens:]
        return token_ids

    def sample_priority(sample: BenchmarkSample) -> int:
        """Give interactive Short trace records first-class admission priority.

        The V5 comparison is about serving quality under mixed traffic, not
        about letting burst ordering accidentally starve Short requests behind
        Long prefill.  Applying the same priority to DP and H1 keeps the
        baseline honest.
        """

        return 1 if sample.klass == "short" else 0

    for sample in warmups:
        handle = generation_loop.submit(
            encode(sample),
            sample.max_new_tokens or max_new_tokens,
            ignore_eos=sample.ignore_eos,
        )
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

    def run_one(item, scheduled_at: float) -> tuple[int, RequestMetrics]:
        index, sample = item
        token_ids = encode(sample)
        per_sample_max = sample.max_new_tokens or max_new_tokens
        started = perf_counter()
        client_queue_ms = max(0.0, (started - scheduled_at) * 1000)
        first_token_at = None
        last_token_at = None
        itl_ms: list[float] = []
        decode_batch_sizes: list[int] = []
        completion_tokens = 0
        finish_reason = "error"
        error = None
        handle = None
        try:
            handle = generation_loop.submit(
                token_ids,
                per_sample_max,
                ignore_eos=sample.ignore_eos,
                priority=sample_priority(sample),
            )
            for event in handle:
                now = perf_counter()
                if event.token_id is not None:
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = now
                    if last_token_at is not None:
                        itl_ms.append((now - last_token_at) * 1000)
                    last_token_at = now
                    batch_size = getattr(event, "decode_batch_size", None)
                    if batch_size is not None:
                        decode_batch_sizes.append(int(batch_size))
                if event.finished:
                    finish_reason = event.finish_reason or "unknown"
                    error = event.error
        except Exception as exc:
            error = str(exc)
        ended = perf_counter()
        latency_ms = (ended - started) * 1000
        ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000
        e2e_ttft_ms = (
            None
            if first_token_at is None
            else max(0.0, (first_token_at - scheduled_at) * 1000)
        )
        tpot_ms = None
        if (
            first_token_at is not None
            and last_token_at is not None
            and completion_tokens > 1
        ):
            tpot_ms = (last_token_at - first_token_at) * 1000 / (completion_tokens - 1)
        release_tail_ms = (
            None if last_token_at is None else max(0.0, (ended - last_token_at) * 1000)
        )
        admission_wait_ms = None if handle is None else handle.request.admission_wait_ms
        observed_service_ms = (
            None if handle is None else handle.request.route_observed_prefill_service_ms
        )
        observed_queue_ms = (
            None if handle is None else handle.request.observed_prefill_queue_wait_ms
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
            e2e_ttft_ms=e2e_ttft_ms,
            client_queue_ms=client_queue_ms,
            release_tail_ms=release_tail_ms,
            itl_ms=tuple(itl_ms),
            decode_batch_sizes=tuple(decode_batch_sizes),
            route=None if handle is None else handle.request.route,
            route_reason=None if handle is None else handle.request.route_reason,
            worker_id=None if handle is None else handle.request.worker_id,
            worker_pool=None if handle is None else handle.request.worker_pool,
            route_collocated_cost_ms=(
                None if handle is None else handle.request.route_collocated_cost_ms
            ),
            route_pd_cost_ms=(
                None if handle is None else handle.request.route_pd_cost_ms
            ),
            route_estimated_savings_ms=(
                None if handle is None else handle.request.route_estimated_savings_ms
            ),
            route_cost_confidence=(
                None if handle is None else handle.request.route_cost_confidence
            ),
            route_decode_load=(
                None if handle is None else handle.request.route_decode_load
            ),
            route_prefill_load=(
                None
                if handle is None
                else getattr(handle.request, "route_prefill_load", None)
            ),
            route_prefill_queue_ahead_ms=(
                0.0 if handle is None else handle.request.route_prefill_queue_ahead_ms
            ),
            route_observed_prefill_service_ms=observed_service_ms,
            admission_wait_ms=admission_wait_ms,
            observed_prefill_queue_wait_ms=observed_queue_ms,
            klass=sample.klass,
            ignore_eos=sample.ignore_eos,
            target_new_tokens=sample.max_new_tokens or max_new_tokens,
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
        scheduled = sorted(
            zip(indexed, offsets, strict=True),
            key=lambda pair: pair[1] + pair[0][1].arrival_offset_ms / 1000.0,
        )
        for item, offset in scheduled:
            index, sample = item
            trace_offset = sample.arrival_offset_ms / 1000.0
            delay = wall_started + offset + trace_offset - perf_counter()
            if delay > 0:
                sleep(delay)
            scheduled_at = wall_started + offset + trace_offset
            futures.append(executor.submit(run_one, item, scheduled_at))
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result
    wall_time = perf_counter() - wall_started
    results = tuple(result for result in ordered_results if result is not None)
    succeeded = tuple(result for result in results if result.error is None)
    divisor = wall_time if wall_time > 0 else float("inf")
    route_counts = _count_routes(succeeded)
    failed_results = tuple(result for result in results if result.error is not None)
    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / divisor,
        output_token_throughput=sum(result.completion_tokens for result in succeeded)
        / divisor,
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
        e2e_ttft_ms=_percentiles(
            result.e2e_ttft_ms for result in succeeded if result.e2e_ttft_ms is not None
        ),
        itl_ms=_percentiles(value for result in succeeded for value in result.itl_ms),
        release_tail_ms=_percentiles(
            result.release_tail_ms
            for result in succeeded
            if result.release_tail_ms is not None
        ),
        client_queue_ms=_percentiles(result.client_queue_ms for result in results),
        route_counts_all=_count_routes(results),
        route_failure_counts=_count_routes(failed_results),
        throughput_valid=not failed_results,
        route_counts=route_counts,
        prefix_cache_stats=_prefix_cache_stats(generation_loop),
        per_worker=_per_worker_stats(results),
        results=results,
        by_class=_by_class_stats(results, wall_time),
        slo=_slo_stats(results, wall_time, slo=slo),
    )


def _iter_completion_sse(
    endpoint: str,
    prompt: str,
    max_new_tokens: int,
    *,
    ignore_eos: bool = False,
    timeout: float = 600.0,
) -> Iterable[tuple[str, str | None]]:
    """Stream one completion from a HydraServe endpoint as SSE ``(kind, value)``.

    ``kind`` is ``token``, ``finish``, or ``error``; ``value`` carries the finish
    reason or error message. The response body is the server's ``text/event-stream``.
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "stream": True,
            "ignore_eos": ignore_eos,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{endpoint.rstrip('/')}/v1/completions", data=body, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    hydraserve_stream = False
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
            if "hydraserve_token_id" in chunk:
                hydraserve_stream = True
                yield "token", None
                continue
            if hydraserve_stream:
                # ``IncrementalDecoder.finish()`` can emit one final text-only
                # SSE chunk. It is not a generated token and must not inflate
                # completion count or add a fake ITL sample.
                continue
            yield "token", None


def run_http_benchmark(
    endpoint: str,
    tokenizer,
    samples: Iterable[BenchmarkSample],
    *,
    max_new_tokens: int = 32,
    concurrency: int = 1,
    max_prompt_tokens: int | None = None,
    warmup_requests: int = 0,
    warmup_samples: Iterable[BenchmarkSample] | None = None,
    request_rate: float | None = None,
    arrival_pattern: str = "burst",
    seed: int = 0,
    timeout_s: float = 600.0,
    slo: SLOConfig | None = None,
) -> BenchmarkSummary:
    """Benchmark a remote HydraServe endpoint (e.g. a load-aware DP proxy).

    Unlike :func:`run_benchmark`, requests are issued over HTTP to ``endpoint``
    (an OpenAI-style ``/v1/completions`` server) rather than an in-process
    generation loop, so multi-process DP deployments can be driven directly.
    ``max_prompt_tokens`` is applied the same way as the in-process runner
    (tail-truncation of the encoded prompt) so both paths feed identical tokens.
    """
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

    def truncate_prompt(sample: BenchmarkSample) -> str:
        if max_prompt_tokens is None:
            return sample.prompt
        ids = tokenizer.encode(sample.prompt)
        if len(ids) <= max_prompt_tokens:
            return sample.prompt
        return tokenizer.decode(ids[-max_prompt_tokens:])

    all_samples = tuple(samples)
    warmup_pool = tuple(warmup_samples) if warmup_samples is not None else all_samples
    warmups = warmup_pool[:warmup_requests]
    if len(warmups) != warmup_requests:
        raise ValueError("not enough independent warmup samples")
    measured_samples = (
        all_samples if warmup_samples is not None else all_samples[warmup_requests:]
    )
    indexed = tuple(enumerate(measured_samples))

    for sample in warmups:
        for kind, value in _iter_completion_sse(
            endpoint,
            truncate_prompt(sample),
            sample.max_new_tokens or max_new_tokens,
            ignore_eos=sample.ignore_eos,
            timeout=timeout_s,
        ):
            if kind == "error":
                raise RuntimeError(f"warmup request {sample.sample_id} failed: {value}")

    def run_one(item, scheduled_at: float) -> tuple[int, RequestMetrics]:
        index, sample = item
        per_sample_max = sample.max_new_tokens or max_new_tokens
        prompt = truncate_prompt(sample)
        prompt_tokens = len(tokenizer.encode(prompt))
        started = perf_counter()
        client_queue_ms = max(0.0, (started - scheduled_at) * 1000)
        first_token_at = None
        last_token_at = None
        itl_ms: list[float] = []
        completion_tokens = 0
        finish_reason = "error"
        error = None
        try:
            for kind, value in _iter_completion_sse(
                endpoint,
                prompt,
                per_sample_max,
                ignore_eos=sample.ignore_eos,
                timeout=timeout_s,
            ):
                now = perf_counter()
                if kind == "token":
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = now
                    if last_token_at is not None:
                        itl_ms.append((now - last_token_at) * 1000)
                    last_token_at = now
                elif kind == "finish":
                    finish_reason = value or "stop"
                elif kind == "error":
                    error = value
        except Exception as exc:  # network error / HTTP status / timeout
            error = str(exc)
        ended = perf_counter()
        latency_ms = (ended - started) * 1000
        ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000
        e2e_ttft_ms = (
            None
            if first_token_at is None
            else max(0.0, (first_token_at - scheduled_at) * 1000)
        )
        tpot_ms = None
        if (
            first_token_at is not None
            and last_token_at is not None
            and completion_tokens > 1
        ):
            tpot_ms = (last_token_at - first_token_at) * 1000 / (completion_tokens - 1)
        release_tail_ms = (
            None if last_token_at is None else max(0.0, (ended - last_token_at) * 1000)
        )
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
            e2e_ttft_ms=e2e_ttft_ms,
            client_queue_ms=client_queue_ms,
            release_tail_ms=release_tail_ms,
            itl_ms=tuple(itl_ms),
            route=None,
            route_reason=None,
            worker_id=None,
            klass=sample.klass,
            ignore_eos=sample.ignore_eos,
            target_new_tokens=per_sample_max,
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
        scheduled = sorted(
            zip(indexed, offsets, strict=True),
            key=lambda pair: pair[1] + pair[0][1].arrival_offset_ms / 1000.0,
        )
        for item, offset in scheduled:
            index, sample = item
            trace_offset = sample.arrival_offset_ms / 1000.0
            delay = wall_started + offset + trace_offset - perf_counter()
            if delay > 0:
                sleep(delay)
            scheduled_at = wall_started + offset + trace_offset
            futures.append(executor.submit(run_one, item, scheduled_at))
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result
    wall_time = perf_counter() - wall_started
    results = tuple(result for result in ordered_results if result is not None)
    succeeded = tuple(result for result in results if result.error is None)
    divisor = wall_time if wall_time > 0 else float("inf")
    route_counts = _count_routes(succeeded)
    failed_results = tuple(result for result in results if result.error is not None)
    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / divisor,
        output_token_throughput=sum(result.completion_tokens for result in succeeded)
        / divisor,
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
        e2e_ttft_ms=_percentiles(
            result.e2e_ttft_ms for result in succeeded if result.e2e_ttft_ms is not None
        ),
        itl_ms=_percentiles(value for result in succeeded for value in result.itl_ms),
        release_tail_ms=_percentiles(
            result.release_tail_ms
            for result in succeeded
            if result.release_tail_ms is not None
        ),
        client_queue_ms=_percentiles(result.client_queue_ms for result in results),
        route_counts_all=_count_routes(results),
        route_failure_counts=_count_routes(failed_results),
        throughput_valid=not failed_results,
        route_counts=route_counts,
        prefix_cache_stats={},
        per_worker={},
        results=results,
        by_class=_by_class_stats(results, wall_time),
        slo=_slo_stats(results, wall_time, slo=slo),
    )
