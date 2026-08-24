#!/usr/bin/env python3
"""Benchmark vLLM 4×DP using HydraServe frozen traces.

Reads a HydraServe JSONL trace (prompt text + arrival_offset_ms + max_new_tokens
+ ignore_eos + class), replays it against a vLLM OpenAI-compatible endpoint with
the same arrival pattern and SLO criteria as HydraServe's run_benchmark.

Usage:
    python vllm_trace_bench.py \
        --endpoint http://127.0.0.1:8000 \
        --trace /path/to/Hydraserve/traces/r1_rag_qa_seed42.jsonl \
        --output results/vllm/r1_d0_vllm_seed42.json \
        --request-rate 2.0 \
        --arrival-pattern poisson \
        --concurrency 16 \
        --seed 42 \
        --warmup 8

The output JSON has the same structure as HydraServe's BenchmarkSummary,
so results can be diffed directly.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
from random import Random
from time import perf_counter, sleep
from typing import Any, Iterable
from urllib import error, request


# ── SLO thresholds (identical to HydraServe) ──────────────────────────────

SLO_SHORT_TTFT_MS = 5000.0
SLO_SHORT_TPOT_MS = 200.0


# ── Data structures (mirror HydraServe's RequestMetrics / BenchmarkSummary) ──

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
    e2e_ttft_ms: float | None = None
    client_queue_ms: float = 0.0
    itl_ms: tuple[float, ...] = ()
    klass: str = "default"
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
    by_class: dict[str, dict[str, int | float]]
    slo: dict[str, int | float]
    results: tuple[dict, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Trace loading ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceEntry:
    sample_id: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_offset_ms: float
    ignore_eos: bool
    klass: str


def load_trace(path: str) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(
                TraceEntry(
                    sample_id=obj["id"],
                    prompt=obj["prompt"],
                    prompt_tokens=obj.get("prompt_tokens", 0),
                    max_new_tokens=obj["max_new_tokens"],
                    arrival_offset_ms=obj.get("arrival_offset_ms", 0.0),
                    ignore_eos=obj.get("ignore_eos", False),
                    klass=obj.get("class", "default"),
                )
            )
    return entries


# ── vLLM streaming client ──────────────────────────────────────────────────

def stream_completion(
    endpoint: str,
    prompt: str,
    max_tokens: int,
    *,
    ignore_eos: bool = False,
    timeout: float = 600.0,
) -> Iterable[tuple[str, str | None]]:
    """Stream one completion from vLLM as (kind, value) tuples.

    kind is "token", "finish", "usage", or "error".
    Uses the /v1/completions endpoint with stream=true.
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "ignore_eos": ignore_eos,
            # Force the full target output (HydraServe ignore_eos semantics):
            # vLLM otherwise stops a few tokens short on synthetic prompts.
            "min_tokens": max_tokens,
            # Ask vLLM for server-truth usage in the final chunk. Client-side
            # delta counting drops special tokens (skip_special_tokens strips
            # them to empty text), so usage.completion_tokens is the
            # authoritative output length for the SLO full-output check.
            "stream_options": {"include_usage": True},
            # Match HydraServe: no temperature/top_p override, use defaults.
        }
    ).encode("utf-8")
    req = request.Request(
        f"{endpoint.rstrip('/')}/v1/completions", data=body, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    message = chunk["error"]
                    if isinstance(message, dict):
                        message = message.get("message", "server error")
                    yield "error", str(message)
                    return
                if "usage" in chunk and chunk["usage"]:
                    yield "usage", chunk["usage"].get("completion_tokens", 0)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    yield "finish", finish_reason
                # vLLM streams text deltas; count as token if text non-empty
                text = choice.get("text", "")
                if text:
                    yield "token", text
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        yield "error", f"HTTP {exc.code}: {body_text[:500]}"
    except Exception as exc:
        yield "error", str(exc)


# ── Percentiles (identical to HydraServe) ──────────────────────────────────

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


# ── SLO computation (identical to HydraServe) ──────────────────────────────

def _slo_stats(results: Iterable[RequestMetrics], wall_time: float) -> dict:
    """Short-request SLO goodput. Same criteria as HydraServe:
    TTFT <= 5s, TPOT <= 200ms, full output produced, no error."""
    met = 0
    met_tokens = 0
    total_short = 0
    ttft_ok = tpot_ok = full_ok = no_error = 0
    for result in results:
        if result.klass != "short":
            continue
        total_short += 1
        ok = True
        if result.error:
            no_error += 0
            ok = False
        else:
            no_error += 1
        slo_ttft = result.e2e_ttft_ms if result.e2e_ttft_ms is not None else result.ttft_ms
        if slo_ttft is not None and slo_ttft <= SLO_SHORT_TTFT_MS:
            ttft_ok += 1
        else:
            ok = False
        if result.tpot_ms is not None and result.tpot_ms <= SLO_SHORT_TPOT_MS:
            tpot_ok += 1
        else:
            ok = False
        target = result.target_new_tokens
        if target is not None and result.completion_tokens >= target:
            full_ok += 1
        elif target is None:
            full_ok += 1
        else:
            ok = False
        if ok:
            met += 1
            met_tokens += result.completion_tokens
    divisor = wall_time if wall_time > 0 else float("inf")
    return {
        "short_requests": total_short,
        "met_requests": met,
        "met_rate": round(met / total_short, 4) if total_short else 0.0,
        "goodput_requests_s": round(met / divisor, 4),
        "goodput_tokens_s": round(met_tokens / divisor, 4),
        "ttft_ok": ttft_ok,
        "tpot_ok": tpot_ok,
        "full_output_ok": full_ok,
        "no_error": no_error,
        "ttft_threshold_ms": SLO_SHORT_TTFT_MS,
        "tpot_threshold_ms": SLO_SHORT_TPOT_MS,
        "require_full_output": True,
    }


def _by_class_stats(results: Iterable[RequestMetrics], wall_time: float) -> dict:
    """Per-class (long/short) latency breakdown."""
    classes: dict[str, list[RequestMetrics]] = {}
    for result in results:
        classes.setdefault(result.klass, []).append(result)
    stats: dict[str, dict] = {}
    for klass, items in classes.items():
        tpot_vals = [r.tpot_ms for r in items if r.tpot_ms is not None]
        ttft_vals = [r.ttft_ms for r in items if r.ttft_ms is not None]
        latency_vals = [r.latency_ms for r in items]
        stats[klass] = {
            "count": len(items),
            "tpot_p50": _percentiles(tpot_vals).get("p50"),
            "tpot_p99": _percentiles(tpot_vals).get("p99"),
            "ttft_p50": _percentiles(ttft_vals).get("p50"),
            "ttft_p99": _percentiles(ttft_vals).get("p99"),
            "latency_p50": _percentiles(latency_vals).get("p50"),
            "latency_p99": _percentiles(latency_vals).get("p99"),
            "output_tokens": sum(r.completion_tokens for r in items),
        }
    return stats


# ── Main benchmark loop (mirrors HydraServe run_http_benchmark) ────────────

def run_benchmark(
    endpoint: str,
    entries: list[TraceEntry],
    *,
    concurrency: int = 16,
    request_rate: float | None = None,
    arrival_pattern: str = "burst",
    seed: int = 42,
    warmup: int = 8,
    timeout_s: float = 600.0,
) -> BenchmarkSummary:
    indexed = tuple(enumerate(entries))

    # Warmup
    for i in range(min(warmup, len(entries))):
        entry = entries[i % len(entries)]
        for kind, value in stream_completion(
            endpoint,
            entry.prompt,
            min(entry.max_new_tokens, 8),
            ignore_eos=True,
            timeout=timeout_s,
        ):
            if kind == "error":
                print(f"warmup {i} failed: {value}", flush=True)

    # Compute Poisson offsets (same logic as HydraServe)
    offsets: list[float] = []
    if arrival_pattern == "burst":
        offsets = [0.0] * len(indexed)
    elif arrival_pattern == "fixed":
        if request_rate is None:
            raise ValueError("fixed/poisson arrivals require --request-rate")
        offsets = [i / request_rate for i in range(len(indexed))]
    else:  # poisson
        if request_rate is None:
            raise ValueError("poisson arrivals require --request-rate")
        rng = Random(seed)
        elapsed = 0.0
        for i in range(len(indexed)):
            if i:
                elapsed += rng.expovariate(request_rate)
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
            index, entry = item
            trace_offset = entry.arrival_offset_ms / 1000.0
            delay = wall_started + offset + trace_offset - perf_counter()
            if delay > 0:
                sleep(delay)
            scheduled_at = wall_started + offset + trace_offset
            futures.append(executor.submit(_run_one, endpoint, item, scheduled_at, timeout_s))

        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = result

    wall_time = perf_counter() - wall_started
    results = tuple(r for r in ordered_results if r is not None)
    succeeded = tuple(r for r in results if r.error is None)

    return BenchmarkSummary(
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        wall_time_s=wall_time,
        request_throughput=len(succeeded) / wall_time if wall_time > 0 else 0,
        output_token_throughput=(
            sum(r.completion_tokens for r in succeeded) / wall_time
            if wall_time > 0
            else 0
        ),
        warmup_requests=warmup,
        offered_request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        ttft_ms=_percentiles(r.ttft_ms for r in succeeded if r.ttft_ms is not None),
        tpot_ms=_percentiles(r.tpot_ms for r in succeeded if r.tpot_ms is not None),
        latency_ms=_percentiles(r.latency_ms for r in succeeded),
        by_class=_by_class_stats(succeeded, wall_time),
        slo=_slo_stats(succeeded, wall_time),
        results=tuple(asdict(r) for r in results),
    )


def _run_one(
    endpoint: str,
    item: tuple[int, TraceEntry],
    scheduled_at: float,
    timeout_s: float,
) -> tuple[int, RequestMetrics]:
    index, entry = item
    started = perf_counter()
    client_queue_ms = max(0.0, (started - scheduled_at) * 1000)
    first_token_at = None
    last_token_at = None
    itl_ms: list[float] = []
    client_tokens = 0
    server_tokens: int | None = None
    finish_reason = "error"
    error = None

    for kind, value in stream_completion(
        endpoint,
        entry.prompt,
        entry.max_new_tokens,
        ignore_eos=entry.ignore_eos,
        timeout=timeout_s,
    ):
        now = perf_counter()
        if kind == "token":
            client_tokens += 1
            if first_token_at is None:
                first_token_at = now
            if last_token_at is not None:
                itl_ms.append((now - last_token_at) * 1000)
            last_token_at = now
        elif kind == "finish":
            finish_reason = value or "stop"
        elif kind == "usage":
            server_tokens = int(value)
        elif kind == "error":
            error = value

    ended = perf_counter()
    latency_ms = (ended - started) * 1000
    ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000
    e2e_ttft_ms = (
        None if first_token_at is None
        else max(0.0, (first_token_at - scheduled_at) * 1000)
    )
    # Authoritative output length from the server; client delta count drops
    # special tokens (stripped by skip_special_tokens), so it undercounts.
    completion_tokens = server_tokens if server_tokens is not None else client_tokens
    # TPOT = measured inter-token latency across actually-received deltas.
    tpot_ms = None
    if first_token_at is not None and last_token_at is not None and client_tokens > 1:
        tpot_ms = (last_token_at - first_token_at) * 1000 / (client_tokens - 1)

    return index, RequestMetrics(
        sample_id=entry.sample_id,
        prompt_tokens=entry.prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        error=error,
        e2e_ttft_ms=e2e_ttft_ms,
        client_queue_ms=client_queue_ms,
        itl_ms=tuple(itl_ms),
        klass=entry.klass,
        target_new_tokens=entry.max_new_tokens,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM DP using HydraServe frozen traces"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000",
                        help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--trace", required=True, type=Path,
                        help="HydraServe JSONL trace file")
    parser.add_argument("--output", type=Path,
                        help="output JSON path (default: stdout)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="max concurrent requests (default: 16)")
    parser.add_argument("--request-rate", type=float, default=2.0,
                        help="Poisson request rate in req/s (default: 2.0 = 0.8x)")
    parser.add_argument("--arrival-pattern",
                        choices=("burst", "fixed", "poisson"), default="poisson")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-request timeout in seconds")
    args = parser.parse_args()

    entries = load_trace(str(args.trace))
    print(f"Loaded {len(entries)} trace entries from {args.trace}", flush=True)
    longs = [e for e in entries if e.klass == "long"]
    shorts = [e for e in entries if e.klass == "short"]
    print(f"  long: {len(longs)}, short: {len(shorts)}", flush=True)
    if longs:
        print(f"  long prompt_tokens: {longs[0].prompt_tokens}, "
              f"long max_new_tokens: {longs[0].max_new_tokens}", flush=True)
    if shorts:
        print(f"  short prompt_tokens: {shorts[0].prompt_tokens}, "
              f"short max_new_tokens: {shorts[0].max_new_tokens}", flush=True)
    print(f"  concurrency={args.concurrency}, rate={args.request_rate}, "
          f"pattern={args.arrival_pattern}, seed={args.seed}", flush=True)

    summary = run_benchmark(
        args.endpoint,
        entries,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        arrival_pattern=args.arrival_pattern,
        seed=args.seed,
        warmup=args.warmup,
        timeout_s=args.timeout,
    )

    # Print summary
    s = summary
    print(f"{'='*60}", flush=True)
    print(f"Results: {s.succeeded}/{s.requests} succeeded, {s.failed} failed", flush=True)
    print(f"Throughput: {s.output_token_throughput:.1f} tok/s", flush=True)
    print(f"TPOT p50/p99: {s.tpot_ms.get('p50', 0):.0f}/{s.tpot_ms.get('p99', 0):.0f} ms", flush=True)
    print(f"TTFT p50/p99: {s.ttft_ms.get('p50', 0):.0f}/{s.ttft_ms.get('p99', 0):.0f} ms", flush=True)
    slo = s.slo
    print(f"Short SLO: {slo['met_requests']}/{slo['short_requests']} "
          f"({slo['met_rate']:.1%})", flush=True)
    print(f"  TTFT ok: {slo['ttft_ok']}, TPOT ok: {slo['tpot_ok']}, "
          f"full output: {slo['full_output_ok']}", flush=True)
    if s.by_class:
        for klass, stats in s.by_class.items():
            print(f"  {klass}: tpot_p50={stats.get('tpot_p50', 0):.0f}, "
                  f"tpot_p99={stats.get('tpot_p99', 0):.0f}", flush=True)
    print(f"{'='*60}", flush=True)

    output = json.dumps(s.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved to {args.output}", flush=True)
    else:
        print(output)

    return int(summary.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())

