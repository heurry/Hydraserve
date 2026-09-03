#!/usr/bin/env python3
"""Side-by-side DP vs H1 V5 result comparison (BENCHMARK_PLAN_V5.md section 8.1).

Usage:
  python scripts/compare_dp_h1.py --dp FILE.json --h1 FILE.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def pct(d, klass, metric, q):
    return d["by_class"][klass][metric][q]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", required=True)
    ap.add_argument("--h1", required=True)
    args = ap.parse_args()
    dp, h1 = load(args.dp), load(args.h1)

    print(f"{'metric':34s} {'DP':>12s} {'H1':>12s}")
    for klass in ("short", "long"):
        for metric in ("e2e_ttft_ms", "tpot_ms"):
            for q in ("p50", "p95", "p99"):
                try:
                    a, b = pct(dp, klass, metric, q), pct(h1, klass, metric, q)
                except KeyError:
                    continue
                print(f"{klass} {metric} {q:>4s}      {a:12.1f} {b:12.1f}")

    for d, tag in ((dp, "DP"), (h1, "H1")):
        s = d["slo"]
        print(
            f"\n{tag}: {d['succeeded']}/{d['requests']} ok | short SLO "
            f"{s.get('met_requests')}/{s.get('short_requests')} "
            f"(goodput {s.get('goodput_tokens_s')} tok/s) | long "
            f"{s.get('long_met_requests')}/{s.get('long_requests')} "
            f"({s.get('long_goodput_tokens_s')} tok/s) | total "
            f"{d['output_token_throughput']} tok/s"
        )
        print(
            f"     admission wait p50/p99/max = "
            f"{s.get('admission_wait_ms', {}).get('p50', 0):.0f}/"
            f"{s.get('admission_wait_ms', {}).get('p99', 0):.0f}/"
            f"{s.get('admission_wait_ms', {}).get('max', 0):.0f} ms | "
            f"starved={s.get('starved_requests')} overflow={s.get('overflow_count')}"
        )

    # V5 section 8.1 checks (per-run view; caller aggregates across 3 seeds).
    s1, s0 = h1["slo"], dp["slo"]
    checks = [
        ("h1 all requests ok", h1["failed"] == 0),
        ("short SLO goodput higher", s1["goodput_tokens_s"] > s0["goodput_tokens_s"]),
        ("short goodput +10%", s1["goodput_tokens_s"] >= s0["goodput_tokens_s"] * 1.10),
        ("short TTFT p50 improved", pct(h1, "short", "e2e_ttft_ms", "p50") < pct(dp, "short", "e2e_ttft_ms", "p50")),
        ("short TPOT p50 improved", pct(h1, "short", "tpot_ms", "p50") < pct(dp, "short", "tpot_ms", "p50")),
        ("total tok >= 90% DP", h1["output_token_throughput"] >= dp["output_token_throughput"] * 0.90),
        ("long no starvation (>30s)", s1["starved_requests"] == 0),
        ("long admission <= 30s max", s1.get("admission_wait_ms", {}).get("max", 0) <= 30000),
    ]
    print("\nV5 8.1 checks (single run):")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
