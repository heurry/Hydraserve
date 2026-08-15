"""Merge the DP-arm JSONs (one per collocated process) and compare with the PD arm.

DP 基线的 4 个进程各自输出一份 benchmark JSON；本脚本把它们合并为与 PD arm
同口径的汇总（wall time 取 max，分位数在合并后的逐请求样本上重算，重算方式
与 runner 的线性插值一致），并打印对比表、写出合并 JSON。

用法：
  python3 scripts/cloud-4gpu/02_merge_results.py \
    --name sharegpt_r16 \
    --dp out/dp_sharegpt_r16_gpu*.json \
    --pd out/pd_sharegpt_r16.json
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path


def _percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}

    def value(percentile):
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {name: round(value(p), 1) for name, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}


def summarize(label, files):
    """返回与 runner 同口径的汇总字典；files 为同一 arm 的多个 JSON。"""
    merged = {"requests": 0, "succeeded": 0, "failed": 0, "wall_time_s": 0.0,
              "warmup_requests": 0, "results": [], "route_counts": {}}
    for path in files:
        data = json.loads(Path(path).read_text())
        merged["requests"] += data["requests"]
        merged["succeeded"] += data["succeeded"]
        merged["failed"] += data["failed"]
        merged["wall_time_s"] = max(merged["wall_time_s"], data["wall_time_s"])
        merged["warmup_requests"] += data["warmup_requests"]
        merged["results"].extend(data["results"])
        for route, count in data.get("route_counts", {}).items():
            merged["route_counts"][route] = merged["route_counts"].get(route, 0) + count
    succeeded = [r for r in merged["results"] if r.get("error") is None]
    divisor = merged["wall_time_s"] if merged["wall_time_s"] > 0 else float("inf")
    tokens = sum(r["completion_tokens"] for r in succeeded)
    prompt_tokens = [r["prompt_tokens"] for r in succeeded]
    return {
        "label": label,
        "requests": merged["requests"],
        "succeeded": len(succeeded),
        "failed": merged["failed"],
        "wall_time_s": round(merged["wall_time_s"], 1),
        "request_throughput": round(len(succeeded) / divisor, 3),
        "output_token_throughput": round(tokens / divisor, 2),
        "ttft_ms": _percentiles(r["ttft_ms"] for r in succeeded if r["ttft_ms"] is not None),
        "tpot_ms": _percentiles(r["tpot_ms"] for r in succeeded if r["tpot_ms"] is not None),
        "latency_ms": _percentiles(r["latency_ms"] for r in succeeded),
        "prompt_tokens": _percentiles(prompt_tokens),
        "route_counts": merged["route_counts"],
    }


def print_table(name, dp, pd):
    metrics = [
        ("requests", "requests", "{}"),
        ("failed", "failed", "{}"),
        ("request/s", "request_throughput", "{}"),
        ("output tok/s", "output_token_throughput", "{}"),
        ("TTFT p50/p95/p99", "ttft_ms", "{p50} / {p95} / {p99}"),
        ("TPOT p50/p95/p99", "tpot_ms", "{p50} / {p95} / {p99}"),
        ("Lat p50/p95/p99", "latency_ms", "{p50} / {p95} / {p99}"),
        ("prompt tok p50", "prompt_tokens", "{p50}"),
    ]
    print(f"\n### {name}")
    print("| 指标 | 4× collocated (DP) | 1P+3D (PD) |")
    print("|---|---|---|")
    for label, key, fmt in metrics:
        d = dp[key]
        p = pd[key]
        df = fmt.format(**d) if isinstance(d, dict) else fmt.format(d)
        pf = fmt.format(**p) if isinstance(p, dict) else fmt.format(p)
        print(f"| {label} | {df} | {pf} |")
    print(f"| routes | {dp['route_counts'] or 'n/a'} | {pd['route_counts'] or 'n/a'} |")
    if dp["wall_time_s"] and pd["wall_time_s"]:
        print(f"| wall time (s) | {dp['wall_time_s']} | {pd['wall_time_s']} |")
    if pd["failed"]:
        print("> 注意：PD arm 有失败请求，对比时先检查失败原因。")
    if dp["failed"]:
        print("> 注意：DP arm 有失败请求，对比时先检查失败原因。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dp", nargs="+", required=True)
    parser.add_argument("--pd", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    missing = [p for p in [*args.dp, args.pd] if not Path(p).exists()]
    if missing:
        print(f"缺少结果文件：{missing}", file=sys.stderr)
        return 1
    dp = summarize("dp", args.dp)
    pd = summarize("pd", [args.pd])
    print_table(args.name, dp, pd)
    out = Path(args.out) if args.out else Path(args.pd).with_suffix(".merged.json")
    out.write_text(json.dumps({"name": args.name, "dp": dp, "pd": pd},
                              ensure_ascii=False, indent=2) + "\n")
    print(f"合并结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
