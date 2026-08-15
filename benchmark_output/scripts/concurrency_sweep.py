#!/usr/bin/env python3
"""DP (collocated) vs PD (static 1P+1D) sweep across concurrency at a fixed context.

Measures TPOT P99 / throughput / TTFT under load -- the regime where PD's
prefill/decode isolation is supposed to pay off.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

MODEL = "/root/autodl-tmp/Qwen3.5-4B"
DATA = "/tmp/csweep_data"
GPUS = ["cuda:1", "cuda:2", "cuda:3"]
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "1"}


def build_cmd(mode, ctx, concurrency, gpus, out, limit, max_new, chunk):
    # KV cache sized for `concurrency` in-flight requests + headroom.
    cache = max(65536, ctx * (concurrency + 4) + 8192)
    cmd = [
        sys.executable, "-m", "hydraserve", "benchmark", MODEL, DATA,
        "--dataset", "sharegpt", "--limit", str(limit),
        "--max-prompt-tokens", str(ctx), "--max-new-tokens", str(max_new),
        "--concurrency", str(concurrency), "--warmup", "8",
        "--arrival-pattern", "burst",
        "--cache-tokens", str(cache), "--kv-headroom-blocks", "128",
        "--prefill-chunk-size", str(chunk),
    ]
    if mode == "pd":
        cmd += ["--pd", "--device", gpus[0], "--decode-device", gpus[1]]
    else:
        cmd += ["--device", gpus[0]]
    cmd += ["--output", out]
    return cmd


def run_one(mode, ctx, concurrency, gpus, out_dir, limit, max_new, chunk):
    tag = f"{mode}_ctx{ctx}_c{concurrency}"
    out = f"{out_dir}/{tag}.json"
    cmd = build_cmd(mode, ctx, concurrency, gpus, out, limit, max_new, chunk)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, **ENV})
    dt = time.time() - t0
    if proc.returncode != 0 or not Path(out).exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        return {"mode": mode, "ctx": ctx, "concurrency": concurrency,
                "error": "\n".join(tail), "wall_s": round(dt, 1)}
    d = json.load(open(out))
    d["mode"] = mode
    d["ctx"] = ctx
    d["concurrency"] = concurrency
    d["run_wall_s"] = round(dt, 1)
    return d


def pct(d, key, p):
    v = d.get(key, {})
    return v.get(f"p{p}", float("nan")) if isinstance(v, dict) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--out-dir", default="/tmp/csweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = args.context
    results = []

    # DP: 1 GPU each, 2-way parallel (leave 2 GPUs for a concurrent PD run)
    dp_tasks = [(c, [GPUS[i % 2]]) for i, c in enumerate(args.concurrencies)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, "dp", args.context, c, g, str(out_dir),
                          args.limit, args.max_new_tokens, chunk)
                for c, g in dp_tasks]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"DP c={r['concurrency']} ERROR: {r['error'][-200:]}", flush=True)

    # PD: 2 GPUs each, 1-way (serial) to avoid GPU contention
    pd_tasks = [(c, [GPUS[1], GPUS[2]]) for c in args.concurrencies]
    for c, g in pd_tasks:
        r = run_one("pd", args.context, c, g, str(out_dir),
                    args.limit, args.max_new_tokens, chunk)
        results.append(r)
        if "error" in r:
            print(f"PD c={r['concurrency']} ERROR: {r['error'][-200:]}", flush=True)

    print(f"\n=== DP vs PD (context={args.context}, max_new={args.max_new_tokens}) ===")
    print(f"{'conc':>5} {'DP TPOT99':>11} {'PD TPOT99':>11} {'DP TPOT50':>11} "
          f"{'PD TPOT50':>11} {'DP tok/s':>10} {'PD tok/s':>10} {'DP TTFT50':>10} {'PD TTFT50':>10}")
    by = {(r["mode"], r["concurrency"]): r for r in results}
    for c in args.concurrencies:
        dp = by.get(("dp", c), {})
        pd = by.get(("pd", c), {})
        if "error" in dp or "error" in pd:
            print(f"{c:>5}  (error on one arm)")
            continue
        print(f"{c:>5} {pct(dp,'tpot_ms',99):>11.1f} {pct(pd,'tpot_ms',99):>11.1f} "
              f"{pct(dp,'tpot_ms',50):>11.1f} {pct(pd,'tpot_ms',50):>11.1f} "
              f"{dp.get('output_token_throughput',0):>10.1f} {pd.get('output_token_throughput',0):>10.1f} "
              f"{pct(dp,'ttft_ms',50):>10.0f} {pct(pd,'ttft_ms',50):>10.0f}")


if __name__ == "__main__":
    main()
