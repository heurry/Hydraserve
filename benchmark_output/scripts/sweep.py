#!/usr/bin/env python3
"""DP (collocated) vs PD (static 1P+1D) sweep across context lengths.

Parallelizes over the 4 GPUs: DP runs are 1-GPU (4-way parallel),
PD runs are 2-GPU (2-way parallel). No adaptive routing -- each mode runs
cleanly on its own at the same (context, concurrency).
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
DATA = "/tmp/stress_data"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def build_cmd(mode: str, ctx: int, concurrency: int, gpus: list[str], out: str,
              chunk_size: int) -> list[str]:
    cache = max(65536, ctx + 8192)
    cmd = [
        sys.executable, "-m", "hydraserve", "benchmark", MODEL, DATA,
        "--dataset", "sharegpt", "--limit", "1",
        "--max-prompt-tokens", str(ctx), "--max-new-tokens", "32",
        "--concurrency", str(concurrency), "--warmup", "1",
        "--arrival-pattern", "burst",
        "--cache-tokens", str(cache), "--kv-headroom-blocks", "128",
        "--prefill-chunk-size", str(chunk_size),
    ]
    if mode == "pd":
        cmd += ["--pd", "--device", gpus[0], "--decode-device", gpus[1]]
    else:
        cmd += ["--device", gpus[0]]
    cmd += ["--output", out]
    return cmd


def run_one(mode: str, ctx: int, concurrency: int, gpus: list[str], out_dir: str,
            chunk_size: int) -> dict:
    tag = f"{mode}_ctx{ctx}_c{concurrency}"
    out = f"{out_dir}/{tag}.json"
    cmd = build_cmd(mode, ctx, concurrency, gpus, out, chunk_size)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
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


def pct(d: dict, key: str, p: int) -> float:
    v = d.get(key, {})
    if isinstance(v, dict):
        return v.get(f"p{p}", float("nan"))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[4096, 8192, 16384, 32768, 65536, 131072, 262144])
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="prefill chunk size (0 = full context)")
    ap.add_argument("--out-dir", default="/tmp/sweep")
    args = ap.parse_args()

    def chunk_for(ctx: int) -> int:
        # Single-chunk prefill now works for 32K/64K/128K (grid fix + correct
        # cache-tokens); each context runs as one FA prefill, no continuation chunks.
        return args.chunk_size or ctx

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    # Phase 1: DP (1 GPU each, 4-way parallel)
    dp_tasks = [(ctx, [GPUS[i % 4]]) for i, ctx in enumerate(args.contexts)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_one, "dp", ctx, args.concurrency, g, str(out_dir), chunk_for(ctx))
                for ctx, g in dp_tasks]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
            r = results[-1]
            if "error" in r:
                print(f"DP ctx={r['ctx']} ERROR: {r['error']}", flush=True)

    # Phase 2: PD (2 GPU each, 2-way parallel)
    pairs = [[GPUS[0], GPUS[1]], [GPUS[2], GPUS[3]]]
    pd_tasks = [(ctx, pairs[i % 2]) for i, ctx in enumerate(args.contexts)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, "pd", ctx, args.concurrency, g, str(out_dir), chunk_for(ctx))
                for ctx, g in pd_tasks]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
            r = results[-1]
            if "error" in r:
                print(f"PD ctx={r['ctx']} ERROR: {r['error']}", flush=True)

    # Comparison table
    print("\n=== DP vs PD (concurrency=%d) ===" % args.concurrency)
    print(f"{'ctx':>8} {'DP TTFT50':>12} {'PD TTFT50':>12} {'DP TPOT50':>11} "
          f"{'PD TPOT50':>11} {'DP tok/s':>10} {'PD tok/s':>10}")
    by = {(r["mode"], r["ctx"]): r for r in results}
    for ctx in args.contexts:
        dp = by.get(("dp", ctx), {})
        pd = by.get(("pd", ctx), {})
        if "error" in dp or "error" in pd:
            print(f"{ctx:>8}  (error on one arm)")
            continue
        def tps(r):
            return r.get("output_token_throughput", float("nan"))
        print(f"{ctx:>8} {pct(dp,'ttft_ms',50):>12.0f} {pct(pd,'ttft_ms',50):>12.0f} "
              f"{pct(dp,'tpot_ms',50):>11.1f} {pct(pd,'tpot_ms',50):>11.1f} "
              f"{tps(dp):>10.1f} {tps(pd):>10.1f}")


if __name__ == "__main__":
    main()
