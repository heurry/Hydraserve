#!/usr/bin/env python3
"""4xDP (collocated) vs 1P+3D (adaptive) on a mixed short/long workload (128K long)."""
import json
import subprocess
import sys
from pathlib import Path

MODEL = "/root/autodl-tmp/Qwen3.5-4B"
DATA = "/tmp/mixed_data"
OUT = Path("/tmp/mixed")
MAX_PROMPT = 131072
MAX_NEW = 128
CHUNK = 131072
CACHE = 140000

OUT.mkdir(parents=True, exist_ok=True)


def split_dataset():
    d = json.load(open(f"{DATA}/sharegpt.json"))
    groups = [[] for _ in range(4)]
    for i, r in enumerate(d):
        groups[i % 4].append(r)
    for g, records in enumerate(groups):
        sub = OUT / f"split{g}"
        sub.mkdir(exist_ok=True)
        json.dump(records, open(sub / "sharegpt.json", "w"))
    return len(d), [len(g) for g in groups]


def summarize(path):
    d = json.load(open(path))
    return (d.get("succeeded"), d.get("failed"), d.get("output_token_throughput", 0),
            d.get("ttft_ms", {}).get("p50", -1), d.get("tpot_ms", {}).get("p50", -1),
            d.get("tpot_ms", {}).get("p99", -1), d.get("route_counts", {}))


def main():
    total, groups = split_dataset()
    print(f"dataset: {total} records -> 4 groups {groups}", flush=True)

    print("\n=== DP arm: 4x collocated ===", flush=True)
    procs = []
    for gpu in range(4):
        cmd = [sys.executable, "-m", "hydraserve", "benchmark", MODEL, str(OUT / f"split{gpu}"),
               "--dataset", "sharegpt", "--limit", str(groups[gpu]-1),
               "--max-prompt-tokens", str(MAX_PROMPT), "--max-new-tokens", str(MAX_NEW),
               "--concurrency", "4", "--warmup", "1", "--arrival-pattern", "burst",
               "--cache-tokens", str(CACHE), "--kv-headroom-blocks", "128",
               "--prefill-chunk-size", str(CHUNK), "--device", f"cuda:{gpu}",
               "--output", str(OUT / f"dp_gpu{gpu}.json")]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for gpu, p in enumerate(procs):
        p.wait()
        print(f"  gpu{gpu} done rc={p.returncode}", flush=True)

    print("\n=== PD arm: 1P+3D adaptive ===", flush=True)
    cmd = [sys.executable, "-m", "hydraserve", "benchmark", MODEL, DATA,
           "--dataset", "sharegpt", "--limit", str(total-1),
           "--max-prompt-tokens", str(MAX_PROMPT), "--max-new-tokens", str(MAX_NEW),
           "--concurrency", "16", "--warmup", "1", "--arrival-pattern", "burst",
           "--cache-tokens", str(CACHE), "--kv-headroom-blocks", "128",
           "--prefill-chunk-size", str(CHUNK),
           "--adaptive", "--device", "cuda:0", "--decode-devices", "cuda:1", "cuda:2", "cuda:3",
           "--output", str(OUT / "pd_1p3d.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  pd done rc={r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-500:])

    print("\n=== 结果 ===")
    for gpu in range(4):
        s = summarize(OUT / f"dp_gpu{gpu}.json")
        print(f"DP gpu{gpu}: succ {s[0]} tok/s {s[2]:.1f} TTFT50 {s[3]:.0f} "
              f"TPOT50 {s[4]:.1f} TPOT99 {s[5]:.1f}")
    s = summarize(OUT / "pd_1p3d.json")
    print(f"PD 1P3D: succ {s[0]} tok/s {s[2]:.1f} TTFT50 {s[3]:.0f} "
          f"TPOT50 {s[4]:.1f} TPOT99 {s[5]:.1f} routes {s[6]}")


if __name__ == "__main__":
    main()
