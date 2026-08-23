#!/usr/bin/env python3
"""V3 P0.5 four-GPU gate check (BENCHMARK_PLAN_V3.md section 3.5).

Runs the ten gates that must pass before any V3 M1 matrix numbers are taken:
  1. ignore_eos=true produces the full target output (finish_reason=length)
  2. 4xDP per-worker balance (8 identical requests -> 2 per GPU) + worker logs
  3. all workers show decode activity (cross-GPU parallelism sanity)
  4. interleaved requests: no PartialDecodeError / token misalignment
  5. trace hash verification (record/token_ids/prompt/trace SHA256 + .meta.json)
  6. --trace replay uses --arrival-pattern burst and no --request-rate
  7. --warmup uses separate synthetic warmup; results.requests == trace rows
  8. --cache-tokens 131072 + memory/health checks after run
  9. same trace, same seed, 4xDP and 2P+2D -> identical metadata.trace_sha256
 10. (informational) run templates are the V3 4.3 commands

Requires a 4xRTX3090 machine with a working NVIDIA driver. Exits 2 with a
message if no CUDA device is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

MODEL = pathlib.Path("/root/autodl-tmp/Qwen3.5-4B")
DATASETS = pathlib.Path("/root/autodl-tmp/data")

RESULTS: list[tuple[str, str, str]] = []  # (gate, status, detail)


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, "PASS" if ok else "FAIL", detail))
    print(f"[{gate}] {'PASS' if ok else 'FAIL'}  {detail}", flush=True)


def run_cmd(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def summary_of(out_path: pathlib.Path) -> dict:
    return json.loads(out_path.read_text(encoding="utf-8"))


def gen_trace(entries_spec: list[dict], path: pathlib.Path, model: pathlib.Path, seed: int = 42) -> dict:
    """Generate a frozen trace via --trace-out (synthetic only, tokenizer-only)."""
    cmd = [
        sys.executable, "-m", "hydraserve", "benchmark", str(model), str(DATASETS),
        "--dataset", "synthetic", "--trace-out", str(path), "--seed", str(seed),
    ]
    # Build the synthetic spec from entries_spec: {long:[(tokens,max_new)], short:[...]}
    longs = entries_spec.get("long", [])
    shorts = entries_spec.get("short", [])
    if longs:
        cmd += ["--num-long", str(len(longs)), "--long-tokens", str(longs[0][0]),
                "--long-new-tokens", str(longs[0][1])]
    if shorts:
        cmd += ["--num-short", str(len(shorts)), "--short-tokens", str(shorts[0][0]),
                "--short-new-tokens", str(shorts[0][1])]
    r = run_cmd(cmd, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"trace generation failed: {r.stderr[-800:]}")
    return json.loads(r.stdout)


def bench(trace: pathlib.Path, out: pathlib.Path, logdir: pathlib.Path, model: pathlib.Path,
          *, workers: list[str], adaptive: bool = False, concurrency: int = 16,
          warmup: int = 2, cache_tokens: int = 131072, extra: list[str] | None = None,
          seed: int = 42, timeout: int = 1800) -> dict:
    cmd = [
        sys.executable, "-m", "hydraserve", "benchmark", str(model), str(DATASETS),
        "--dataset", "synthetic", "--trace", str(trace),
        "--concurrency", str(concurrency), "--warmup", str(warmup),
        "--arrival-pattern", "burst",
        "--kv-quant", "int8", "--prefix-cache-blocks", "0",
        "--cache-tokens", str(cache_tokens),
        "--block-size", "256",
        "--worker-log-dir", str(logdir),
        "--output", str(out), "--seed", str(seed),
    ]
    if adaptive:
        split = len(workers) // 2
        if split == 0 or split == len(workers):
            raise ValueError("adaptive gate requires at least one P and one D worker")
        cmd += ["--adaptive", "--force-pd-tokens", "1",
                "--prefill-devices", *workers[:split],
                "--decode-devices", *workers[split:]]
    else:
        cmd += ["--dp-devices", *workers]
    if extra:
        cmd += extra
    r = run_cmd(cmd, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"benchmark failed (rc={r.returncode}): {r.stderr[-1200:]}")
    return summary_of(out)


def gate1_ignore_eos(tmp: pathlib.Path, model: pathlib.Path) -> None:
    trace = tmp / "g1_trace.jsonl"
    gen_trace({"short": [(2048, 128)] * 4}, trace, model)
    out = tmp / "g1.json"
    logdir = tmp / "g1_logs"
    s = bench(trace, out, logdir, model, workers=["0"])
    bad = []
    for r in s["results"]:
        if r.get("error"):
            bad.append(f"{r['sample_id']}:error={r['error']}")
        elif r.get("finish_reason") != "length" or r.get("completion_tokens") != 128:
            bad.append(
                f"{r['sample_id']}:finish={r.get('finish_reason')} "
                f"tokens={r.get('completion_tokens')}"
            )
    record("G1", not bad, f"{len(s['results'])} reqs full-output (finish=length, 128 tok): "
                          f"{'clean' if not bad else '; '.join(bad[:3])}")


def gate2_dp_balance(tmp: pathlib.Path, model: pathlib.Path, workers: list[str]) -> None:
    trace = tmp / "g2_trace.jsonl"
    gen_trace({"short": [(1024, 32)] * 8}, trace, model)
    out = tmp / "g2.json"
    logdir = tmp / "g2_logs"
    s = bench(trace, out, logdir, model, workers=workers, concurrency=8)
    pw = s["per_worker"]
    per = {k: v["requests"] for k, v in pw.items()}
    expected = 8 // len(workers)
    balanced = len(pw) == len(workers) and all(v == expected for v in per.values())
    logs = [p for p in logdir.glob("dp-*.log")]
    logs_nonempty = [p for p in logs if p.stat().st_size > 0]
    record(
        "G2",
        balanced and len(logs_nonempty) == len(workers),
        f"per_worker={per} (expect {expected}/card); worker logs {len(logs_nonempty)}/{len(workers)} non-empty",
    )


def gate3_decode_parallelism(tmp: pathlib.Path, model: pathlib.Path, workers: list[str]) -> None:
    # Sanity: every worker log shows decode activity under a mixed burst. True
    # overlap needs Nsight/NCCL timing; we at least require all workers active.
    trace = tmp / "g3_trace.jsonl"
    gen_trace({"short": [(2048, 64)] * 16}, trace, model)
    out = tmp / "g3.json"
    logdir = tmp / "g3_logs"
    s = bench(trace, out, logdir, model, workers=workers, concurrency=16)
    logs = [p for p in logdir.glob("dp-*.log")]
    active = sum(1 for p in logs if p.stat().st_size > 0)
    record(
        "G3",
        active == len(workers) and s["succeeded"] == 16,
        f"{active}/{len(workers)} workers active, {s['succeeded']}/16 succeeded "
        f"(confirm real overlap with Nsight if needed)",
    )


def gate4_interleaved(tmp: pathlib.Path, model: pathlib.Path, workers: list[str]) -> None:
    trace = tmp / "g4_trace.jsonl"
    gen_trace({"short": [(2048, 64)] * 8}, trace, model)
    out = tmp / "g4.json"
    logdir = tmp / "g4_logs"
    s = bench(trace, out, logdir, model, workers=workers, concurrency=4, warmup=0)
    errs = {r["sample_id"]: r["error"] for r in s["results"] if r.get("error")}
    misaligned = any("PartialDecodeError" in (e or "") for e in errs.values())
    ok = s["failed"] == 0 and not misaligned
    record("G4", ok, f"8 interleaved reqs, failed={s['failed']}, "
                     f"partial-decode={'YES' if misaligned else 'no'}, "
                     f"{'clean' if ok else '; '.join(str(v)[:80] for v in list(errs.values())[:2])}")


def gate5_trace_hash(tmp: pathlib.Path, model: pathlib.Path) -> None:
    from hydraserve.benchmark.datasets import iter_trace, write_trace
    from hydraserve.model import QwenTokenizer

    tok = QwenTokenizer(str(model))
    trace = tmp / "g5_trace.jsonl"
    entries = [
        {"id": "s0", "class": "short", "prompt_tokens": 128, "max_new_tokens": 32,
         "ignore_eos": True, "seed": 42},
        {"id": "s1", "class": "short", "prompt_tokens": 128, "max_new_tokens": 32,
         "ignore_eos": True, "seed": 42},
    ]
    meta = write_trace(tok, entries, trace, seed=42)
    samples = list(iter_trace(tok, trace, seed=42))  # verifies record/token_ids/prompt hashes
    meta_path = trace.with_suffix(trace.suffix + ".meta.json")
    meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
    # Recompute trace sha256 of the file bytes.
    file_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    ok = meta["trace_sha256"] == file_sha and len(samples) == 2 and meta_path.is_file()
    record("G5", ok, f"trace_sha256 match={meta['trace_sha256'] == file_sha}, "
                     f"2 samples replayed, .meta.json present={meta_path.is_file()}")


def gate6_burst_no_rate(tmp: pathlib.Path, model: pathlib.Path) -> None:
    trace = tmp / "g6_trace.jsonl"
    gen_trace({"short": [(512, 16)] * 4}, trace, model)
    out = tmp / "g6.json"
    logdir = tmp / "g6_logs"
    # V3 rule: replay with burst, never pass --request-rate.
    s = bench(trace, out, logdir, model, workers=["0"], concurrency=4, warmup=1)
    ok = s["arrival_pattern"] == "burst" and s["offered_request_rate"] is None
    record("G6", ok, f"arrival_pattern={s['arrival_pattern']}, "
                     f"offered_request_rate={s['offered_request_rate']} (must be None)")


def gate7_warmup_independent(tmp: pathlib.Path, model: pathlib.Path) -> None:
    trace = tmp / "g7_trace.jsonl"
    gen_trace({"short": [(512, 16)] * 6}, trace, model)
    out = tmp / "g7.json"
    logdir = tmp / "g7_logs"
    s = bench(trace, out, logdir, model, workers=["0"], concurrency=4, warmup=3)
    ok = s["requests"] == 6 and s["warmup_requests"] == 3
    record("G7", ok, f"results.requests={s['requests']} (== trace rows 6), "
                     f"warmup_requests={s['warmup_requests']} (independent)")


def gate8_memory(tmp: pathlib.Path, model: pathlib.Path, workers: list[str]) -> None:
    trace = tmp / "g8_trace.jsonl"
    gen_trace({"short": [(2048, 64)] * 12}, trace, model)
    out = tmp / "g8.json"
    logdir = tmp / "g8_logs"
    s = bench(trace, out, logdir, model, workers=workers, concurrency=12,
              cache_tokens=131072)
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    lines = [ln for ln in nvidia.stdout.strip().splitlines() if ln]
    oom = any("out of memory" in (r.get("error") or "").lower() for r in s["results"])
    ok = s["failed"] == 0 and not oom and len(lines) == len(workers)
    record("G8", ok, f"failed={s['failed']}, oom={oom}; nvidia-smi:\n" + "\n".join(f"    {l}" for l in lines))


def gate9_same_trace_dp_pd(tmp: pathlib.Path, model: pathlib.Path, workers: list[str]) -> None:
    trace = tmp / "g9_trace.jsonl"
    gen_trace({"short": [(1024, 32)] * 8}, trace, model)
    out_d0 = tmp / "g9_d0.json"
    log_d0 = tmp / "g9_d0_logs"
    s_d0 = bench(trace, out_d0, log_d0, model, workers=workers, concurrency=8)
    out_p0 = tmp / "g9_p0.json"
    log_p0 = tmp / "g9_p0_logs"
    s_p0 = bench(trace, out_p0, log_p0, model, workers=workers,
                 adaptive=True, concurrency=8)
    h_d0 = s_d0["metadata"].get("trace_sha256")
    h_p0 = s_p0["metadata"].get("trace_sha256")
    ok = h_d0 and h_d0 == h_p0
    record("G9", ok, f"D0 trace_sha256={h_d0[:12] if h_d0 else None} == "
                     f"P0 trace_sha256={h_p0[:12] if h_p0 else None}")


def main() -> int:
    global DATASETS
    parser = argparse.ArgumentParser(description="V3 P0.5 four-GPU gate check")
    parser.add_argument("--model", type=pathlib.Path, default=MODEL)
    parser.add_argument("--datasets", type=pathlib.Path, default=DATASETS)
    parser.add_argument("--workers", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--only", nargs="+", help="e.g. G1 G2")
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/v3_gates"))
    args = parser.parse_args()
    DATASETS = args.datasets

    try:
        import torch
        if not torch.cuda.is_available():
            print("GATE-SKIP: no CUDA device; this script must run on the 4x3090 machine.", flush=True)
            return 2
    except Exception as exc:
        print(f"GATE-SKIP: cannot probe CUDA ({exc})", flush=True)
        return 2

    tmp = args.workdir
    tmp.mkdir(parents=True, exist_ok=True)
    model = args.model
    workers = args.workers
    selected = set(args.only) if args.only else None

    def run_gate(name, fn):
        if selected and name not in selected:
            return
        print(f"\n=== {name} ===", flush=True)
        try:
            import inspect

            if len(inspect.signature(fn).parameters) >= 3:
                fn(tmp, model, workers)
            else:
                fn(tmp, model)
        except Exception as exc:
            record(name, False, f"exception: {type(exc).__name__}: {exc}")

    run_gate("G1", gate1_ignore_eos)
    run_gate("G2", gate2_dp_balance)
    run_gate("G3", gate3_decode_parallelism)
    run_gate("G4", gate4_interleaved)
    run_gate("G5", gate5_trace_hash)
    run_gate("G6", gate6_burst_no_rate)
    run_gate("G7", gate7_warmup_independent)
    run_gate("G8", gate8_memory)
    run_gate("G9", gate9_same_trace_dp_pd)

    print("\n=== SUMMARY ===")
    for gate, status, _ in RESULTS:
        print(f"  {gate}: {status}")
    failed = sum(1 for _, status, _ in RESULTS if status == "FAIL")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} gates passed.")
    print("G10: run templates are the V3 4.3 commands (MODEL/DATASETS/SHORT_RATE/"
          "FROZEN_CHUNK/FROZEN_STEP must be substituted).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
