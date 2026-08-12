"""
Qwen3.6-27B FP8 benchmark via vLLM API (TP=2).

Tests: TTFT vs context, TPOT vs batch, throughput, generation quality.
This is the real 27B + TP=2 baseline (main.md config C).
"""

import json, time, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import urllib.request

BASE = "http://localhost:8001"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def chat(prompt, max_tokens=100, stream=False):
    """Send chat completion request."""
    payload = json.dumps({
        "model": "/models/Qwen3.6-27B-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    usage = data["usage"]
    return {
        "ttft_ms": None,  # non-stream: no TTFT
        "total_ms": elapsed * 1000,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "tpot_ms": elapsed * 1000 / max(1, usage["completion_tokens"]),
        "text": data["choices"][0]["message"]["content"][:150],
    }

print("=" * 70)
print("BENCHMARK: Qwen3.6-27B FP8 via vLLM (TP=2)")
print("=" * 70)

# ── Test 1: Context sweep (TTFT proxy: total time for 1 token) ──
print("\n[1] Context sweep (max_tokens=1, measure prefill)...")
ctx_results = []
for ctx_tokens in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    prompt = "The quick brown fox jumps over the lazy dog. " * (ctx_tokens // 10)
    try:
        r = chat(prompt, max_tokens=1)
        # prefill time ≈ total_ms - 1 token decode
        prefill_ms = r["total_ms"] - r["tpot_ms"]
        ctx_results.append({
            "ctx": ctx_tokens, "prefill_ms": round(prefill_ms, 0),
            "total_ms": round(r["total_ms"], 0),
            "tokens_per_s": round(ctx_tokens / max(0.001, prefill_ms) * 1000, 0),
        })
        print(f"  {ctx_tokens:6d} tok: prefill={prefill_ms:.0f}ms "
              f"({ctx_tokens/max(0.001,prefill_ms)*1000:.0f} tok/s)")
    except Exception as e:
        print(f"  {ctx_tokens:6d}: {e}")
        break

# ── Test 2: Generation throughput ──
print("\n[2] Generation throughput (200 tokens)...")
gen_results = []
for label, prompt in [
    ("short", "Explain what machine learning is:"),
    ("medium", "Write a detailed explanation of quantum computing: "),
]:
    r = chat(prompt, max_tokens=200)
    gen_results.append({
        "label": label, "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "total_ms": r["total_ms"], "tpot_ms": round(r["tpot_ms"], 1),
        "tok_per_s": round(r["completion_tokens"] / r["total_ms"] * 1000, 1),
        "text": r["text"],
    })
    print(f"  {label}: {r['completion_tokens']} tok in {r['total_ms']:.0f}ms "
          f"({r['completion_tokens']/r['total_ms']*1000:.0f} tok/s, "
          f"TPOT={r['tpot_ms']:.1f}ms)")
    print(f"    → {r['text'][:100]}...")

# ── Test 3: Concurrency sweep ──
print("\n[3] Concurrency sweep (parallel requests)...")
conc_results = []
for n_concurrent in [1, 2, 4, 8]:
    prompt = "What is the capital of France? " * 10
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        futures = [ex.submit(chat, prompt, 100) for _ in range(n_concurrent)]
        results = [f.result() for f in as_completed(futures)]
    wall = time.perf_counter() - t0
    total_tok = sum(r["completion_tokens"] for r in results)
    conc_results.append({
        "concurrency": n_concurrent,
        "wall_ms": round(wall * 1000, 0),
        "total_tokens": total_tok,
        "throughput_tok_s": round(total_tok / wall, 1),
        "avg_tpot_ms": round(statistics.mean([r["tpot_ms"] for r in results]), 1),
    })
    print(f"  concurrency={n_concurrent}: {total_tok} tok in {wall:.1f}s "
          f"({total_tok/wall:.1f} tok/s), avg TPOT={conc_results[-1]['avg_tpot_ms']}ms")

# ── Save ──
results = {
    "model": "/models/Qwen3.6-27B-FP8",
    "serving": "vLLM v0.19.1, TP=2, max-model-len 32768, enforce-eager",
    "context_sweep": ctx_results,
    "generation": gen_results,
    "concurrency": conc_results,
}
with open(f"{OUTPUT_DIR}/bench_27b_vllm.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults: {OUTPUT_DIR}/bench_27b_vllm.json")
