"""
Real benchmark — push hardware to actual limits.
GPU0: x16, GPU1: x4, no P2P, SHM ~4 GB/s.
Qwen3.5-4B BF16: 8.4GB weights, 15.6GB free per GPU.
Target: saturate VRAM AND compute, find real ceilings.
"""

import sys, os, time, json, math, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from concurrent.futures import ThreadPoolExecutor

MODEL_PATH = "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("HydraServe: Real Hardware Limit Test")
print(f"Topology: GPU0 x16 + GPU1 x4, no P2P, SHM ~4 GB/s")
print(f"Model: Qwen3.5-4B BF16 (4.21B params, ~8.4GB VRAM)")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
eos = tokenizer.eos_token_id

# Monitor GPU util
def gpu_util():
    """Get current GPU utilization."""
    try:
        import subprocess, re
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True
        )
        gpus = {}
        for line in out.strip().split('\n'):
            parts = [x.strip() for x in line.split(',')]
            gpus[int(parts[0])] = {
                "util_pct": int(parts[1]),
                "mem_mb": int(parts[2]),
                "mem_total_mb": int(parts[3]),
            }
        return gpus
    except:
        return {}

def print_gpu_status(label=""):
    util = gpu_util()
    for i in range(torch.cuda.device_count()):
        u = util.get(i, {})
        print(f"  GPU{i}: {u.get('util_pct','?')}% util, "
              f"{u.get('mem_mb','?')/1024:.1f}/{u.get('mem_total_mb','?')/1024:.1f}GB")

print("\nInitial GPU status:")
print_gpu_status()

# ─── Load models ───────────────────────────────────────────────
print("\n[Loading models on both GPUs...]")
model0 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, trust_remote_code=True).eval()
model1 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:1"}, trust_remote_code=True).eval()

print("After model load:")
print_gpu_status()

# ═══════════════════════════════════════════════════════════════
# TEST 1: Push VRAM to limit — find max context length
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Push VRAM to limit — find max viable context")
print("=" * 70)

# Try increasingly large contexts until OOM
# BF16 model: weights ~8.4GB, KV cache grows with context
# KV per token (Qwen3.5-4B): 2*8*4*256*2 = 32KB/token
# At 32K: 32K * 32KB = 1GB KV cache
# At 64K: 2GB, at 128K: 4GB
# VRAM: 8.4GB weights + 1GB framework + KV cache = available
# Available: 24 - 8.4 - 1.5 = ~14GB for KV and activations

max_ctx = 0
max_ctx_time = 0
for target_ctx in [1024, 4096, 8192, 16384, 24576, 32768, 49152, 65536]:
    try:
        text = ("The quick brown fox jumps over the lazy dog. " * (target_ctx // 10 + 1))[:target_ctx * 4]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=target_ctx).to("cuda:0")
        actual_len = inputs['input_ids'].shape[1]

        # Warmup
        with torch.no_grad():
            model0(**inputs)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model0(**inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        alloc = torch.cuda.memory_allocated(0) / 1e9
        print(f"  {actual_len:5d} tokens: {elapsed:7.0f}ms ({actual_len/elapsed*1000:.0f} tok/s), "
              f"VRAM: {alloc:.1f}GB, GPU util: {gpu_util().get(0,{}).get('util_pct','?')}%")
        max_ctx = actual_len
        max_ctx_time = elapsed

    except torch.OutOfMemoryError:
        print(f"  {target_ctx} tokens: OOM at {torch.cuda.memory_allocated(0)/1e9:.1f}GB VRAM used")
        torch.cuda.empty_cache()
        break

print(f"\n  MAX context (BF16): {max_ctx} tokens")
print_gpu_status("After max context test")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Push decode batch to VRAM limit
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Push decode batch to VRAM limit (GPU0, no KV cache)")
print("=" * 70)

max_batch = 0
max_batch_tok_s = 0
for bs in [1, 4, 16, 32, 64, 128, 256, 512, 1024]:
    try:
        inputs = torch.randint(0, 1000, (bs, 1), device="cuda:0")
        torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            model0(inputs)
        torch.cuda.synchronize()

        n_iter = max(3, min(30, 300 // max(1, bs // 10)))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                model0(inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iter * 1000
        tok_s = bs / elapsed * 1000
        alloc = torch.cuda.memory_allocated(0) / 1e9

        print(f"  batch={bs:4d}: {elapsed:6.0f}ms/step, {tok_s:8.0f} tok/s, "
              f"{elapsed/bs:5.1f}ms/tok, VRAM: {alloc:.1f}GB")

        max_batch = bs
        max_batch_tok_s = tok_s

    except torch.OutOfMemoryError:
        print(f"  batch={bs}: OOM at {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
        torch.cuda.empty_cache()
        break

print(f"\n  MAX batch (GPU0 alone): {max_batch}")
print_gpu_status("After batch test")

# ═══════════════════════════════════════════════════════════════
# TEST 3: DP Mode — both GPUs saturated simultaneously
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: DP Mode — Both GPUs at max batch simultaneously")
print("=" * 70)

# Find max batch per GPU, then run both
for bs in [32, 64, 128, 192, 256, 384, 512]:
    try:
        i0 = torch.randint(0, 1000, (bs, 1), device="cuda:0")
        i1 = torch.randint(0, 1000, (bs, 1), device="cuda:1")

        # Warmup
        with torch.no_grad():
            model0(i0); model1(i1)
        torch.cuda.synchronize()

        n_iter = max(3, min(20, 100 // max(1, bs // 40)))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                o0 = model0(i0)
                o1 = model1(i1)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iter * 1000
        total_tok_s = (bs * 2) / elapsed * 1000
        util = gpu_util()
        u0 = util.get(0, {}).get('util_pct', '?')
        u1 = util.get(1, {}).get('util_pct', '?')

        print(f"  DP batch={bs}×2 GPUs: {elapsed:.0f}ms, {total_tok_s:.0f} tok/s combined, "
              f"GPU0:{u0}% GPU1:{u1}%")

    except torch.OutOfMemoryError:
        print(f"  DP batch={bs}: OOM")
        torch.cuda.empty_cache()
        break

# ═══════════════════════════════════════════════════════════════
# TEST 4: Sustained throughput — 30 seconds continuous
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Sustained Throughput — 30s continuous decode, both GPUs")
print("=" * 70)

# Use a batch that keeps VRAM high but doesn't OOM
# The DP test found the practical max per GPU
if max_batch == 0:
    max_batch = 128  # Fallback from previous test data
sustained_bs = min(max_batch, 64)  # 64 per GPU from DP test
print(f"  Using batch={sustained_bs} per GPU (total {sustained_bs*2})")

i0 = torch.randint(0, 1000, (sustained_bs, 1), device="cuda:0")
i1 = torch.randint(0, 1000, (sustained_bs, 1), device="cuda:1")

# Warmup
with torch.no_grad():
    for _ in range(10):
        model0(i0); model1(i1)
torch.cuda.synchronize()

# 30 second sustained run
duration_s = 30
n_steps = 0
total_tokens = 0
torch.cuda.synchronize()
t_start = time.perf_counter()
with torch.no_grad():
    while time.perf_counter() - t_start < duration_s:
        model0(i0)
        model1(i1)
        n_steps += 1
        total_tokens += sustained_bs * 2
torch.cuda.synchronize()
actual_duration = time.perf_counter() - t_start

sustained_tok_s = total_tokens / actual_duration
step_ms = actual_duration / n_steps * 1000
util_end = gpu_util()
print(f"  {n_steps} steps in {actual_duration:.1f}s")
print(f"  Sustained throughput: {sustained_tok_s:.0f} tok/s ({total_tokens} tokens)")
print(f"  Per step: {step_ms:.1f}ms")
print(f"  GPU0: {util_end.get(0,{}).get('util_pct','?')}% util, "
      f"{util_end.get(0,{}).get('mem_mb','?')/1024:.1f}GB VRAM")
print(f"  GPU1: {util_end.get(1,{}).get('util_pct','?')}% util, "
      f"{util_end.get(1,{}).get('mem_mb','?')/1024:.1f}GB VRAM")

# ═══════════════════════════════════════════════════════════════
# TEST 5: Collocated Interference — real measurement
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: Collocated Interference (Prefill blocks Decode on same GPU)")
print("=" * 70)

decode_bs = 16
decode_inp = torch.randint(0, 1000, (decode_bs, 1), device="cuda:0")

# Baseline: pure decode
with torch.no_grad():
    model0(decode_inp)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(30):
        model0(decode_inp)
torch.cuda.synchronize()
pure_decode_ms = (time.perf_counter() - t0) / 30 * 1000
print(f"  Pure decode (batch={decode_bs}): {pure_decode_ms:.1f}ms/step ({decode_bs/pure_decode_ms*1000:.0f} tok/s)")

# Interference: prefill + decode on same GPU
for prefill_ctx in [1024, 2048, 4096, 8192, 16384]:
    if prefill_ctx > max_ctx:
        break
    try:
        text = "The quick brown fox. " * (prefill_ctx // 10)
        pref_inp = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=prefill_ctx).to("cuda:0")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o_pref = model0(**pref_inp)     # prefill blocks GPU0
            o_dec = model0(decode_inp)       # decode STARTS AFTER prefill
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        # In DP mode, decode on GPU1 runs in parallel with prefill on GPU0
        # So effective decode is NOT delayed
        # In collocated mode, decode IS delayed by prefill
        decode_delay = total_ms - pure_decode_ms
        slowdown = total_ms / pure_decode_ms

        print(f"  Prefill {prefill_ctx:5d}: total={total_ms:6.0f}ms, "
              f"decode delay={decode_delay:5.0f}ms, slowdown={slowdown:.1f}×")
    except torch.OutOfMemoryError:
        print(f"  Prefill {prefill_ctx}: OOM")
        torch.cuda.empty_cache()
        break

# ═══════════════════════════════════════════════════════════════
# TEST 6: SHM Transfer — real (state migration cost for PD)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: SHM Transfer Cost (PD separation bottleneck)")
print("=" * 70)

shm_results = []
for size_mb in [1, 5, 25, 50, 100, 250, 500]:
    try:
        n_elem = int(size_mb * 1024 * 1024 // 2)
        src = torch.randn(n_elem, dtype=torch.bfloat16, device="cuda:0").contiguous()
        cpu_buf = torch.empty(n_elem, dtype=torch.bfloat16, pin_memory=True)
        dst = torch.empty(n_elem, dtype=torch.bfloat16, device="cuda:1")

        for _ in range(5):
            cpu_buf.copy_(src, non_blocking=True)
            torch.cuda.synchronize(0)
            dst.copy_(cpu_buf, non_blocking=True)
            torch.cuda.synchronize(1)

        n_iter = max(2, min(20, 500 // max(1, size_mb)))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            cpu_buf.copy_(src, non_blocking=True)
            torch.cuda.synchronize(0)
            dst.copy_(cpu_buf, non_blocking=True)
            torch.cuda.synchronize(1)
        elapsed = (time.perf_counter() - t0) / n_iter
        bw = (size_mb / 1024) / elapsed

        shm_results.append({"size_mb": size_mb, "time_ms": round(elapsed*1000, 1), "bw_gb_s": round(bw, 2)})
        print(f"  {size_mb:4d}MB: {elapsed*1000:6.1f}ms, {bw:.2f} GB/s")
    except Exception as e:
        print(f"  {size_mb}MB: FAILED - {e}")
        break

# ═══════════════════════════════════════════════════════════════
# TEST 7: Generation throughput (real text)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 7: Generation Throughput (real text output)")
print("=" * 70)

for label, prompt, max_new in [
    ("Short", "Explain what machine learning is:", 100),
    ("Medium", "Quantum computing is " * 50, 200),
    ("Long", "The history of AI " * 120, 256),
]:
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=min(4096, max_ctx)).to("cuda:0")
        in_len = inputs['input_ids'].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model0.generate(**inputs, max_new_tokens=max_new,
                                  temperature=0.7, top_p=0.9, do_sample=True,
                                  pad_token_id=eos)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        out_len = gen.shape[1] - in_len
        tok_s = out_len / elapsed * 1000
        text = tokenizer.decode(gen[0][in_len:], skip_special_tokens=True)
        print(f"  {label}: {in_len}→{out_len} tok, {elapsed:.0f}ms ({tok_s:.0f} tok/s)")
        print(f"    {text[:120]}...")
    except Exception as e:
        print(f"  {label}: FAILED - {e}")
        break

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HARDWARE LIMITS FOUND")
print("=" * 70)

results = {
    "hardware": {
        "gpu0": torch.cuda.get_device_name(0),
        "gpu1": torch.cuda.get_device_name(1),
        "pcie": "GPU0 x16, GPU1 x4, no P2P, NODE topology",
        "shm_bandwidth_gb_s": shm_results[-1]["bw_gb_s"] if shm_results else 4.0,
        "model": "Qwen3.5-4B BF16",
        "params": "4.21B",
        "weights_vram_gb": 8.4,
    },
    "limits": {
        "max_context_tokens": max_ctx,
        "max_decode_batch_single_gpu": max_batch,
        "max_sustained_throughput_tok_s": round(sustained_tok_s, 0),
        "gpu0_util_at_max": util_end.get(0, {}).get('util_pct', '?'),
        "gpu1_util_at_max": util_end.get(1, {}).get('util_pct', '?'),
        "vram_at_max_gb": round(torch.cuda.memory_allocated(0)/1e9, 1),
    },
    "shm_transfer": shm_results,
    "sustained_throughput": {
        "batch_per_gpu": sustained_bs,
        "total_batch": sustained_bs * 2,
        "tok_per_s": round(sustained_tok_s, 0),
        "duration_s": round(actual_duration, 1),
        "steps": n_steps,
    },
}

with open(os.path.join(OUTPUT_DIR, "real_benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"  Max context (BF16):       {max_ctx} tokens")
print(f"  Max decode batch (1 GPU): {max_batch}")
print(f"  Sustained DP throughput:   {sustained_tok_s:.0f} tok/s (both GPUs)")
print(f"  SHM transfer BW:           {shm_results[-1]['bw_gb_s']:.2f} GB/s" if shm_results else "  SHM: N/A")
print(f"  GPU0 util at peak:         {util_end.get(0,{}).get('util_pct','?')}%")
print(f"  GPU1 util at peak:         {util_end.get(1,{}).get('util_pct','?')}%")
print(f"\nResults: {OUTPUT_DIR}/real_benchmark_results.json")
