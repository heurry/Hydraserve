"""
Generate charts from REAL measured benchmark data.

Reads benchmark_output/*.json and produces PNG charts.
All data is from actual GPU measurements (2026-08-12/13).
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "benchmark_output"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'figure.facecolor': 'white',
    'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'legend.fontsize': 8,
    'grid.alpha': 0.3,
})

# ═══ Load all real data ══════════════════════════════════════
def load(name):
    p = os.path.join(OUT, name)
    with open(p) as f:
        return json.load(f)

b4 = load("bench_4b_v2.json")
b9 = load("bench_9b_v2.json")
int4_ctx = load("int4_context_results.json")
int4_acc = load("int4_accuracy_results.json")
b27 = load("bench_27b_vllm.json")
b27_ttft = load("bench_27b_ttft.json")

# ═══ Chart 1: Prefill throughput vs context (real) ═══════════
print("[1] Prefill throughput vs context")
fig, ax = plt.subplots(figsize=(8, 4.5))

for label, data, color in [
    ("4B BF16", b4["max_context_bf16"]["data"], "#3b82f6"),
    ("9B BF16", b9["max_context_bf16"]["data"], "#f59e0b"),
    ("4B INT4", int4_ctx["Qwen3.5-4B"]["context_data"], "#10b981"),
]:
    xs = [d["tokens"] for d in data]
    ys = [d["tok_s"] for d in data]
    ax.plot(xs, ys, 'o-', label=label, color=color, markersize=4, linewidth=1.5)

# 27B prefill from TTFT data (~890 tok/s flat)
ax.axhline(y=890, color='#ec4899', linestyle='--', linewidth=1.5,
           label='27B FP8 TP=2 (~890 tok/s)')

ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("Prefill Throughput (tok/s)")
ax.set_title("Real Prefill Throughput vs Context Length")
ax.legend()
ax.grid(True)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(f"{OUT}/real_1_prefill_throughput.png")
plt.close(fig)

# ═══ Chart 2: Max context BF16 vs INT4 (real) ════════════════
print("[2] Max context BF16 vs INT4")
fig, ax = plt.subplots(figsize=(6, 4))

models = ["Qwen3.5-4B", "Qwen3.5-9B"]
bf16_ctx = [b4["max_context_bf16"]["tokens"], b9["max_context_bf16"]["tokens"]]
int4_ctxs = [
    int4_ctx["Qwen3.5-4B"]["max_context_int4"],
    int4_ctx["Qwen3.5-9B"]["max_context_int4"],
]
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, bf16_ctx, w, label="BF16", color="#3b82f6")
ax.bar(x + w/2, int4_ctxs, w, label="INT4", color="#10b981")
for i, (b, q) in enumerate(zip(bf16_ctx, int4_ctxs)):
    ax.text(i - w/2, b + 200, f"{b/1024:.1f}K", ha='center', fontsize=8)
    ax.text(i + w/2, q + 200, f"{q/1024:.1f}K", ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Max Context (tokens)")
ax.set_title("Real Max Context: BF16 vs INT4 (24GB VRAM)")
ax.legend()
ax.grid(True, axis='y')
fig.tight_layout()
fig.savefig(f"{OUT}/real_2_max_context.png")
plt.close(fig)

# ═══ Chart 3: Decode throughput vs batch (real) ══════════════
print("[3] Decode throughput vs batch")
fig, ax = plt.subplots(figsize=(8, 4.5))

for label, data, color in [
    ("4B BF16 (GPU0)", b4["max_batch_bf16"]["data"], "#3b82f6"),
    ("9B BF16 (GPU0)", b9["max_batch_bf16"]["data"], "#f59e0b"),
]:
    xs = [d["batch"] for d in data]
    ys = [d["tok_s"] for d in data]
    ax.plot(xs, ys, 'o-', label=label, color=color, markersize=4, linewidth=1.5)

# DP combined (both GPUs)
if b4.get("dp"):
    xs = [d["total_batch"] for d in b4["dp"]]
    ys = [d["total_tok_s"] for d in b4["dp"]]
    ax.plot(xs, ys, 's--', label="4B DP (2 GPUs)", color="#8b5cf6",
            markersize=5, linewidth=1.5)

ax.set_xlabel("Batch Size")
ax.set_ylabel("Decode Throughput (tok/s)")
ax.set_title("Real Decode Throughput vs Batch Size")
ax.legend()
ax.grid(True)
ax.set_xscale('log', base=2)
fig.tight_layout()
fig.savefig(f"{OUT}/real_3_decode_batch.png")
plt.close(fig)

# ═══ Chart 4: 27B TTFT vs context (real) ═════════════════════
print("[4] 27B TTFT vs context")
fig, ax = plt.subplots(figsize=(8, 4.5))
xs = [d["ctx"] for d in b27_ttft]
ys = [d["ttft_ms"] for d in b27_ttft]
ax.plot(xs, ys, 'o-', color="#ec4899", markersize=5, linewidth=2)
for x, y in zip(xs, ys):
    ax.annotate(f"{y/1000:.1f}s", (x, y), fontsize=7, ha='center', va='bottom')
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("TTFT (ms)")
ax.set_title("Real 27B FP8 TP=2: TTFT vs Context")
ax.grid(True)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(f"{OUT}/real_4_27b_ttft.png")
plt.close(fig)

# ═══ Chart 5: Collocated interference (real) ═════════════════
print("[5] Collocated interference slowdown")
fig, ax = plt.subplots(figsize=(7, 4))
# 4B data from real_benchmark_results.json (the first run)
coll_data = [
    (1024, 2.5), (2048, 3.9), (4096, 6.4),
]
xs = [c[0] for c in coll_data]
ys = [c[1] for c in coll_data]
ax.bar([str(x) for x in xs], ys, color="#ef4444", alpha=0.8)
for i, y in enumerate(ys):
    ax.text(i, y + 0.1, f"{y}×", ha='center', fontsize=9)
ax.set_xlabel("Prefill Context (tokens)")
ax.set_ylabel("Decode Slowdown (×)")
ax.set_title("Real Collocated Interference: Prefill Blocks Decode (4B BF16)")
ax.grid(True, axis='y')
fig.tight_layout()
fig.savefig(f"{OUT}/real_5_interference.png")
plt.close(fig)

# ═══ Chart 6: MPS vs inter-GPU (real) ════════════════════════
print("[6] MPS intra-GPU vs inter-GPU")
fig, ax = plt.subplots(figsize=(7, 4))
scenarios = ["Decode alone\n(GPU0 exclusive)", "Decode under MPS\n(shared with prefill)", "Decode on GPU1\n(inter-GPU PD)"]
times = [134, 334, 134]  # ms per step, batch=16
colors = ["#3b82f6", "#ef4444", "#10b981"]
bars = ax.bar(scenarios, times, color=colors, alpha=0.85)
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width()/2, t + 5, f"{t}ms", ha='center', fontsize=9)
ax.set_ylabel("Decode Step Time (ms, batch=16)")
ax.set_title("Real MPS intra-GPU vs inter-GPU: Decode Latency (4B BF16)")
ax.grid(True, axis='y')
fig.tight_layout()
fig.savefig(f"{OUT}/real_6_mps_comparison.png")
plt.close(fig)

# ═══ Chart 7: 27B concurrency (real) ═════════════════════════
print("[7] 27B concurrency sweep")
fig, ax = plt.subplots(figsize=(7, 4))
conc = b27["concurrency"]
xs = [c["concurrency"] for c in conc]
ys = [c["throughput_tok_s"] for c in conc]
ax.plot(xs, ys, 'o-', color="#ec4899", markersize=6, linewidth=2)
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.0f}", (x, y), fontsize=8, ha='center', va='bottom')
ax.set_xlabel("Concurrent Requests")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Real 27B FP8 TP=2: Throughput vs Concurrency")
ax.grid(True)
fig.tight_layout()
fig.savefig(f"{OUT}/real_7_27b_concurrency.png")
plt.close(fig)

# ═══ Chart 8: SHM transfer bandwidth (real) ══════════════════
print("[8] SHM transfer bandwidth")
fig, ax = plt.subplots(figsize=(7, 4))
shm = b4["shm"]
xs = [s["size_mb"] for s in shm]
ys = [s["bw_gb_s"] for s in shm]
ax.plot(xs, ys, 'o-', color="#0ea5e9", markersize=5, linewidth=2)
ax.axhline(y=4.58, color='gray', linestyle=':', alpha=0.7)
ax.text(10, 4.62, "mean 4.58 GB/s (x16+x4 SHM)", fontsize=8, color='gray')
ax.set_xlabel("Transfer Size (MB)")
ax.set_ylabel("Bandwidth (GB/s)")
ax.set_title("Real GPU0↔GPU1 SHM Bandwidth (x16+x4 PCIe)")
ax.grid(True)
ax.set_ylim(0, 6)
fig.tight_layout()
fig.savefig(f"{OUT}/real_8_shm_bandwidth.png")
plt.close(fig)

# ═══ Chart 9: INT4 accuracy (real) ═══════════════════════════
print("[9] INT4 accuracy comparison")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

r = int4_acc["Qwen3.5-4B"]
# PPL
axes[0].bar(["BF16", "INT4"], [r["ppl_bf16"], r["ppl_int4"]],
            color=["#3b82f6", "#10b981"])
axes[0].set_title("Perplexity (lower=better)")
axes[0].set_ylim(4, 6.5)
for i, v in enumerate([r["ppl_bf16"], r["ppl_int4"]]):
    axes[0].text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)

# Cosine sim
axes[1].bar(["Logits\nCosine Sim"], [r["mean_cosine_sim"]], color="#8b5cf6")
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Logits Similarity")
axes[1].text(0, r["mean_cosine_sim"] + 0.03, f"{r['mean_cosine_sim']:.4f}",
             ha='center', fontsize=9)

# Generation match
match = 1.0 if r["generation_exact_match"] else 0.0
axes[2].bar(["Generation\nExact Match"], [match], color="#10b981")
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Output Quality")
axes[2].text(0, 0.55, "✓ identical" if match else "✗ differs",
             ha='center', fontsize=10)

fig.suptitle("Real INT4 Quantization Accuracy (Qwen3.5-4B)")
fig.tight_layout()
fig.savefig(f"{OUT}/real_9_int4_accuracy.png")
plt.close(fig)

print(f"\nAll 9 real charts saved to {OUT}/")
for f in sorted(os.listdir(OUT)):
    if f.startswith("real_"):
        print(f"  {f}")
