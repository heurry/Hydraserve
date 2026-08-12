"""
Qwen3.6-27B smoke test (pre-gdn-fix, single safetensors, 17.7GB AWQ INT4).
Simpler structure: model.safetensors + model-vision-bf16.safetensors (921MB).
"""

import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

MODEL_PATH = "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4-smoke.pre-gdn-fix"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("BENCHMARK: Qwen3.6-27B smoke (pre-gdn-fix, AWQ W4A16_ASYM)")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
eos = tok.eos_token_id

# Load with CPU offload for vision
print("[1] Loading...")
t0 = time.perf_counter()
m0 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "16GiB", "cpu": "80GiB"},
    trust_remote_code=True
).eval()
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.1f}s")
print(f"  GPU VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
gpu_mods = sum(1 for v in m0.hf_device_map.values() if v == 0)
cpu_mods = sum(1 for v in m0.hf_device_map.values() if v == 'cpu')
print(f"  Modules: {gpu_mods} GPU, {cpu_mods} CPU")

# Spec
tc = getattr(m0.config, 'text_config', m0.config)
hidden, layers = tc.hidden_size, tc.num_hidden_layers
n_full = layers // tc.full_attention_interval
lin_layers = layers - n_full
kv_per_tok = 2 * n_full * tc.num_key_value_heads * tc.head_dim * 2
ssm = lin_layers * tc.linear_num_key_heads * tc.linear_key_head_dim * tc.linear_value_head_dim * 4
print(f"  Spec: {hidden} hidden, {layers} layers, KV {kv_per_tok}B/tok, SSM {ssm/1e6:.1f}MB")

# Forward test
print("\n[2] Forward pass test...")
try:
    inp = tok("Hello world, this is a test of the 27B model.",
              return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m0(**inp)
    torch.cuda.synchronize()
    el = (time.perf_counter() - t0) * 1000
    print(f"  Forward: {el:.0f}ms, logits shape {out.logits.shape}")
    print(f"  VRAM after: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
except Exception as e:
    print(f"  Forward FAILED: {e}")
    import traceback; traceback.print_exc()

# Context sweep
print("\n[3] Context sweep...")
max_ctx = 0
for target in [128, 256, 512, 1024, 2048, 4096]:
    try:
        torch.cuda.empty_cache()
        text = ("The quick brown fox. " * (target // 10 + 1))[:target * 4]
        inp = tok(text, return_tensors="pt", truncation=True,
                  max_length=target).to("cuda:0")
        actual = inp['input_ids'].shape[1]
        with torch.no_grad():
            _ = m0(**inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = m0(**inp)
        torch.cuda.synchronize()
        el = (time.perf_counter() - t0) * 1000
        max_ctx = actual
        print(f"  {actual:5d}: {el:7.0f}ms ({actual/el*1000:.0f} tok/s), "
              f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB")
    except torch.OutOfMemoryError:
        print(f"  {target:5d}: OOM")
        torch.cuda.empty_cache()
        break

# Decode batch
print("\n[4] Decode batch...")
for bs in [1, 2, 4, 8]:
    try:
        torch.cuda.empty_cache()
        inp = torch.randint(0, 1000, (bs, 1), device="cuda:0")
        with torch.no_grad():
            _ = m0(inp)
        torch.cuda.synchronize()
        n = max(3, 10)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                _ = m0(inp)
        torch.cuda.synchronize()
        el = (time.perf_counter() - t0) / n * 1000
        print(f"  batch={bs}: {el:.0f}ms, {bs/el*1000:.0f} tok/s, {el/bs:.1f}ms/tok")
    except torch.OutOfMemoryError:
        print(f"  batch={bs}: OOM")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"  batch={bs}: {e}")
        break

results = {
    "model": "Qwen3.6-27B-smoke.pre-gdn-fix",
    "load_time_s": round(load_time, 1),
    "vram_gb": round(torch.cuda.memory_allocated(0)/1e9, 1),
    "max_context": max_ctx,
    "gpu_modules": gpu_mods, "cpu_modules": cpu_mods,
}
with open(f"{OUTPUT_DIR}/bench_27b_smoke.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults: {OUTPUT_DIR}/bench_27b_smoke.json")
