"""
Qwen3.6-27B benchmark (AWQ-INT4, compressed-tensors format).

The 27B model is INT4 quantized with compressed-tensors.
Key: GDN linear_attn layers remain BF16 (not quantized) — by design.

Tests: load, max context, decode batch, generation, VRAM.
"""

import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

MODEL_PATH = "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("BENCHMARK: Qwen3.6-27B (AWQ-INT4, compressed-tensors)")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
eos = tok.eos_token_id

# Load on GPU0
print("[1] Loading 27B AWQ-INT4 model...")
t0 = time.perf_counter()
m0 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, trust_remote_code=True
).eval()
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.1f}s")

# Spec
tc = getattr(m0.config, 'text_config', m0.config)
hidden, layers, max_pos = tc.hidden_size, tc.num_hidden_layers, tc.max_position_embeddings
full_int, kv_heads, head_dim = tc.full_attention_interval, tc.num_key_value_heads, tc.head_dim
n_full = layers // full_int
lin_layers = layers - n_full
kv_per_tok = 2 * n_full * kv_heads * head_dim * 2
ssm_state = lin_layers * tc.linear_num_key_heads * tc.linear_key_head_dim * tc.linear_value_head_dim * 4

print(f"  Spec: {hidden} hidden, {layers} layers ({n_full} full + {lin_layers} linear)")
print(f"  KV/token: {kv_per_tok} bytes ({kv_per_tok/1024:.0f}KB)")
print(f"  SSM state: {ssm_state/1e6:.1f}MB")
print(f"  VRAM used: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")

# Check quantization status of layers
print("\n[2] Layer quantization analysis...")
bf16_linear = 0
int4_linear = 0
for name, module in m0.named_modules():
    if isinstance(module, torch.nn.Linear):
        w = module.weight
        if w.dtype == torch.bfloat16 or w.dtype == torch.float16 or w.dtype == torch.float32:
            bf16_linear += 1
        else:
            int4_linear += 1
print(f"  BF16 Linear layers: {bf16_linear}")
print(f"  INT4 (quantized) Linear layers: {int4_linear}")

# Max context sweep
print("\n[3] Max context sweep...")
max_ctx = 0
ctx_data = []
for target in [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152]:
    try:
        torch.cuda.empty_cache()
        text = ("The quick brown fox jumps over the lazy dog. " * (target//10+1))[:target*4]
        inp = tok(text, return_tensors="pt", truncation=True, max_length=target).to("cuda:0")
        actual = inp['input_ids'].shape[1]

        with torch.no_grad():
            _ = m0(**inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = m0(**inp)
        torch.cuda.synchronize()
        el = (time.perf_counter()-t0)*1000
        ctx_data.append({"tokens": actual, "ms": round(el,1),
                        "tok_s": round(actual/el*1000),
                        "vram": round(torch.cuda.memory_allocated(0)/1e9,1)})
        max_ctx = actual
        print(f"  {actual:6d}: {el:7.0f}ms ({actual/el*1000:.0f} tok/s), "
              f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB")
    except torch.OutOfMemoryError:
        print(f"  {target:6d}: OOM at {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
        torch.cuda.empty_cache()
        break

# Decode batch
print("\n[4] Decode batch sweep...")
batch_data = []
for bs in [1, 2, 4, 8, 16, 32, 64]:
    try:
        torch.cuda.empty_cache()
        inp = torch.randint(0, 1000, (bs, 1), device="cuda:0")
        with torch.no_grad():
            _ = m0(inp)
        torch.cuda.synchronize()
        n = max(3, min(20, 100//max(1,bs)))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                _ = m0(inp)
        torch.cuda.synchronize()
        el = (time.perf_counter()-t0)/n*1000
        batch_data.append({"batch": bs, "ms": round(el,1),
                          "tok_s": round(bs/el*1000), "ms_per_tok": round(el/bs,1)})
        print(f"  batch={bs:3d}: {el:7.0f}ms, {bs/el*1000:6.0f} tok/s, {el/bs:.1f}ms/tok")
    except torch.OutOfMemoryError:
        print(f"  batch={bs}: OOM")
        torch.cuda.empty_cache()
        break

# Generation
print("\n[5] Generation...")
gen_data = []
for label, prompt, max_new in [
    ("math", "Solve: 23 * 47 = ?\nAnswer:", 30),
    ("code", "def is_prime(n):\n    ", 60),
]:
    try:
        inp = tok(prompt, return_tensors="pt", max_length=1024).to("cuda:0")
        in_len = inp['input_ids'].shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m0.generate(**inp, max_new_tokens=max_new,
                             temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=eos)
        torch.cuda.synchronize()
        el = (time.perf_counter()-t0)*1000
        out_len = out.shape[1]-in_len
        text = tok.decode(out[0][in_len:], skip_special_tokens=True)
        gen_data.append({"label": label, "ms": round(el,0),
                        "tok_s": round(out_len/el*1000), "text": text[:200]})
        print(f"  {label}: {in_len}→{out_len} tok, {el:.0f}ms ({out_len/el*1000:.0f} tok/s)")
        print(f"    → {text[:120]}...")
    except Exception as e:
        print(f"  {label}: {e}")
        break

# Save
results = {
    "model": "Qwen3.6-27B-AWQ-INT4",
    "load_time_s": round(load_time,1),
    "spec": {"hidden": hidden, "layers": layers, "n_full": n_full, "n_linear": lin_layers,
             "kv_per_tok": kv_per_tok, "ssm_state_mb": round(ssm_state/1e6,1)},
    "quantization": {"bf16_linear": bf16_linear, "int4_linear": int4_linear},
    "max_context": {"tokens": max_ctx, "data": ctx_data},
    "decode_batch": batch_data,
    "generation": gen_data,
    "final_vram_gb": round(torch.cuda.memory_allocated(0)/1e9,1),
}

with open(f"{OUTPUT_DIR}/bench_27b.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults: {OUTPUT_DIR}/bench_27b.json")
print(f"Max context: {max_ctx} tokens")
print(f"Final VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
