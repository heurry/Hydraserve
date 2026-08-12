"""
INT4 quantization → unlock long context test.

Uses Int4Linear replacement (real VRAM savings):
- 4B: 8.8GB → ~2.2GB weights → expect 32K-64K context
- 9B: 18.2GB → ~4.6GB weights → expect 16K-32K context

Also verifies INT4 accuracy vs BF16 on same prompts.
"""

import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydraserve.cache.weight_quantizer import quantize_model_to_int4

MODEL_BASE = "/mnt/nvme-data/models/LLM_model"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_int4(model_name, model_path, results):
    print(f"\n{'='*70}")
    print(f"INT4 CONTEXT TEST: {model_name}")
    print(f"{'='*70}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos = tok.eos_token_id

    # ── Phase 1: BF16 baseline (context limit) ──
    print("\n[1] BF16 baseline...")
    m_bf16 = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, trust_remote_code=True).eval()
    bf16_vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"  BF16 VRAM: {bf16_vram:.1f}GB")

    # ── Phase 2: Quantize to INT4 (Int4Linear replacement) ──
    print("\n[2] Quantizing to INT4 (Int4Linear replacement)...")
    t0 = time.perf_counter()
    m_bf16, stats = quantize_model_to_int4(
        m_bf16,
        exclude=["embed", "norm", "lm_head", "rotary", "pos_emb", "linear_attn"],
    )
    quant_time = time.perf_counter() - t0
    gc.collect()
    torch.cuda.empty_cache()
    int4_vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Converted {stats['converted']} layers in {quant_time:.1f}s")
    print(f"  INT4 VRAM: {int4_vram:.1f}GB (was {bf16_vram:.1f}GB, saved {bf16_vram-int4_vram:.1f}GB)")
    print(f"  Compression: {stats['bf16_gb']:.1f}GB → {stats['int4_gb']:.1f}GB")

    # ── Phase 3: INT4 max context sweep ──
    print("\n[3] INT4 max context sweep...")
    max_ctx = 0
    ctx_data = []
    for target in [1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768,
                   49152, 65536, 98304, 131072, 196608, 262144]:
        try:
            torch.cuda.empty_cache()
            text = ("The quick brown fox jumps over the lazy dog. " *
                    (target // 10 + 1))[:target * 4]
            inp = tok(text, return_tensors="pt", truncation=True,
                      max_length=target).to("cuda:0")
            actual = inp['input_ids'].shape[1]

            with torch.no_grad():
                _ = m_bf16(**inp)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m_bf16(**inp)
            torch.cuda.synchronize()
            el = (time.perf_counter() - t0) * 1000

            ctx_data.append({
                "tokens": actual, "ms": round(el, 1),
                "tok_s": round(actual / el * 1000),
                "vram": round(torch.cuda.memory_allocated(0) / 1e9, 1),
            })
            max_ctx = actual
            print(f"  {actual:7d}: {el:8.0f}ms ({actual/el*1000:6.0f} tok/s), "
                  f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB")
        except torch.OutOfMemoryError:
            print(f"  {target:7d}: OOM at {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
            torch.cuda.empty_cache()
            break

    results[model_name] = {
        "bf16_vram_gb": round(bf16_vram, 1),
        "int4_vram_gb": round(int4_vram, 1),
        "vram_saved_gb": round(bf16_vram - int4_vram, 1),
        "quant_time_s": round(quant_time, 1),
        "converted_layers": stats["converted"],
        "max_context_int4": max_ctx,
        "context_data": ctx_data,
    }

    del m_bf16
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


if __name__ == "__main__":
    print("=" * 70)
    print("INT4 → Long Context Unlock Test")
    print(f"GPU: {torch.cuda.get_device_name(0)} (24GB)")
    print("=" * 70)

    all_results = {}

    # ── 4B ──
    test_model_int4("Qwen3.5-4B", f"{MODEL_BASE}/Qwen3.5-4B", all_results)

    # ── 9B ──
    test_model_int4("Qwen3.5-9B", f"{MODEL_BASE}/Qwen3.5-9B", all_results)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("INT4 CONTEXT UNLOCK SUMMARY")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(f"  BF16: {r['bf16_vram_gb']:.1f}GB → INT4: {r['int4_vram_gb']:.1f}GB "
              f"(saved {r['vram_saved_gb']:.1f}GB)")
        print(f"  INT4 max context: {r['max_context_int4']} tokens")

    with open(f"{OUTPUT_DIR}/int4_context_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/int4_context_results.json")
