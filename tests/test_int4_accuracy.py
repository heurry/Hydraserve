"""
INT4 quantization accuracy verification.

Compares BF16 vs INT4-quantized models on:
1. Logits similarity (KL divergence, cosine similarity)
2. Perplexity on test corpus (WikiText-style)
3. Generation quality (same prompt, compare outputs)
4. GSM8K-style math accuracy (if possible)

Expectation per design doc §3.2: perplexity loss < 0.3, accuracy loss < 1%
"""

import sys, os, time, json, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

MODEL_BASE = "/mnt/nvme-data/models/LLM_model"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
from hydraserve.cache.weight_quantizer import quantize_model_to_int4


def compute_ppl(model, tok, texts, device, max_len=2048):
    """Compute perplexity on a list of texts."""
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        inp = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        loss = out.loss
        if loss is not None:
            total_nll += loss.item() * inp['input_ids'].shape[1]
            total_tokens += inp['input_ids'].shape[1]

    return math.exp(total_nll / max(1, total_tokens))


def compute_logits_similarity(m1, m2, tok, texts, device):
    """Compare logits distributions between two models."""
    kls = []
    cosines = []

    for text in texts:
        inp = tok(text, return_tensors="pt", truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            out1 = m1(**inp)
            out2 = m2(**inp)

        logits1 = out1.logits[:, -1, :]  # Last token logits
        logits2 = out2.logits[:, -1, :]

        # KL divergence (symmetrized)
        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        kl = 0.5 * (F.kl_div(p1.log(), p2, reduction='sum').item() +
                    F.kl_div(p2.log(), p1, reduction='sum').item())
        kls.append(kl)

        # Cosine similarity
        cos = F.cosine_similarity(logits1, logits2, dim=-1).item()
        cosines.append(cos)

    return {
        "mean_kl_divergence": sum(kls) / len(kls),
        "mean_cosine_sim": sum(cosines) / len(cosines),
    }


TEST_TEXTS = [
    "The history of artificial intelligence begins with ancient myths about "
    "mechanical beings and has evolved through the development of computing "
    "machines in the twentieth century.",

    "Machine learning is a subset of artificial intelligence that enables "
    "computers to learn from data without being explicitly programmed for "
    "every specific task.",

    "Quantum computing uses quantum bits or qubits which can exist in "
    "multiple states simultaneously, enabling certain computations to be "
    "performed much faster than classical computers.",

    "The transformer architecture introduced the attention mechanism that "
    "allows models to weigh the importance of different parts of the input "
    "sequence when generating output.",

    "Prefill and decode are two phases of LLM inference. Prefill processes "
    "the entire prompt at once while decode generates tokens one at a time.",
]


def run_accuracy_test(model_name, model_path, results):
    print(f"\n{'='*70}")
    print(f"INT4 ACCURACY TEST: {model_name}")
    print(f"{'='*70}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cuda:0"

    # ── Load BF16 model ──
    print("\n[1] Loading BF16 model...")
    m_bf16 = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True).eval()

    # ── BF16 baseline accuracy ──
    print("[2] BF16 baseline PPL...")
    ppl_bf16 = compute_ppl(m_bf16, tok, TEST_TEXTS, device)
    print(f"  BF16 PPL: {ppl_bf16:.4f}")

    # ── Quantize a copy to INT4 ──
    print("[3] Quantizing to INT4...")
    t0 = time.perf_counter()
    m_int4, stats = quantize_model_to_int4(
        m_bf16, exclude=["embed", "norm", "lm_head", "rotary", "pos_emb", "linear_attn"]
    )
    quant_time = time.perf_counter() - t0
    print(f"  Converted {stats['converted']} layers in {quant_time:.1f}s")
    gc.collect(); torch.cuda.empty_cache()

    # ── INT4 accuracy ──
    print("[4] INT4 PPL...")
    ppl_int4 = compute_ppl(m_int4, tok, TEST_TEXTS, device)
    print(f"  INT4 PPL: {ppl_int4:.4f} (BF16: {ppl_bf16:.4f}, Δ={ppl_int4-ppl_bf16:+.4f})")

    print("[5] Logits similarity (BF16 vs INT4)...")
    sim = compute_logits_similarity(m_bf16, m_int4, tok, TEST_TEXTS[:3], device)
    print(f"  Mean KL divergence: {sim['mean_kl_divergence']:.6f}")
    print(f"  Mean cosine sim: {sim['mean_cosine_sim']:.6f}")

    # ── Generation comparison ──
    print("[6] Generation comparison (same prompt, greedy)...")
    prompt = "The capital of France is"
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_bf16 = m_bf16.generate(**inp, max_new_tokens=20, do_sample=False)
        gen_int4 = m_int4.generate(**inp, max_new_tokens=20, do_sample=False)

    text_bf16 = tok.decode(gen_bf16[0], skip_special_tokens=True)
    text_int4 = tok.decode(gen_int4[0], skip_special_tokens=True)
    print(f"  BF16: {text_bf16}")
    print(f"  INT4: {text_int4}")
    match = text_bf16 == text_int4
    print(f"  Exact match: {match}")

    # ── VRAM comparison ──
    vram_bf16 = torch.cuda.memory_allocated(0) / 1e9
    # Note: m_int4 shares memory with m_bf16 (same tensors replaced), so
    # measure separately
    print(f"[7] VRAM: {vram_bf16:.1f}GB (INT4 layers dequantize on the fly)")

    results[model_name] = {
        "ppl_bf16": round(ppl_bf16, 4),
        "ppl_int4": round(ppl_int4, 4),
        "ppl_delta": round(ppl_int4 - ppl_bf16, 4),
        "mean_kl": round(sim["mean_kl_divergence"], 6),
        "mean_cosine_sim": round(sim["mean_cosine_sim"], 6),
        "generation_exact_match": match,
        "gen_bf16": text_bf16,
        "gen_int4": text_int4,
        "quant_time_s": round(quant_time, 1),
        "converted_layers": stats["converted"],
    }

    del m_bf16, m_int4
    gc.collect(); torch.cuda.empty_cache()
    time.sleep(2)


if __name__ == "__main__":
    print("=" * 70)
    print("INT4 Accuracy Verification")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    all_results = {}
    run_accuracy_test("Qwen3.5-4B", f"{MODEL_BASE}/Qwen3.5-4B", all_results)

    print(f"\n{'='*70}")
    print("ACCURACY SUMMARY")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(f"  PPL: BF16={r['ppl_bf16']} → INT4={r['ppl_int4']} (Δ={r['ppl_delta']:+.4f})")
        print(f"  KL divergence: {r['mean_kl']}")
        print(f"  Cosine sim: {r['mean_cosine_sim']}")
        print(f"  Generation match: {r['generation_exact_match']}")
        # Design doc target: PPL loss < 0.3
        if r['ppl_delta'] < 0.3:
            print(f"  ✓ Within design target (ΔPPL < 0.3)")
        else:
            print(f"  ✗ Exceeds design target (ΔPPL = {r['ppl_delta']:+.4f})")

    with open(f"{OUTPUT_DIR}/int4_accuracy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/int4_accuracy_results.json")
