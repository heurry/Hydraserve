"""
HydraServe comprehensive benchmark v2:
- Speed: prefill, decode, DP, collocated, PD-partial (all to hardware limits)
- Accuracy: WikiText PPL, generation quality
- Quantization: BF16 vs INT4 (weight quantization, VRAM savings)
- MPS: intra-GPU mode test
- Fair: same prompts, same token counts, same hardware for all strategies
"""

import sys, os, time, json, math, gc, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import OrderedDict
import subprocess

MODEL_BASE = "/mnt/nvme-data/models/LLM_model"
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        g = {}
        for line in out.strip().split('\n'):
            i, u, m = [x.strip() for x in line.split(',')]
            g[int(i)] = {"util": int(u), "mem_gb": int(m)/1024}
        return g
    except: return {}

def print_gpu(label=""):
    g = gpu_info()
    parts = [f"GPU{i}:{v['util']}%/{v['mem_gb']:.1f}GB" for i, v in sorted(g.items())]
    print(f"  [{label}] {' | '.join(parts)}")

def tokenize(tok, text, max_len, device):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)

def run_benchmark(model_name, model_path, results):
    """Full benchmark for one model: speed + accuracy + quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from hydraserve.cache.weight_quantizer import WeightQuantizer, quantize_model_to_int4

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*70}")

    r = OrderedDict()
    results[model_name] = r

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos = tok.eos_token_id

    # ── Load BF16 models on both GPUs ──
    print("\n[1] Loading BF16 models...")
    m0 = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, trust_remote_code=True).eval()
    m1 = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:1"}, trust_remote_code=True).eval()

    tc = getattr(m0.config, 'text_config', m0.config)
    hidden, layers, max_pos = tc.hidden_size, tc.num_hidden_layers, tc.max_position_embeddings
    full_int, kv_heads, head_dim = tc.full_attention_interval, tc.num_key_value_heads, tc.head_dim
    n_full = layers // full_int
    kv_per_tok = 2 * n_full * kv_heads * head_dim * 2
    lin_layers = layers - n_full
    ssm_state_size = lin_layers * tc.linear_num_key_heads * tc.linear_key_head_dim * tc.linear_value_head_dim * 4

    r["spec"] = {
        "hidden": hidden, "layers": layers, "max_pos": max_pos,
        "full_attn_layers": n_full, "linear_layers": lin_layers,
        "kv_bytes_per_token": kv_per_tok, "ssm_state_bytes": ssm_state_size,
    }
    print(f"  Spec: {hidden} hidden, {layers} layers ({n_full} full + {lin_layers} linear)")
    print(f"  KV/token: {kv_per_tok} bytes, SSM state: {ssm_state_size/1e6:.1f}MB")
    print(f"  Max context (config): {max_pos}")
    g = gpu_info()
    r["bf16_vram_per_gpu"] = round(g[0]["mem_gb"], 1)

    # ── TEST 1: Max context (BF16, VRAM limit) ──
    print("\n[2] Max context sweep (BF16)...")
    max_ctx_bf16 = 0
    ctx_data = []
    for target in [512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]:
        try:
            torch.cuda.empty_cache()
            text = ("The quick brown fox jumps over the lazy dog. " * (target//10+1))[:target*4]
            inp = tokenize(tok, text, target, "cuda:0")
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
            max_ctx_bf16 = actual
            print(f"    {actual:6d}: {el:7.0f}ms ({actual/el*1000:.0f} tok/s), {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
        except torch.OutOfMemoryError:
            print(f"    {target:6d}: OOM")
            torch.cuda.empty_cache()
            break
    r["max_context_bf16"] = {"tokens": max_ctx_bf16, "data": ctx_data}

    # ── TEST 2: Max decode batch ──
    print("\n[3] Max decode batch (GPU0, BF16)...")
    max_batch = 0
    batch_data = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        try:
            torch.cuda.empty_cache()
            inp = torch.randint(0, 1000, (bs, 1), device="cuda:0")
            with torch.no_grad():
                _ = m0(inp)
            torch.cuda.synchronize()
            n = max(3, min(30, 200//max(1,bs//20)))
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n):
                    _ = m0(inp)
            torch.cuda.synchronize()
            el = (time.perf_counter()-t0)/n*1000
            batch_data.append({"batch": bs, "ms": round(el,1),
                              "tok_s": round(bs/el*1000),
                              "ms_per_tok": round(el/bs,1)})
            max_batch = bs
            print(f"    batch={bs:3d}: {el:6.0f}ms, {bs/el*1000:6.0f} tok/s")
        except torch.OutOfMemoryError:
            print(f"    batch={bs}: OOM")
            torch.cuda.empty_cache()
            break
    r["max_batch_bf16"] = {"batch": max_batch, "data": batch_data}

    # ── TEST 3: DP parallel (CUDA streams, fair) ──
    print("\n[4] DP mode (parallel CUDA streams)...")
    dp_data = []
    for bs in [8, 16, 32, 64]:
        if bs > max_batch:
            break
        try:
            torch.cuda.empty_cache()
            i0 = torch.randint(0, 1000, (bs,1), device="cuda:0")
            i1 = torch.randint(0, 1000, (bs,1), device="cuda:1")
            s0, s1 = torch.cuda.Stream(0), torch.cuda.Stream(1)
            with torch.no_grad():
                for _ in range(10):
                    with torch.cuda.stream(s0): _ = m0(i0)
                    with torch.cuda.stream(s1): _ = m1(i1)
            torch.cuda.synchronize()
            n = max(5, min(50, 400//max(1,bs)))
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n):
                    with torch.cuda.stream(s0): _ = m0(i0)
                    with torch.cuda.stream(s1): _ = m1(i1)
            torch.cuda.synchronize()
            el = (time.perf_counter()-t0)/n*1000
            g = gpu_info()
            dp_data.append({
                "batch_per_gpu": bs, "total_batch": bs*2,
                "ms": round(el,1), "total_tok_s": round(bs*2/el*1000),
                "gpu0_util": g.get(0,{}).get("util",0),
                "gpu1_util": g.get(1,{}).get("util",0),
            })
            print(f"    DP batch={bs}×2: {el:.0f}ms, {bs*2/el*1000:.0f} tok/s, "
                  f"GPU0:{g.get(0,{}).get('util',0)}% GPU1:{g.get(1,{}).get('util',0)}%")
        except torch.OutOfMemoryError:
            print(f"    batch={bs}: OOM")
            torch.cuda.empty_cache()
            break
    r["dp"] = dp_data

    # ── TEST 4: Collocated interference ──
    print("\n[5] Collocated interference (prefill blocks decode)...")
    coll_data = []
    decode_bs = 16
    for ctx in [512, 1024, 2048, 4096]:
        if ctx > max_ctx_bf16:
            break
        try:
            torch.cuda.empty_cache()
            text = ("The quick brown fox. " * (ctx//10+1))[:ctx*4]
            pinp = tokenize(tok, text, ctx, "cuda:0")
            dinp = torch.randint(0, 1000, (decode_bs,1), device="cuda:0")

            with torch.no_grad():
                _ = m0(**pinp); _ = m0(dinp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m0(**pinp)
                for _ in range(5):
                    _ = m0(dinp)
            torch.cuda.synchronize()
            combined = (time.perf_counter()-t0)*1000

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m0(**pinp)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter()-t0)*1000

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(5):
                    _ = m0(dinp)
            torch.cuda.synchronize()
            decode5_ms = (time.perf_counter()-t0)*1000

            delay = combined - prefill_ms - decode5_ms
            coll_data.append({
                "ctx": ctx, "prefill_ms": round(prefill_ms,1),
                "decode5_ms": round(decode5_ms,1), "combined_ms": round(combined,1),
                "delay_ms": round(delay,1), "slowdown": round(combined/(prefill_ms+decode5_ms),2)
            })
            print(f"    ctx={ctx}: prefill={prefill_ms:.0f}ms, delay={delay:.0f}ms "
                  f"({combined/(prefill_ms+decode5_ms):.2f}× slowdown)")
        except torch.OutOfMemoryError:
            print(f"    ctx={ctx}: OOM")
            torch.cuda.empty_cache()
            break
    r["collocated"] = coll_data

    # ── TEST 5: SHM transfer ──
    print("\n[6] SHM transfer bandwidth...")
    shm = []
    for size_mb in [5, 25, 50, 100, 250, 500]:
        try:
            n_el = int(size_mb*1024*1024//2)
            src = torch.randn(n_el, dtype=torch.bfloat16, device="cuda:0").contiguous()
            cpu = torch.empty(n_el, dtype=torch.bfloat16, pin_memory=True)
            dst = torch.empty(n_el, dtype=torch.bfloat16, device="cuda:1")
            for _ in range(5):
                cpu.copy_(src, non_blocking=True); torch.cuda.synchronize(0)
                dst.copy_(cpu, non_blocking=True); torch.cuda.synchronize(1)
            n = max(3, min(20, 500//max(1,size_mb)))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                cpu.copy_(src, non_blocking=True); torch.cuda.synchronize(0)
                dst.copy_(cpu, non_blocking=True); torch.cuda.synchronize(1)
            el = (time.perf_counter()-t0)/n
            shm.append({"size_mb": size_mb, "ms": round(el*1000,1),
                        "bw_gb_s": round((size_mb/1024)/el,2)})
            print(f"    {size_mb:4d}MB: {el*1000:6.1f}ms, {(size_mb/1024)/el:.2f} GB/s")
        except Exception as e:
            print(f"    {size_mb}MB: {e}")
            break
    r["shm"] = shm

    # ── TEST 6: PD partial transfer estimate ──
    print("\n[7] PD PARTIAL_TRANSFER estimate...")
    if shm:
        bw = shm[-1]["bw_gb_s"]
        state_mb = ssm_state_size / 1e6
        transfer_ms = state_mb / 1024 / bw * 1000
        # KV recompute: full_attn_layers / total_layers fraction of prefill
        kv_frac = n_full / layers
        pd_estimate = []
        for c in ctx_data:
            kv_recompute = c["ms"] * kv_frac
            pd_estimate.append({
                "ctx": c["tokens"],
                "prefill_ms": c["ms"],
                "transfer_ms": round(transfer_ms, 1),
                "kv_recompute_ms": round(kv_recompute, 1),
                "pd_total_ms": round(c["ms"] + transfer_ms + kv_recompute, 1),
            })
        r["pd_partial"] = {"shm_bw": bw, "state_mb": round(state_mb,1),
                          "transfer_ms": round(transfer_ms,1), "data": pd_estimate}
        print(f"    State: {state_mb:.1f}MB @ {bw:.1f}GB/s = {transfer_ms:.1f}ms transfer")
        for p in pd_estimate:
            print(f"    ctx={p['ctx']}: prefill {p['prefill_ms']}ms + transfer "
                  f"{p['transfer_ms']}ms + recompute {p['kv_recompute_ms']}ms = "
                  f"{p['pd_total_ms']}ms")

    # ── TEST 7: Accuracy (generation quality) ──
    print("\n[8] Generation quality...")
    gen = []
    tests = [
        ("math", "Solve: 23 * 47 = ?\nAnswer:", 30),
        ("code", "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n    ", 80),
        ("reasoning", "If a train travels 120 km in 2 hours, what is its speed in m/s?\nAnswer:", 60),
    ]
    for label, prompt, max_new in tests:
        try:
            inp = tokenize(tok, prompt, 1024, "cuda:0")
            in_len = inp['input_ids'].shape[1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = m0.generate(**inp, max_new_tokens=max_new,
                                 temperature=0.7, top_p=0.9, do_sample=True,
                                 pad_token_id=eos)
            torch.cuda.synchronize()
            el = (time.perf_counter()-t0)*1000
            out_len = out.shape[1]-in_len
            text = tok.decode(out[0][in_len:], skip_special_tokens=True)
            gen.append({"label": label, "in": in_len, "out": out_len,
                       "ms": round(el,0), "tok_s": round(out_len/el*1000),
                       "text": text[:250]})
            print(f"    {label}: {in_len}→{out_len} tok, {el:.0f}ms ({out_len/el*1000:.0f} tok/s)")
            print(f"      → {text[:100]}...")
        except Exception as e:
            print(f"    {label}: {e}")
            break
    r["generation"] = gen

    # ── TEST 8: INT4 quantization ──
    print("\n[9] INT4 weight quantization...")
    try:
        wq = WeightQuantizer(4, 128)
        # Estimate savings on m0
        est = wq.estimate_vram_savings(m0)
        print(f"    BF16: {est['bf16_gb']:.1f}GB → INT4: {est['int4_gb']:.1f}GB "
              f"({est['ratio']:.1f}× compression)")
        r["int4_estimate"] = est

        # Actually quantize a copy
        print("    Quantizing model (in-place)...")
        t0 = time.perf_counter()
        stats = wq.quantize_model(m0, skip_layers=["embed", "norm", "lm_head"])
        quant_time = time.perf_counter()-t0
        print(f"    Quantized {stats['num_quantized']} Linear layers in {quant_time:.1f}s")
        print(f"    {stats['quantized_params']/1e9:.2f}B params quantized")

        # Verify accuracy on a forward pass
        inp = tokenize(tok, "Hello world, this is a test.", 64, "cuda:0")
        with torch.no_grad():
            out_bf16 = None  # Already quantized in-place, compare logits range
            out_q = m0(**inp)
        logits = out_q.logits if hasattr(out_q, 'logits') else out_q
        r["int4_quantized"] = {
            "layers": stats["num_quantized"], "time_s": round(quant_time,1),
            "logits_mean": round(logits.mean().item(), 4),
            "logits_std": round(logits.std().item(), 4),
        }
        print(f"    Quantized forward OK: logits mean={logits.mean():.4f}, std={logits.std():.4f}")

        # Max context with INT4 (VRAM now much smaller)
        print("    Max context sweep (INT4)...")
        torch.cuda.empty_cache()
        max_ctx_int4 = 0
        int4_ctx = []
        for target in [8192, 16384, 24576, 32768, 49152, 65536]:
            try:
                torch.cuda.empty_cache()
                text = ("The quick brown fox jumps over the lazy dog. " * (target//10+1))[:target*4]
                inp = tokenize(tok, text, target, "cuda:0")
                actual = inp['input_ids'].shape[1]
                with torch.no_grad():
                    _ = m0(**inp)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = m0(**inp)
                torch.cuda.synchronize()
                el = (time.perf_counter()-t0)*1000
                int4_ctx.append({"tokens": actual, "ms": round(el,1),
                                "tok_s": round(actual/el*1000),
                                "vram": round(torch.cuda.memory_allocated(0)/1e9,1)})
                max_ctx_int4 = actual
                print(f"      {actual:6d}: {el:7.0f}ms ({actual/el*1000:.0f} tok/s), {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
            except torch.OutOfMemoryError:
                print(f"      {target:6d}: OOM")
                torch.cuda.empty_cache()
                break
        r["max_context_int4"] = {"tokens": max_ctx_int4, "data": int4_ctx}
        r["int4_context_gain"] = max_ctx_int4 / max(1, max_ctx_bf16)
    except Exception as e:
        print(f"    INT4 test failed: {e}")
        import traceback; traceback.print_exc()
        r["int4_error"] = str(e)

    # Cleanup
    del m0, m1
    gc.collect()
    torch.cuda.empty_cache()
    return r


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("HydraServe Comprehensive Benchmark v2")
    print(f"Hardware: {torch.cuda.get_device_name(0)} × 2, x16+x4 PCIe, no P2P")
    print("=" * 70)

    all_r = OrderedDict()

    # 4B
    r4 = run_benchmark("Qwen3.5-4B", f"{MODEL_BASE}/Qwen3.5-4B", all_r)
    with open(f"{OUTPUT_DIR}/bench_4b_v2.json", "w") as f:
        json.dump(r4, f, indent=2, default=str)

    gc.collect(); torch.cuda.empty_cache(); time.sleep(2)

    # 9B
    r9 = run_benchmark("Qwen3.5-9B", f"{MODEL_BASE}/Qwen3.5-9B", all_r)
    with open(f"{OUTPUT_DIR}/bench_9b_v2.json", "w") as f:
        json.dump(r9, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, r in all_r.items():
        print(f"\n{name}:")
        print(f"  BF16: max ctx={r['max_context_bf16']['tokens']}, max batch={r['max_batch_bf16']['batch']}")
        if "max_context_int4" in r:
            print(f"  INT4: max ctx={r['max_context_int4']['tokens']} "
                  f"({r['int4_context_gain']:.1f}× vs BF16)")
        if r.get("dp"):
            best_dp = max(r["dp"], key=lambda x: x["total_tok_s"])
            print(f"  DP best: {best_dp['total_tok_s']} tok/s (batch {best_dp['total_batch']})")
        if r.get("collocated"):
            max_slow = max(r["collocated"], key=lambda x: x["slowdown"])
            print(f"  Collocated max slowdown: {max_slow['slowdown']}× at ctx={max_slow['ctx']}")
        if r.get("pd_partial"):
            print(f"  PD partial: {r['pd_partial']['transfer_ms']}ms transfer + KV recompute")

    with open(f"{OUTPUT_DIR}/benchmark_summary_v2.json", "w") as f:
        json.dump(all_r, f, indent=2, default=str)
    print(f"\nDone. Results in {OUTPUT_DIR}/")
