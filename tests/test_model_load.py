"""
Quick test: Load Qwen3.5-4B and run a forward pass.

Run: python tests/test_model_load.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

# Model path
MODEL_PATH = "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B"

def test_load_model():
    """Test loading Qwen3.5-4B from local path."""
    print("=" * 60)
    print("HydraServe: Qwen3.5-4B Model Load Test")
    print("=" * 60)

    # Check if GPU has enough free memory (>5GB)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        free_mem = torch.cuda.mem_get_info(0)[0] / 1e9
        use_cuda = free_mem > 5.0  # Need at least 5GB free
        if not use_cuda:
            print(f"  GPU 0 only has {free_mem:.1f}GB free, using CPU fallback")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Try loading with HuggingFace transformers
    print(f"\n[1] Loading model from {MODEL_PATH}...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

        device_map = {"": device} if str(device) != "cpu" else None
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if device_map:
            load_kwargs["device_map"] = device_map
        else:
            load_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            **load_kwargs,
        )
        model.eval()
        print(f"  Model loaded successfully.")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params / 1e9:.2f}B")
        print(f"  Dtype: {next(model.parameters()).dtype}")

        # Check layer structure
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
            print(f"  Layers: {n_layers}")

            # Try to identify full vs linear attention layers
            # Qwen3.5 stores layer_types in config
            cfg = model.config
            if hasattr(cfg, 'text_config'):
                tc = cfg.text_config
                if hasattr(tc, 'layer_types'):
                    types = tc.layer_types
                    full_count = sum(1 for t in types if t == 'full_attention')
                    linear_count = sum(1 for t in types if t == 'linear_attention')
                    print(f"  Full attention layers: {full_count}")
                    print(f"  Linear attention layers: {linear_count}")

        # Quick forward pass test
        print(f"\n[2] Running forward pass test...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        print(f"  Input tokens: {inputs['input_ids'].shape[1]}")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Forward pass: {elapsed:.1f}ms")
        print(f"  Logits shape: {outputs.logits.shape}")

        # Generate a few tokens
        print(f"\n[3] Generating 50 tokens...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"  Generation time: {elapsed:.1f}ms")
        print(f"  Tokens/s: {50 / (elapsed/1000):.1f}")
        print(f"  Output: {output_text[:200]}...")

        # Memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n[4] GPU Memory:")
            print(f"  Allocated: {mem_used:.2f} GB")
            print(f"  Reserved: {mem_reserved:.2f} GB")

        print(f"\n{'='*60}")
        print("All tests passed! Model is ready for HydraServe integration.")
        print(f"{'='*60}")

        return model, tokenizer

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, tokenizer = test_load_model()
