"""Quantize a Qwen3.5/3.6 hybrid-attention checkpoint to HydraServe block-scaled FP8.

Converts the 2D linear-projection weights under ``model.language_model.*`` from
BF16 to E4M3 with a per-128x128-block inverse scale, exactly matching the
``BlockScaledFP8Weight`` layout consumed by ``QwenTextRuntime.from_checkpoint``
(see ``hydraserve/kernels/fp8.py`` and ``tests/test_fp8_kernel.py``).

Every GEMM weight ``<name>.weight`` becomes two tensors:

* ``<name>.weight``        — ``torch.float8_e4m3fn`` data, same shape as before
* ``<name>.weight_scale_inv`` — BF16 grid of shape
  ``(ceil(out/128), ceil(in/128))`` holding the per-block inverse scale

Non-GEMM tensors (embeddings, layernorms, conv1d, A_log/dt_bias, and the
``mtp.*``/``model.visual.*`` blocks that HydraServe ignores) are copied through
unchanged, so the output directory stays a complete checkpoint.

Usage:
    python scripts/quantize_fp8.py /path/to/Qwen3.5-4B /path/to/Qwen3.5-4B-FP8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LANGUAGE_PREFIX = "model.language_model"
E4M3_MAX = 448.0
BLOCK = 128

# Non-weight files carried over so the directory remains a loadable checkpoint.
COPY_FILES = (
    "config.json",
    "configuration.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    ".gitattributes",
    "LICENSE",
    "README.md",
)


def should_quantize(name: str, ndim: int) -> bool:
    """Quantize only the 2D GEMM projections the runtime runs through ``fp8_linear``."""
    if ndim != 2:
        return False
    if name == "lm_head.weight":
        return True
    return name.startswith(f"{LANGUAGE_PREFIX}.layers.") and name.endswith(".weight")


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e4m3_data, scale_inv)`` for a symmetric per-128x128-block E4M3 weight."""
    w = weight.float()
    out_features, in_features = w.shape
    n_out = (out_features + BLOCK - 1) // BLOCK
    n_in = (in_features + BLOCK - 1) // BLOCK
    pad_out = n_out * BLOCK - out_features
    pad_in = n_in * BLOCK - in_features
    padded = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    blocks = padded.view(n_out, BLOCK, n_in, BLOCK).permute(0, 2, 1, 3)
    scale_inv = (blocks.abs().amax(dim=(2, 3)).clamp_min(1e-6) / E4M3_MAX).to(
        torch.bfloat16
    )
    data = (blocks / scale_inv[:, :, None, None]).to(torch.float8_e4m3fn)
    data = data.permute(0, 2, 1, 3).reshape(n_out * BLOCK, n_in * BLOCK)[
        :out_features, :in_features
    ]
    return data.contiguous(), scale_inv.contiguous()


def discover_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8"))
        return sorted({model_dir / name for name in meta["weight_map"].values()})
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")


def iter_tensors(model_dir: Path):
    for shard in discover_shards(model_dir):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # get_tensor may return an mmap-backed view; copy before the handle closes.
                yield key, handle.get_tensor(key).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="source BF16 model directory")
    parser.add_argument("dst", type=Path, help="output FP8 model directory")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"output directory already exists: {dst}")
    dst.mkdir(parents=True)

    for filename in COPY_FILES:
        candidate = src / filename
        if candidate.is_file():
            shutil.copy2(candidate, dst / filename)

    out_tensors: dict[str, torch.Tensor] = {}
    n_quantized = 0
    for name, tensor in iter_tensors(src):
        if should_quantize(name, tensor.ndim):
            data, scale_inv = quantize_fp8(tensor)
            out_tensors[name] = data
            out_tensors[f"{name}_scale_inv"] = scale_inv
            n_quantized += 1
        else:
            out_tensors[name] = tensor

    save_file(out_tensors, dst / "model.safetensors")
    print(f"quantized {n_quantized} GEMM tensors; wrote {dst / 'model.safetensors'}")


if __name__ == "__main__":
    main()
