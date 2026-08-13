from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from hydraserve.kernels.fp8 import dequantize_fp8_weight, fp8_linear
from hydraserve.model.weights import BlockScaledFP8Weight, ShardedSafeTensorLoader


def _quantized_weight(output_features: int, input_features: int, *, device: str):
    generator = torch.Generator(device=device).manual_seed(41)
    dense = torch.randn(
        output_features,
        input_features,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    scales = torch.empty(
        (output_features + 127) // 128,
        (input_features + 127) // 128,
        device=device,
        dtype=torch.float32,
    )
    quantized = torch.empty_like(dense, dtype=torch.float8_e4m3fn)
    for output_start in range(0, output_features, 128):
        for input_start in range(0, input_features, 128):
            block = dense[
                output_start : output_start + 128,
                input_start : input_start + 128,
            ]
            scale = block.abs().max().clamp_min(1e-6) / 448.0
            scales[output_start // 128, input_start // 128] = scale
            quantized[
                output_start : output_start + 128,
                input_start : input_start + 128,
            ] = (block / scale).to(torch.float8_e4m3fn)
    return dense, BlockScaledFP8Weight(
        quantized.contiguous(),
        scales.to(torch.bfloat16).contiguous(),
        (output_features, input_features),
    )


def test_block_scaled_fp8_cpu_oracle_handles_partial_blocks() -> None:
    dense, weight = _quantized_weight(143, 259, device="cpu")
    restored = dequantize_fp8_weight(weight)
    assert restored.shape == dense.shape
    assert (restored - dense).abs().mean() < 0.02
    hidden = torch.randn(5, 259, dtype=torch.bfloat16)
    actual = fp8_linear(hidden, weight)
    expected = hidden @ restored.to(torch.bfloat16).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 143, 259), (17, 256, 384)])
def test_triton_block_scaled_fp8_matches_materialized_oracle(shape) -> None:
    rows, output_features, input_features = shape
    _, weight = _quantized_weight(output_features, input_features, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(43)
    hidden = torch.randn(
        rows,
        input_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = hidden @ dequantize_fp8_weight(
        weight, dtype=torch.bfloat16
    ).transpose(0, 1)
    actual = fp8_linear(hidden, weight)
    torch.testing.assert_close(actual, expected, atol=1.25e-1, rtol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_fp8_streams_host_resident_weight() -> None:
    _, weight = _quantized_weight(143, 259, device="cpu")
    generator = torch.Generator(device="cuda").manual_seed(47)
    hidden = torch.randn(
        3, 259, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    expected = hidden @ dequantize_fp8_weight(
        weight, dtype=torch.bfloat16
    ).to("cuda").transpose(0, 1)
    actual = fp8_linear(hidden, weight)
    assert actual.device.type == "cuda"
    assert weight.data.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=1.25e-1, rtol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ampere_manual_e4m3fn_decode_matches_every_finite_bit_pattern() -> None:
    bits = torch.stack(
        (
            torch.arange(128, dtype=torch.uint8),
            torch.arange(128, dtype=torch.uint8) + 128,
        )
    )
    # E4M3FN reserves sign variants 0x7f/0xff for NaN.
    bits[:, -1] = 0
    data = bits.view(torch.float8_e4m3fn).cuda()
    weight = BlockScaledFP8Weight(
        data,
        torch.ones(1, 1, device="cuda", dtype=torch.bfloat16),
        (2, 128),
    )
    hidden = torch.eye(128, device="cuda", dtype=torch.bfloat16)
    actual = fp8_linear(hidden, weight)
    expected = data.to(torch.bfloat16).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp8_scale_grid_is_validated() -> None:
    _, weight = _quantized_weight(129, 129, device="cpu")
    invalid = BlockScaledFP8Weight(
        weight.data, weight.scale_inv[:1], weight.original_shape
    )
    with pytest.raises(ValueError, match="scale grid"):
        fp8_linear(torch.randn(1, 129), invalid)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("HYDRASERVE_RUN_REAL_FP8_LAYER") != "1",
    reason="set HYDRASERVE_RUN_REAL_FP8_LAYER=1 to load a real 27B FP8 layer",
)
def test_real_27b_fp8_projection_matches_materialized_oracle() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    loader = ShardedSafeTensorLoader(
        "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8"
    )
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    weight = BlockScaledFP8Weight(
        loader.tensor(name, device="cuda").contiguous(),
        loader.tensor(
            f"{name}_scale_inv", device="cuda", dtype=torch.bfloat16
        ).contiguous(),
        loader.tensor_shape(name),
    )
    generator = torch.Generator(device="cuda").manual_seed(53)
    hidden = torch.randn(
        1,
        weight.shape[1],
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = hidden @ dequantize_fp8_weight(
        weight, dtype=torch.bfloat16
    ).transpose(0, 1)
    actual = fp8_linear(hidden, weight)
    torch.testing.assert_close(actual, expected, atol=5e-1, rtol=8e-2)
