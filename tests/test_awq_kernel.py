from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.kernels.awq import awq_linear, dequantize_int4_weight
from hydraserve.model.weights import PackedInt4Weight


def _pack_last(values):
    rows, columns = values.shape
    result = torch.zeros(rows, columns // 8, dtype=torch.int64, device=values.device)
    for nibble in range(8):
        result |= values[:, nibble::8].to(torch.int64) << (nibble * 4)
    return result.to(torch.int32)


def _pack_first(values):
    rows, groups = values.shape
    result = torch.zeros(rows // 8, groups, dtype=torch.int64, device=values.device)
    for nibble in range(8):
        result |= values[nibble::8].to(torch.int64) << (nibble * 4)
    return result.to(torch.int32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_asymmetric_int4_gemm_matches_dense(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    generator = torch.Generator(device=device).manual_seed(41)
    output_features, input_features, group_size = 64, 256, 128
    quantized = torch.randint(
        0, 16, (output_features, input_features), generator=generator, device=device
    )
    zero_point = torch.randint(
        0,
        16,
        (output_features, input_features // group_size),
        generator=generator,
        device=device,
    )
    scale = torch.rand(
        output_features,
        input_features // group_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16) * 0.2
    weight = PackedInt4Weight(
        _pack_last(quantized).contiguous(),
        scale.contiguous(),
        _pack_first(zero_point).contiguous(),
        (output_features, input_features),
        group_size,
    )
    groups = torch.arange(input_features, device=device) // group_size
    expected_weight = (
        (quantized - zero_point[:, groups]).to(torch.bfloat16) * scale[:, groups]
    )
    torch.testing.assert_close(
        dequantize_int4_weight(weight), expected_weight, atol=0, rtol=0
    )
    hidden = torch.randn(
        3, input_features, generator=generator, device=device, dtype=torch.bfloat16
    )
    expected = hidden @ expected_weight.transpose(0, 1)
    actual = awq_linear(hidden, weight)
    torch.testing.assert_close(actual, expected, atol=0.4, rtol=0.03)
