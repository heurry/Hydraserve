"""HydraServe block-scaled E4M3 weight-only GEMM."""

from __future__ import annotations


def dequantize_fp8_weight(weight, *, dtype=None):
    """Materialized oracle for tests and CPU execution."""
    import torch

    output_features, input_features = weight.original_shape
    block_n, block_k = weight.block_size
    expected_scales = (
        (output_features + block_n - 1) // block_n,
        (input_features + block_k - 1) // block_k,
    )
    if tuple(weight.scale_inv.shape) != expected_scales:
        raise ValueError("FP8 inverse-scale grid does not match weight shape")
    output_blocks = torch.arange(output_features, device=weight.data.device) // block_n
    input_blocks = torch.arange(input_features, device=weight.data.device) // block_k
    scales = weight.scale_inv[output_blocks[:, None], input_blocks[None, :]]
    return weight.data.to(dtype or torch.float32) * scales.to(dtype or torch.float32)


def fp8_linear(hidden, weight):
    """Multiply BF16/FP16 activations by block-scaled E4M3 weights."""
    import torch

    if hidden.shape[-1] != weight.original_shape[1]:
        raise ValueError("activation width does not match FP8 weight")
    if weight.block_size != (128, 128):
        raise ValueError("HydraServe FP8 kernel currently requires 128x128 blocks")
    if hidden.device != weight.data.device:
        if not hidden.is_cuda or weight.data.device.type != "cpu":
            raise ValueError("unsupported activation/FP8 weight device placement")
        staged = type(weight)(
            weight.data.to(hidden.device, non_blocking=True).contiguous(),
            weight.scale_inv.to(hidden.device, non_blocking=True).contiguous(),
            weight.original_shape,
            weight.block_size,
        )
        return fp8_linear(hidden, staged)
    output_features, input_features = weight.original_shape
    expected_scales = (
        (output_features + 127) // 128,
        (input_features + 127) // 128,
    )
    if tuple(weight.scale_inv.shape) != expected_scales:
        raise ValueError("FP8 inverse-scale grid does not match weight shape")
    shape = hidden.shape[:-1] + (output_features,)
    flattened = hidden.reshape(-1, input_features).contiguous()
    if not hidden.is_cuda:
        dense = dequantize_fp8_weight(weight, dtype=hidden.dtype)
        return (flattened @ dense.transpose(0, 1)).reshape(shape)
    if not (
        weight.data.is_cuda
        and weight.scale_inv.is_cuda
        and weight.data.is_contiguous()
        and weight.scale_inv.is_contiguous()
    ):
        raise ValueError("FP8 data and scales must be contiguous CUDA tensors")
    if weight.data.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 kernel requires torch.float8_e4m3fn weight storage")
    output = torch.empty(
        flattened.shape[0], output_features, device=hidden.device, dtype=hidden.dtype
    )
    import triton

    block_m = 16 if flattened.shape[0] <= 16 else 32
    if flattened.shape[0] > 64:
        block_m = 64
    block_n = 64 if flattened.shape[0] > 256 else 32
    block_k = 32
    _fp8_gemm_kernel[
        (
            triton.cdiv(flattened.shape[0], block_m),
            triton.cdiv(output_features, block_n),
        )
    ](
        flattened,
        weight.data.view(torch.uint8),
        weight.scale_inv,
        output,
        flattened.shape[0],
        output_features,
        input_features,
        flattened.stride(0),
        weight.data.stride(0),
        weight.scale_inv.stride(0),
        output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SCALE_BLOCK_N=128,
        SCALE_BLOCK_K=128,
        num_warps=4,
    )
    return output.reshape(shape)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_gemm_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        rows,
        output_features,
        input_features: tl.constexpr,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK_N: tl.constexpr,
        SCALE_BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        output_mask = output_offsets < output_features
        for start in range(0, input_features, BLOCK_K):
            input_offsets = start + tl.arange(0, BLOCK_K)
            input_mask = input_offsets < input_features
            activation = tl.load(
                x_ptr
                + row_offsets[:, None] * stride_xm
                + input_offsets[None, :],
                mask=(row_offsets[:, None] < rows) & input_mask[None, :],
                other=0.0,
            )
            quantized_bits = tl.load(
                weight_ptr
                + output_offsets[None, :] * stride_wn
                + input_offsets[:, None],
                mask=output_mask[None, :] & input_mask[:, None],
                other=0.0,
            )
            exponent = (quantized_bits >> 3) & 0xF
            mantissa = quantized_bits & 0x7
            magnitude = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.001953125,
                (1.0 + mantissa.to(tl.float32) * 0.125)
                * tl.exp2(exponent.to(tl.float32) - 7.0),
            )
            quantized = tl.where((quantized_bits & 0x80) != 0, -magnitude, magnitude)
            inverse_scale = tl.load(
                scale_ptr
                + (output_offsets[None, :] // SCALE_BLOCK_N) * stride_sn
                + input_offsets[:, None] // SCALE_BLOCK_K,
                mask=output_mask[None, :] & input_mask[:, None],
                other=0.0,
            )
            dequantized = quantized * inverse_scale
            accumulator += tl.dot(activation, dequantized.to(tl.bfloat16))
        tl.store(
            output_ptr
            + row_offsets[:, None] * stride_om
            + output_offsets[None, :],
            accumulator,
            mask=(row_offsets[:, None] < rows) & output_mask[None, :],
        )
except ImportError:
    _fp8_gemm_kernel = None
