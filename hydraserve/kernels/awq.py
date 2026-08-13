"""HydraServe grouped asymmetric INT4 weight-only GEMM."""

from __future__ import annotations


def dequantize_int4_weight(weight):
    """Small CPU/GPU oracle; production CUDA execution stays packed."""
    import torch

    output_features, input_features = weight.original_shape
    shifts = torch.arange(8, device=weight.packed.device, dtype=torch.int32) * 4
    quantized = (
        (weight.packed.unsqueeze(-1) >> shifts) & 0xF
    ).reshape(output_features, -1)[:, :input_features]
    zero_point = (
        (weight.zero_point.unsqueeze(-1) >> shifts) & 0xF
    ).permute(0, 2, 1).reshape(-1, weight.scale.shape[1])[:output_features]
    groups = torch.arange(input_features, device=weight.packed.device) // weight.group_size
    return (
        (quantized - zero_point[:, groups]).to(weight.scale.dtype)
        * weight.scale[:, groups]
    )


def awq_linear(hidden, weight):
    import torch

    if hidden.shape[-1] != weight.original_shape[1]:
        raise ValueError("activation width does not match packed INT4 weight")
    if weight.group_size != 128:
        raise ValueError("HydraServe AWQ kernel currently requires group_size=128")
    if hidden.device != weight.packed.device:
        raise ValueError("activation and packed weight must be on the same device")
    shape = hidden.shape[:-1] + (weight.original_shape[0],)
    flattened = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    if not hidden.is_cuda:
        dense = dequantize_int4_weight(weight)
        return (flattened @ dense.transpose(0, 1)).reshape(shape)
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (weight.packed, weight.scale, weight.zero_point)
    ):
        raise ValueError("packed INT4 tensors must be contiguous CUDA tensors")
    output = torch.empty(
        flattened.shape[0],
        weight.original_shape[0],
        device=hidden.device,
        dtype=hidden.dtype,
    )
    import triton

    block_m, block_n, block_k = 16, 32, 32
    _awq_gemm_kernel[
        (triton.cdiv(flattened.shape[0], block_m), triton.cdiv(weight.original_shape[0], block_n))
    ](
        flattened,
        weight.packed,
        weight.scale,
        weight.zero_point,
        output,
        flattened.shape[0],
        weight.original_shape[0],
        weight.original_shape[1],
        flattened.stride(0),
        weight.packed.stride(0),
        weight.scale.stride(0),
        weight.zero_point.stride(0),
        output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE=weight.group_size,
        num_warps=4,
    )
    return output.reshape(shape)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _awq_gemm_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        zero_ptr,
        output_ptr,
        rows,
        output_features,
        input_features: tl.constexpr,
        stride_xm,
        stride_wn,
        stride_sn,
        stride_zn,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        rows_offset = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        outputs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        output_mask = outputs < output_features
        output_word = outputs // 8
        output_shift = (outputs % 8) * 4
        for start in range(0, input_features, BLOCK_K):
            inputs = start + tl.arange(0, BLOCK_K)
            input_mask = inputs < input_features
            activation = tl.load(
                x_ptr + rows_offset[:, None] * stride_xm + inputs[None, :],
                mask=(rows_offset[:, None] < rows) & input_mask[None, :],
                other=0.0,
            )
            packed_word = tl.load(
                packed_ptr
                + outputs[None, :] * stride_wn
                + (inputs[:, None] // 8),
                mask=output_mask[None, :] & input_mask[:, None],
                other=0,
            )
            quantized = (packed_word >> ((inputs[:, None] % 8) * 4)) & 0xF
            groups = inputs // GROUP_SIZE
            zero_word = tl.load(
                zero_ptr + output_word[None, :] * stride_zn + groups[:, None],
                mask=output_mask[None, :] & input_mask[:, None],
                other=0,
            )
            zero_point = (zero_word >> output_shift[None, :]) & 0xF
            scale = tl.load(
                scale_ptr + outputs[None, :] * stride_sn + groups[:, None],
                mask=output_mask[None, :] & input_mask[:, None],
                other=0.0,
            )
            dequantized = (quantized - zero_point).to(tl.float32) * scale
            accumulator += tl.dot(activation, dequantized.to(tl.bfloat16))
        tl.store(
            output_ptr + rows_offset[:, None] * stride_om + outputs[None, :],
            accumulator,
            mask=(rows_offset[:, None] < rows) & output_mask[None, :],
        )
except ImportError:
    _awq_gemm_kernel = None
