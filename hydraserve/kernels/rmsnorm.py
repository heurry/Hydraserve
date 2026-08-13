from __future__ import annotations


def rms_norm(x, weight, eps: float = 1e-6, *, zero_centered: bool = True):
    """HydraServe Triton RMSNorm; no framework normalization op is called."""
    import torch
    import triton
    import triton.language as tl

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("Triton RMSNorm requires CUDA tensors")
    if x.shape[-1] != weight.numel() or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("RMSNorm expects contiguous tensors and matching hidden size")
    columns = x.shape[-1]
    block = triton.next_power_of_2(columns)
    if block > 65_536:
        raise ValueError("hidden dimension exceeds Triton RMSNorm limit")
    output = torch.empty_like(x)
    rows = x.numel() // columns
    _rms_norm_kernel[(rows,)](
        x,
        weight,
        output,
        columns,
        eps,
        ZERO_CENTERED=zero_centered,
        BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return output


def gated_rms_norm(x, gate, weight, eps: float = 1e-6):
    """RMSNorm followed by a SiLU gate, fused in one Triton kernel."""
    import torch
    import triton

    if not (x.is_cuda and gate.is_cuda and weight.is_cuda):
        raise ValueError("Triton gated RMSNorm requires CUDA tensors")
    if x.shape != gate.shape or x.shape[-1] != weight.numel():
        raise ValueError("invalid gated RMSNorm shapes")
    x = x.contiguous()
    gate = gate.contiguous()
    weight = weight.contiguous()
    columns = x.shape[-1]
    block = triton.next_power_of_2(columns)
    output = torch.empty_like(x)
    rows = x.numel() // columns
    _gated_rms_norm_kernel[(rows,)](
        x,
        gate,
        weight,
        output,
        columns,
        eps,
        BLOCK=block,
        num_warps=4,
    )
    return output


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_kernel(x_ptr, weight_ptr, output_ptr, columns, eps, ZERO_CENTERED: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        x = tl.load(x_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / columns
        normalized = x * tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            weight += 1.0
        tl.store(output_ptr + row * columns + offsets, normalized * weight, mask=mask)

    @triton.jit
    def _gated_rms_norm_kernel(x_ptr, gate_ptr, weight_ptr, output_ptr, columns, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        x = tl.load(x_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / columns
        normalized = x * tl.rsqrt(variance + eps) * weight
        activated_gate = gate * tl.sigmoid(gate)
        tl.store(output_ptr + row * columns + offsets, normalized * activated_gate, mask=mask)
except ImportError:
    _rms_norm_kernel = None
    _gated_rms_norm_kernel = None
