from __future__ import annotations


def silu_and_mul(gate, up):
    """Compute ``SiLU(gate) * up`` with one CUDA kernel."""
    import torch
    import triton

    if not (gate.is_cuda and up.is_cuda):
        raise ValueError("Triton SiLU-and-mul requires CUDA tensors")
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError("gate and up tensors must have matching shape and dtype")
    if not (gate.is_contiguous() and up.is_contiguous()):
        raise ValueError("SiLU-and-mul expects contiguous tensors")
    output = torch.empty_like(gate)
    elements = gate.numel()
    _silu_and_mul_kernel[(triton.cdiv(elements, 256),)](
        gate,
        up,
        output,
        elements,
        BLOCK=256,
        num_warps=4,
    )
    return output


def gdn_gating(beta, step, a_log, dt_bias):
    """Fuse GDN beta sigmoid and decay parameterization into one kernel."""
    import torch
    import triton

    tensors = (beta, step, a_log, dt_bias)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton GDN gating requires CUDA tensors")
    if beta.shape != step.shape or beta.ndim < 1:
        raise ValueError("beta and step tensors must have matching shapes")
    heads = beta.shape[-1]
    if a_log.numel() != heads or dt_bias.numel() != heads:
        raise ValueError("GDN decay parameters must match the projected head count")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("GDN gating expects contiguous tensors")
    beta_output = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
    decay_output = torch.empty_like(beta_output)
    elements = beta.numel()
    _gdn_gating_kernel[(triton.cdiv(elements, 256),)](
        beta,
        step,
        a_log,
        dt_bias,
        beta_output,
        decay_output,
        elements,
        heads,
        BLOCK=256,
        num_warps=4,
    )
    return beta_output, decay_output


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _silu_and_mul_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        elements,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(output_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)

    @triton.jit
    def _gdn_gating_kernel(
        beta_ptr,
        step_ptr,
        a_log_ptr,
        dt_bias_ptr,
        beta_output_ptr,
        decay_output_ptr,
        elements,
        heads,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        head = offsets % heads
        beta = tl.load(beta_ptr + offsets, mask=mask).to(tl.float32)
        step = tl.load(step_ptr + offsets, mask=mask).to(tl.float32)
        step += tl.load(dt_bias_ptr + head, mask=mask).to(tl.float32)
        a_log = tl.load(a_log_ptr + head, mask=mask).to(tl.float32)
        softplus = tl.maximum(step, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(step))
        )
        tl.store(beta_output_ptr + offsets, tl.sigmoid(beta), mask=mask)
        tl.store(
            decay_output_ptr + offsets,
            -tl.exp(a_log) * softplus,
            mask=mask,
        )
except ImportError:
    _silu_and_mul_kernel = None
    _gdn_gating_kernel = None
