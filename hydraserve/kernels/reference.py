"""Readable PyTorch equations used as correctness oracles for Triton kernels.

No Transformers, FlashAttention, FLA, vLLM, or other inference backend is used.
"""

from __future__ import annotations

import math


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("reference kernels require the 'gpu' optional dependencies") from exc
    return torch


def silu(x):
    torch = _torch()
    return x * torch.sigmoid(x)


def rms_norm(x, weight, eps: float = 1e-6, *, zero_centered: bool = True):
    torch = _torch()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    scale = 1.0 + weight.float() if zero_centered else weight.float()
    return (normalized * scale).to(x.dtype)


def gated_rms_norm(x, gate, weight, eps: float = 1e-6):
    return (rms_norm(x, weight, eps, zero_centered=False).float() * silu(gate.float())).to(x.dtype)


def l2_normalize(x, eps: float = 1e-6):
    torch = _torch()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def causal_depthwise_conv(x, weight, state=None, *, activate: bool = True):
    """Causal depthwise convolution implemented as an explicit state recurrence.

    Args:
        x: ``[batch, sequence, channels]``.
        weight: ``[channels, kernel]`` (cross-correlation order).
        state: optional prior ``[batch, channels, kernel]`` input history.
    """
    torch = _torch()
    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("invalid depthwise-convolution shapes")
    batch, sequence, channels = x.shape
    kernel = weight.shape[1]
    if state is None:
        state = torch.zeros(batch, channels, kernel, dtype=x.dtype, device=x.device)
    elif state.shape != (batch, channels, kernel):
        raise ValueError("invalid convolution state shape")
    else:
        state = state.clone()
    outputs = []
    for token in range(sequence):
        state = torch.cat((state[..., 1:], x[:, token].unsqueeze(-1)), dim=-1)
        value = (state.to(weight.dtype) * weight.unsqueeze(0)).sum(dim=-1).to(x.dtype)
        outputs.append(silu(value) if activate else value)
    return torch.stack(outputs, dim=1), state


def gated_delta_rule(query, key, value, log_decay, beta, initial_state=None):
    """Exact recurrent GDN equation in FP32.

    Shapes are ``q/k=[B,T,H,K]``, ``v=[B,T,H,V]``, ``g/beta=[B,T,H]`` and
    state ``[B,H,K,V]``. Query/key head expansion must happen before this call.
    """
    torch = _torch()
    if query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("incompatible GDN q/k/v shapes")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if log_decay.shape != (batch, sequence, heads) or beta.shape != log_decay.shape:
        raise ValueError("incompatible GDN decay/beta shapes")
    q = l2_normalize(query.float()) * (key_dim**-0.5)
    k = l2_normalize(key.float())
    v = value.float()
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    outputs = []
    for token in range(sequence):
        state = state * log_decay[:, token].float().exp()[..., None, None]
        prediction = (state * k[:, token, :, :, None]).sum(dim=-2)
        delta = (v[:, token] - prediction) * beta[:, token, :, None].float()
        state = state + k[:, token, :, :, None] * delta[..., None, :]
        outputs.append((state * q[:, token, :, :, None]).sum(dim=-2))
    return torch.stack(outputs, dim=1).to(query.dtype), state


def causal_gqa_attention(
    query, key, value, *, scale: float | None = None, query_start: int = 0
):
    """Unfused causal GQA attention used only as a kernel oracle."""
    torch = _torch()
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("attention tensors must be [B,T,H,D]")
    batch, query_sequence, query_heads, head_dim = query.shape
    key_sequence = key.shape[1]
    kv_heads = key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    groups = query_heads // kv_heads
    expanded_k = key.repeat_interleave(groups, dim=2)
    expanded_v = value.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", query.float(), expanded_k.float())
    scores *= scale if scale is not None else head_dim**-0.5
    query_positions = torch.arange(
        query_start, query_start + query_sequence, device=query.device
    )[:, None]
    key_positions = torch.arange(key_sequence, device=query.device)[None, :]
    mask = key_positions > query_positions
    scores.masked_fill_(mask, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return torch.einsum("bhts,bshd->bthd", probabilities, expanded_v.float()).to(query.dtype)


def apply_text_rope(x, positions, theta: float, rotary_dim: int):
    """Apply text-only partial RoPE to ``[B,T,H,D]``."""
    torch = _torch()
    if rotary_dim % 2:
        raise ValueError("rotary_dim must be even")
    frequencies = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if (
        positions.ndim != 2
        or positions.shape[1] != x.shape[1]
        or positions.shape[0] not in (1, x.shape[0])
    ):
        raise ValueError("positions must have shape [tokens] or [batch, tokens]")
    angles = positions.float().unsqueeze(-1) * frequencies.reshape(1, 1, -1)
    embedding = torch.cat((angles, angles), dim=-1)
    cos, sin = embedding.cos()[..., None, :], embedding.sin()[..., None, :]
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).to(x.dtype)


def paged_attention(query, key_cache, value_cache, block_table, sequence_lengths):
    """Reference decode attention for physical KV blocks."""
    torch = _torch()
    batch, query_heads, head_dim = query.shape
    block_size, kv_heads = key_cache.shape[1:3]
    groups = query_heads // kv_heads
    output = torch.empty_like(query)
    for request in range(batch):
        length = int(sequence_lengths[request])
        logical_blocks = (length + block_size - 1) // block_size
        physical = block_table[request, :logical_blocks].long()
        keys = key_cache[physical].reshape(-1, kv_heads, head_dim)[:length]
        values = value_cache[physical].reshape(-1, kv_heads, head_dim)[:length]
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        scores = torch.einsum("hd,thd->ht", query[request].float(), keys.float()) / math.sqrt(head_dim)
        probability = scores.softmax(dim=-1)
        output[request] = torch.einsum("ht,thd->hd", probability, values.float()).to(query.dtype)
    return output
