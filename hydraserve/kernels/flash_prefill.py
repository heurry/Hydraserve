"""The one explicitly permitted third-party kernel: FlashAttention prefill."""

from __future__ import annotations


def flash_attention_varlen(
    query,
    key,
    value,
    cu_seqlens,
    max_sequence_length: int,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
):
    """Run FlashAttention varlen GQA on packed ``[tokens, heads, dim]`` tensors."""
    if not (query.is_cuda and key.is_cuda and value.is_cuda and cu_seqlens.is_cuda):
        raise ValueError("FlashAttention prefill requires CUDA tensors")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("packed q/k/v tensors must have rank three")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("incompatible packed q/k/v shapes")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError(
            "prefill requires flash-attn; install the GPU optional dependencies"
        ) from exc
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens,
        cu_seqlens,
        max_sequence_length,
        max_sequence_length,
        softmax_scale=softmax_scale,
        causal=causal,
    )
