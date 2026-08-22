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


def paged_flash_prefill(
    query,
    key_pages,
    value_pages,
    block_table,
    cache_seqlens,
    *,
    causal: bool = True,
):
    """FlashAttention-style attention over paged KV for continuation chunks.

    Replaces the O(n^2) decode-style ``paged_prefill_attention`` fallback: each
    KV tile is loaded into SRAM once and reused across a whole block of query
    positions (B2 / W6), so chunked prefill is no longer a slow path.

    Layouts:
      - query: ``[batch, tokens, heads, head_dim]``
      - key/value_pages: ``[blocks, block_size, kv_heads, head_dim]``
      - block_table: ``[batch, max_logical_blocks]`` (int32, -1 padded)
      - cache_seqlens: ``[batch]`` total KV tokens to attend over
    """
    import torch

    if not (query.is_cuda and key_pages.is_cuda and value_pages.is_cuda):
        raise ValueError("paged flash prefill requires CUDA tensors")
    if query.ndim != 4 or key_pages.ndim != 4 or value_pages.ndim != 4:
        raise ValueError("expected [B,T,H,D] query and [blocks,block_size,kv_heads,D] pages")
    if key_pages.shape != value_pages.shape:
        raise ValueError("key and value pages must share a shape")
    batch, _tokens, _heads, head_dim = query.shape
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block table batch does not match query")
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError as exc:
        raise RuntimeError("paged flash prefill requires flash-attn") from exc
    query = query.contiguous()
    table = torch.as_tensor(block_table, device=query.device, dtype=torch.int32).contiguous()
    seqlens = torch.as_tensor(
        cache_seqlens, device=query.device, dtype=torch.int32
    ).contiguous()
    if seqlens.ndim == 0:
        seqlens = seqlens.expand(batch).contiguous()
    if seqlens.shape != (batch,):
        raise ValueError("cache sequence lengths must be scalar or [batch]")
    # With block_table set, FlashAttention expects the physical paged layout
    # directly: [num_blocks, page_block_size, kv_heads, head_dim]. Expanding it
    # to [batch, ...] duplicates the whole cache and violates the API layout.
    return flash_attn_with_kvcache(
        query,
        key_pages,
        value_pages,
        cache_seqlens=seqlens,
        block_table=table,
        causal=causal,
        softmax_scale=head_dim**-0.5,
    )
