"""HydraServe Triton decode attention over non-contiguous KV blocks."""

from __future__ import annotations


def paged_attention(query, key_cache, value_cache, block_table, sequence_lengths):
    """Decode one token per request using online softmax.

    Layouts:
      - query: ``[batch, query_heads, head_dim]``
      - K/V: ``[physical_blocks, block_size, kv_heads, head_dim]``
      - block_table: ``[batch, max_logical_blocks]``
    """
    import torch
    import triton

    tensors = (query, key_cache, value_cache, block_table, sequence_lengths)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("Triton paged attention requires CUDA tensors")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4 or query.ndim != 3:
        raise ValueError("invalid paged-attention tensor shapes")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("paged attention currently requires contiguous tensors")
    batch, query_heads, head_dim = query.shape
    _, block_size, kv_heads, cache_dim = key_cache.shape
    if cache_dim != head_dim or query_heads % kv_heads:
        raise ValueError("incompatible attention heads or head dimension")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    max_context = block_table.shape[1] * block_size
    output = torch.empty_like(query)
    _paged_attention_kernel[(batch, query_heads)](
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        output,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        query_heads,
        kv_heads,
        head_dim,
        SCALE=head_dim**-0.5,
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        MAX_CONTEXT=max_context,
        num_warps=8,
    )
    return output


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        lengths_ptr,
        output_ptr,
        stride_qb,
        stride_qh,
        stride_cb,
        stride_ct,
        stride_ch,
        stride_tb,
        stride_ob,
        stride_oh,
        query_heads,
        kv_heads,
        head_dim,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < head_dim
        group_size = query_heads // kv_heads
        kv_head = query_head // group_size
        query = tl.load(
            query_ptr + request * stride_qb + query_head * stride_qh + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        length = tl.load(lengths_ptr + request)
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for token in tl.range(0, MAX_CONTEXT, 1):
            valid = token < length
            logical_block = token // BLOCK_SIZE
            block_offset = token % BLOCK_SIZE
            physical_block = tl.load(
                table_ptr + request * stride_tb + logical_block, mask=valid, other=0
            )
            cache_offsets = (
                physical_block * stride_cb
                + block_offset * stride_ct
                + kv_head * stride_ch
                + dimensions
            )
            key = tl.load(key_ptr + cache_offsets, mask=valid & dimension_mask, other=0.0).to(tl.float32)
            score = tl.sum(query * key, axis=0) * SCALE
            score = tl.where(valid, score, -float("inf"))
            next_maximum = tl.maximum(maximum, score)
            old_scale = tl.exp(maximum - next_maximum)
            probability = tl.where(valid, tl.exp(score - next_maximum), 0.0)
            value = tl.load(value_ptr + cache_offsets, mask=valid & dimension_mask, other=0.0).to(tl.float32)
            accumulator = accumulator * old_scale + probability * value
            denominator = denominator * old_scale + probability
            maximum = next_maximum
        result = accumulator / denominator
        tl.store(
            output_ptr + request * stride_ob + query_head * stride_oh + dimensions,
            result,
            mask=dimension_mask,
        )
except ImportError:
    _paged_attention_kernel = None
