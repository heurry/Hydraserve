"""Triton scatter kernel for writing projected K/V into physical cache pages."""

from __future__ import annotations


def write_paged_kv(key, value, positions, block_ids, key_cache, value_cache):
    import triton

    tensors = (key, value, positions, block_ids, key_cache, value_cache)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("paged KV write requires CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("paged KV write requires contiguous tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must have shape [tokens, heads, dim]")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("cache must have shape [blocks, block_size, heads, dim]")
    tokens, heads, head_dim = key.shape
    _, block_size, cache_heads, cache_dim = key_cache.shape
    if heads != cache_heads or head_dim != cache_dim:
        raise ValueError("projected KV and cache dimensions differ")
    _write_paged_kv_kernel[(tokens, heads)](
        key,
        value,
        positions,
        block_ids,
        key_cache,
        value_cache,
        tokens,
        heads,
        head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def write_paged_kv_batch(
    key, value, positions, block_table, key_cache, value_cache
):
    """Scatter one projected KV token per request with one Triton launch."""
    import triton

    tensors = (key, value, positions, block_table, key_cache, value_cache)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("batched paged KV write requires contiguous CUDA tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("batched key/value must have shape [batch, heads, dim]")
    if positions.shape != (key.shape[0],) or block_table.ndim != 2:
        raise ValueError("invalid batched KV positions or block table")
    if block_table.shape[0] != key.shape[0]:
        raise ValueError("block table batch does not match projected KV")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("cache must have shape [blocks, block_size, heads, dim]")
    batch, heads, head_dim = key.shape
    _, block_size, cache_heads, cache_dim = key_cache.shape
    if (heads, head_dim) != (cache_heads, cache_dim):
        raise ValueError("projected KV and cache dimensions differ")
    _write_paged_kv_batch_kernel[(batch, heads)](
        key,
        value,
        positions,
        block_table,
        key_cache,
        value_cache,
        key.stride(0),
        key.stride(1),
        block_table.stride(0),
        heads,
        head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def write_paged_kv_batch_quantized(
    key,
    value,
    key_scale,
    value_scale,
    positions,
    block_table,
    key_cache,
    value_cache,
    key_scale_cache,
    value_scale_cache,
):
    """Scatter quantized KV and its per-head scales with one Triton launch."""
    import torch
    import triton

    tensors = (
        key,
        value,
        key_scale,
        value_scale,
        positions,
        block_table,
        key_cache,
        value_cache,
        key_scale_cache,
        value_scale_cache,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("quantized paged KV write requires contiguous CUDA tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("batched key/value must have shape [batch, heads, dim]")
    batch, heads, head_dim = key.shape
    if key.dtype != torch.int8 or value.dtype != torch.int8:
        raise TypeError("quantized paged KV write requires INT8 key/value")
    if key_scale.shape != value_scale.shape or key_scale.shape != (batch, heads):
        raise ValueError("KV scales must have shape [batch, heads]")
    if positions.shape != (batch,) or block_table.ndim != 2:
        raise ValueError("invalid batched KV positions or block table")
    if block_table.shape[0] != batch:
        raise ValueError("block table batch does not match projected KV")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("cache must have shape [blocks, block_size, heads, dim]")
    if (
        key_scale_cache.shape != value_scale_cache.shape
        or key_scale_cache.ndim != 3
    ):
        raise ValueError("scale cache must have shape [blocks, block_size, heads]")
    _, block_size, cache_heads, cache_dim = key_cache.shape
    if (heads, head_dim) != (cache_heads, cache_dim):
        raise ValueError("projected KV and cache dimensions differ")
    if tuple(key_scale_cache.shape) != tuple(key_cache.shape[:-1]):
        raise ValueError("scale cache and KV cache dimensions differ")
    _write_paged_kv_batch_quantized_kernel[(batch, heads)](
        key,
        value,
        key_scale,
        value_scale,
        positions,
        block_table,
        key_cache,
        value_cache,
        key_scale_cache,
        value_scale_cache,
        key.stride(0),
        key.stride(1),
        key_scale.stride(0),
        block_table.stride(0),
        heads,
        head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _write_paged_kv_kernel(
        key_ptr,
        value_ptr,
        positions_ptr,
        block_ids_ptr,
        key_cache_ptr,
        value_cache_ptr,
        tokens,
        heads,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        mask = dimensions < head_dim
        position = tl.load(positions_ptr + token)
        logical_block = position // BLOCK_SIZE
        block_offset = position % BLOCK_SIZE
        physical_block = tl.load(block_ids_ptr + logical_block)
        source = (token * heads + head) * head_dim + dimensions
        destination = (
            ((physical_block * BLOCK_SIZE + block_offset) * heads + head) * head_dim
            + dimensions
        )
        key = tl.load(key_ptr + source, mask=mask)
        value = tl.load(value_ptr + source, mask=mask)
        tl.store(key_cache_ptr + destination, key, mask=mask)
        tl.store(value_cache_ptr + destination, value, mask=mask)

    @triton.jit
    def _write_paged_kv_batch_kernel(
        key_ptr,
        value_ptr,
        positions_ptr,
        table_ptr,
        key_cache_ptr,
        value_cache_ptr,
        stride_kb,
        stride_kh,
        stride_tb,
        heads,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        head = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        mask = dimensions < head_dim
        position = tl.load(positions_ptr + request)
        logical_block = position // BLOCK_SIZE
        block_offset = position % BLOCK_SIZE
        physical_block = tl.load(table_ptr + request * stride_tb + logical_block)
        source = request * stride_kb + head * stride_kh + dimensions
        destination = (
            ((physical_block * BLOCK_SIZE + block_offset) * heads + head) * head_dim
            + dimensions
        )
        projected_key = tl.load(key_ptr + source, mask=mask)
        projected_value = tl.load(value_ptr + source, mask=mask)
        tl.store(key_cache_ptr + destination, projected_key, mask=mask)
        tl.store(value_cache_ptr + destination, projected_value, mask=mask)

    @triton.jit
    def _write_paged_kv_batch_quantized_kernel(
        key_ptr,
        value_ptr,
        key_scale_ptr,
        value_scale_ptr,
        positions_ptr,
        table_ptr,
        key_cache_ptr,
        value_cache_ptr,
        key_scale_cache_ptr,
        value_scale_cache_ptr,
        stride_kb,
        stride_kh,
        stride_sb,
        stride_tb,
        heads,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        head = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        mask = dimensions < head_dim
        position = tl.load(positions_ptr + request)
        logical_block = position // BLOCK_SIZE
        block_offset = position % BLOCK_SIZE
        physical_block = tl.load(table_ptr + request * stride_tb + logical_block)
        source = request * stride_kb + head * stride_kh + dimensions
        destination = (
            ((physical_block * BLOCK_SIZE + block_offset) * heads + head) * head_dim
            + dimensions
        )
        projected_key = tl.load(key_ptr + source, mask=mask)
        projected_value = tl.load(value_ptr + source, mask=mask)
        tl.store(key_cache_ptr + destination, projected_key, mask=mask)
        tl.store(value_cache_ptr + destination, projected_value, mask=mask)
        scale_destination = (
            (physical_block * BLOCK_SIZE + block_offset) * heads + head
        )
        tl.store(
            key_scale_cache_ptr + scale_destination,
            tl.load(key_scale_ptr + request * stride_sb + head),
        )
        tl.store(
            value_scale_cache_ptr + scale_destination,
            tl.load(value_scale_ptr + request * stride_sb + head),
        )
except ImportError:
    _write_paged_kv_kernel = None
    _write_paged_kv_batch_kernel = None
    _write_paged_kv_batch_quantized_kernel = None
