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
        BLOCK_T=16,
        MAX_CONTEXT=max_context,
        num_warps=8,
    )
    return output


def paged_prefill_attention(
    query, key_cache, value_cache, block_table, *, query_start: int | object
):
    """Causal multi-token attention over paged history using the decode kernel.

    Each ``[request, query_position]`` pair becomes one online-softmax program
    with a different logical context length. This avoids materializing an
    attention-score matrix for continuation chunks.
    """
    import torch

    if query.ndim != 4 or block_table.ndim != 2:
        raise ValueError("query/table must be [B,T,H,D] and [B,blocks]")
    batch, tokens, query_heads, head_dim = query.shape
    if block_table.shape[0] != batch:
        raise ValueError("block table batch does not match query")
    starts = torch.as_tensor(query_start, device=query.device, dtype=torch.int32)
    if starts.ndim == 0:
        starts = starts.expand(batch)
    if starts.shape != (batch,) or bool((starts < 0).any()):
        raise ValueError("query_start must be non-negative scalar or [batch]")
    tables = (
        block_table[:, None, :]
        .expand(batch, tokens, block_table.shape[1])
        .reshape(batch * tokens, block_table.shape[1])
        .contiguous()
    )
    lengths = (
        starts[:, None]
        + torch.arange(1, tokens + 1, device=query.device, dtype=torch.int32)[None, :]
    ).reshape(-1).contiguous()
    flattened = query.reshape(batch * tokens, query_heads, head_dim).contiguous()
    if query.is_cuda:
        import os

        if (
            os.environ.get("HYDRASERVE_PAGED_PREFILL") == "reference"
            or head_dim < 16
        ):
            output = paged_attention(
                flattened, key_cache, value_cache, tables, lengths
            )
        else:
            output = paged_prefill_attention_tiled(
                query, key_cache, value_cache, block_table, query_start=starts
            )
    else:
        from hydraserve.kernels.reference import paged_attention as reference_paged_attention

        output = reference_paged_attention(
            flattened, key_cache, value_cache, tables, lengths
        )
    return output.reshape_as(query)


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
        BLOCK_T: tl.constexpr,
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
        for token_start in tl.range(0, MAX_CONTEXT, BLOCK_T):
            tokens = token_start + tl.arange(0, BLOCK_T)
            valid = tokens < length
            logical_block = tokens // BLOCK_SIZE
            block_offset = tokens % BLOCK_SIZE
            physical_block = tl.load(
                table_ptr + request * stride_tb + logical_block,
                mask=valid,
                other=0,
            )
            cache_offsets = (
                physical_block[:, None] * stride_cb
                + block_offset[:, None] * stride_ct
                + kv_head * stride_ch
                + dimensions[None, :]
            )
            tile_mask = valid[:, None] & dimension_mask[None, :]
            key = tl.load(
                key_ptr + cache_offsets, mask=tile_mask, other=0.0
            ).to(tl.float32)
            score = tl.sum(query[None, :] * key, axis=1) * SCALE
            score = tl.where(valid, score, -float("inf"))
            tile_maximum = tl.max(score, axis=0)
            next_maximum = tl.maximum(maximum, tile_maximum)
            old_scale = tl.exp(maximum - next_maximum)
            probability = tl.where(valid, tl.exp(score - next_maximum), 0.0)
            value = tl.load(
                value_ptr + cache_offsets, mask=tile_mask, other=0.0
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(
                probability[:, None] * value, axis=0
            )
            denominator = denominator * old_scale + tl.sum(probability, axis=0)
            maximum = next_maximum
        result = accumulator / denominator
        tl.store(
            output_ptr + request * stride_ob + query_head * stride_oh + dimensions,
            result,
            mask=dimension_mask,
        )
except ImportError:
    _paged_attention_kernel = None



def paged_attention_splitk(
    query,
    key_cache,
    value_cache,
    block_table,
    sequence_lengths,
    *,
    num_splits: int = 4,
    block_t: int = 64,
    num_warps: int = 4,
):
    """FlashDecoding-style split-K decode attention over paged KV.

    Each ``(request, head, split)`` program reduces a contiguous KV range into
    partial online-softmax state (max, sum, weighted acc); a second kernel
    merges the partials. The reference kernel's sequential full-context scan
    is parallelized across ``num_splits`` programs with a wider ``block_t``.
    """
    import torch
    import triton

    tensors = (query, key_cache, value_cache, block_table, sequence_lengths)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("paged attention requires CUDA tensors")
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
    if block_t & (block_t - 1) or block_t % block_size:
        raise ValueError("block_t must be a power of two and a multiple of block_size")
    if num_splits <= 0 or num_splits & (num_splits - 1):
        raise ValueError("num_splits must be a positive power of two")
    max_context = block_table.shape[1] * block_size
    max_tiles = (max_context + block_t - 1) // block_t
    tiles_per_split = (max_tiles + num_splits - 1) // num_splits
    partial_m = torch.empty(
        (num_splits, batch, query_heads), device=query.device, dtype=torch.float32
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        (num_splits, batch, query_heads, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(query)
    block_d = triton.next_power_of_2(head_dim)
    _paged_attention_split_kernel[(batch, query_heads, num_splits)](
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        partial_m,
        partial_l,
        partial_acc,
        batch,
        query_heads,
        kv_heads,
        head_dim,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        block_table.stride(0),
        SCALE=head_dim**-0.5,
        BLOCK_SIZE=block_size,
        BLOCK_D=block_d,
        BLOCK_T=block_t,
        MAX_TILES=max_tiles,
        SPLIT_TILES=tiles_per_split,
        num_warps=num_warps,
    )
    _paged_attention_reduce_kernel[(batch, query_heads)](
        partial_m,
        partial_l,
        partial_acc,
        output,
        batch,
        query_heads,
        head_dim,
        output.stride(0),
        output.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_attention_split_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        lengths_ptr,
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        batch,
        query_heads,
        kv_heads,
        head_dim,
        stride_qb,
        stride_qh,
        stride_cb,
        stride_ct,
        stride_ch,
        stride_tb,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        MAX_TILES: tl.constexpr,
        SPLIT_TILES: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        split = tl.program_id(2)
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
        tile_start = split * SPLIT_TILES
        for tile_index in tl.range(
            tile_start, tl.minimum(tile_start + SPLIT_TILES, MAX_TILES), 1
        ):
            tokens = tile_index * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = tokens < length
            logical_block = tokens // BLOCK_SIZE
            block_offset = tokens % BLOCK_SIZE
            physical_block = tl.load(
                table_ptr + request * stride_tb + logical_block,
                mask=valid,
                other=0,
            )
            cache_offsets = (
                physical_block[:, None] * stride_cb
                + block_offset[:, None] * stride_ct
                + kv_head * stride_ch
                + dimensions[None, :]
            )
            tile_mask = valid[:, None] & dimension_mask[None, :]
            key = tl.load(
                key_ptr + cache_offsets,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            score = tl.sum(query[None, :] * key, axis=1) * SCALE
            score = tl.where(valid, score, -float("inf"))
            tile_maximum = tl.max(score, axis=0)
            next_maximum = tl.maximum(maximum, tile_maximum)
            # Guard the -inf - -inf case (empty tile before any valid one).
            old_scale = tl.where(
                next_maximum == maximum, 1.0, tl.exp(maximum - next_maximum)
            )
            probability = tl.where(valid, tl.exp(score - next_maximum), 0.0)
            value = tl.load(
                value_ptr + cache_offsets,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(
                probability[:, None] * value, axis=0
            )
            denominator = denominator * old_scale + tl.sum(probability, axis=0)
            maximum = next_maximum
        state_row = request * query_heads + query_head
        tl.store(partial_m_ptr + split * (batch * query_heads) + state_row, maximum)
        tl.store(partial_l_ptr + split * (batch * query_heads) + state_row, denominator)
        tl.store(
            partial_acc_ptr
            + split * (batch * query_heads * BLOCK_D)
            + state_row * BLOCK_D
            + dimensions,
            accumulator,
            mask=dimension_mask,
        )

    @triton.jit
    def _paged_attention_reduce_kernel(
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        output_ptr,
        batch,
        query_heads,
        head_dim,
        stride_ob,
        stride_oh,
        NUM_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_head = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < head_dim
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        row = request * query_heads + query_head
        for split in tl.range(0, NUM_SPLITS, 1):
            split_m = tl.load(partial_m_ptr + split * (batch * query_heads) + row)
            split_l = tl.load(partial_l_ptr + split * (batch * query_heads) + row)
            active = split_l > 0.0
            next_maximum = tl.where(active, tl.maximum(maximum, split_m), maximum)
            old_scale = tl.where(
                next_maximum == maximum, 1.0, tl.exp(maximum - next_maximum)
            )
            split_scale = tl.where(active, tl.exp(split_m - next_maximum), 0.0)
            accumulator = accumulator * old_scale + tl.load(
                partial_acc_ptr
                + split * (batch * query_heads * BLOCK_D)
                + row * BLOCK_D
                + dimensions,
                mask=dimension_mask,
                other=0.0,
            ) * split_scale
            denominator = denominator * old_scale + split_l * split_scale
            maximum = next_maximum
        result = accumulator / denominator
        tl.store(
            output_ptr + request * stride_ob + query_head * stride_oh + dimensions,
            result,
            mask=dimension_mask,
        )
except ImportError:
    _paged_attention_split_kernel = None
    _paged_attention_reduce_kernel = None



def paged_prefill_attention_tiled(
    query, key_cache, value_cache, block_table, *, query_start
):
    """Block-causal prefill attention with KV-tile reuse (B2).

    One program per (request, query tile, query head): each KV tile is
    loaded once and reused across BLOCK_M queries, replacing the per-query
    program of the flattened decode-kernel path. Causal masking is
    elementwise (key position <= query position); tiles beyond the block
    table width are masked out of the loads.
    """
    import torch
    import triton

    if query.ndim != 4 or block_table.ndim != 2:
        raise ValueError("query/table must be [B,T,H,D] and [B,blocks]")
    if query.device != key_cache.device or not query.is_cuda:
        raise ValueError("tiled paged prefill requires CUDA tensors")
    if not all(t.is_contiguous() for t in (query, key_cache, value_cache, block_table)):
        raise ValueError("tiled paged prefill requires contiguous tensors")
    batch, tokens, query_heads, head_dim = query.shape
    _, block_size, kv_heads, cache_dim = key_cache.shape
    if cache_dim != head_dim or query_heads % kv_heads:
        raise ValueError("incompatible attention heads or head dimension")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    starts = torch.as_tensor(query_start, device=query.device, dtype=torch.int32)
    if starts.ndim == 0:
        starts = starts.expand(batch)
    if starts.shape != (batch,) or bool((starts < 0).any()):
        raise ValueError("query_start must be non-negative scalar or [batch]")
    starts = starts.contiguous()
    max_context = block_table.shape[1] * block_size
    block_m, block_n = 32, 32
    output = torch.empty_like(query)
    _paged_prefill_tiled_kernel[
        (batch, triton.cdiv(tokens, block_m), query_heads)
    ](
        query,
        key_cache,
        value_cache,
        block_table,
        starts,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        SCALE=head_dim**-0.5,
        BLOCK_SIZE=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        MAX_CONTEXT=max_context,
        num_warps=4,
    )
    return output


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_prefill_tiled_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        starts_ptr,
        output_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_cb,
        stride_ct,
        stride_ch,
        stride_tb,
        stride_ob,
        stride_ot,
        stride_oh,
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
    ):
        request = tl.program_id(0)
        query_tile = tl.program_id(1)
        query_head = tl.program_id(2)
        rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < tokens
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < head_dim
        group_size = query_heads // kv_heads
        kv_head = query_head // group_size
        query = tl.load(
            query_ptr
            + request * stride_qb
            + rows[:, None] * stride_qt
            + query_head * stride_qh
            + dimensions[None, :],
            mask=row_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        start = tl.load(starts_ptr + request)
        q_positions = (start + rows)[:, None]
        maximum = tl.full((BLOCK_M, 1), float("-inf"), dtype=tl.float32)
        denominator = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        keys = tl.arange(0, BLOCK_N)
        for tile_index in tl.range(0, MAX_CONTEXT // BLOCK_N, 1, num_stages=1):
            key_positions = tile_index * BLOCK_N + keys
            in_table = key_positions < MAX_CONTEXT
            logical_block = key_positions // BLOCK_SIZE
            block_offset = key_positions % BLOCK_SIZE
            physical_block = tl.load(
                table_ptr + request * stride_tb + logical_block,
                mask=in_table,
                other=0,
            )
            cache_offsets = (
                physical_block[:, None] * stride_cb
                + block_offset[:, None] * stride_ct
                + kv_head * stride_ch
                + dimensions[None, :]
            )
            tile_mask = in_table[:, None] & dimension_mask[None, :]
            key = tl.load(
                key_ptr + cache_offsets,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            scores = tl.dot(query, tl.trans(key)) * SCALE
            causal = in_table[None, :] & (key_positions[None, :] <= q_positions)
            scores = tl.where(causal, scores, float("-inf"))
            tile_maximum = tl.max(scores, axis=1)[:, None]
            next_maximum = tl.maximum(maximum, tile_maximum)
            old_scale = tl.where(
                next_maximum == maximum, 1.0, tl.exp(maximum - next_maximum)
            )
            probability = tl.where(causal, tl.exp(scores - next_maximum), 0.0)
            value = tl.load(
                value_ptr + cache_offsets,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.dot(probability, value)
            denominator = denominator * old_scale + tl.sum(
                probability, axis=1
            )[:, None]
            maximum = next_maximum
        result = accumulator / denominator
        tl.store(
            output_ptr
            + request * stride_ob
            + rows[:, None] * stride_ot
            + query_head * stride_oh
            + dimensions[None, :],
            result,
            mask=row_mask[:, None] & dimension_mask[None, :],
        )
except ImportError:
    _paged_prefill_tiled_kernel = None
