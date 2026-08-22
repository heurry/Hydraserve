"""Fused paged-KV gather/scatter kernels used by the transfer staging path."""

from __future__ import annotations


def fused_gather_paged_kv(key_cache, value_cache, block_ids, start: int, end: int):
    """Gather ``[layer, K/V, token, head, dim]`` with one Triton launch."""
    import torch
    import triton
    import triton.language as tl

    if not key_cache.is_cuda or key_cache.shape != value_cache.shape:
        raise ValueError("fused KV gather requires matching CUDA caches")
    if key_cache.ndim != 5 or not 0 <= start < end:
        raise ValueError("invalid paged KV gather shape or range")
    layers, _blocks, block_size, heads, dim = key_cache.shape
    tokens = end - start
    output = torch.empty(
        (layers, 2, tokens, heads, dim),
        device=key_cache.device,
        dtype=key_cache.dtype,
    )

    @triton.jit
    def _gather(key_ptr, value_ptr, blocks_ptr, out_ptr, START: tl.constexpr,
                TOKENS: tl.constexpr, NUM_BLOCKS: tl.constexpr,
                BLOCK_SIZE: tl.constexpr, HEADS: tl.constexpr,
                DIM: tl.constexpr, WIDTH_BLOCK: tl.constexpr):
        layer_kv = tl.program_id(0)
        token_slot = tl.program_id(1)
        width_block = tl.program_id(2)
        layer = layer_kv // 2
        is_value = layer_kv % 2
        logical_token = START + token_slot
        physical = tl.load(blocks_ptr + logical_token // BLOCK_SIZE)
        width = width_block * WIDTH_BLOCK + tl.arange(0, WIDTH_BLOCK)
        mask = width < HEADS * DIM
        source_offset = (
            ((layer * NUM_BLOCKS + physical) * BLOCK_SIZE
             + logical_token % BLOCK_SIZE) * HEADS * DIM
            + width
        )
        key = tl.load(key_ptr + source_offset, mask=mask)
        value = tl.load(value_ptr + source_offset, mask=mask)
        out_offset = ((layer_kv * TOKENS + token_slot) * HEADS * DIM + width)
        tl.store(out_ptr + out_offset, tl.where(is_value == 1, value, key), mask=mask)

    width_block = 256
    _gather[
        (layers * 2, tokens, triton.cdiv(heads * dim, width_block))
    ](
        key_cache,
        value_cache,
        block_ids,
        output,
        START=start,
        TOKENS=tokens,
        NUM_BLOCKS=key_cache.shape[1],
        BLOCK_SIZE=block_size,
        HEADS=heads,
        DIM=dim,
        WIDTH_BLOCK=width_block,
    )
    return output


def fused_scatter_paged_kv(staging, key_cache, value_cache, block_ids, start: int) -> None:
    """Scatter a contiguous transfer staging tensor with one Triton launch."""
    import triton
    import triton.language as tl

    if not staging.is_cuda or not key_cache.is_cuda or key_cache.shape != value_cache.shape:
        raise ValueError("fused KV scatter requires CUDA tensors")
    if staging.ndim != 5 or staging.shape[:2] != (key_cache.shape[0], 2):
        raise ValueError("invalid KV staging tensor")
    layers, _, tokens, heads, dim = staging.shape
    if (heads, dim) != key_cache.shape[3:] or start < 0:
        raise ValueError("KV staging/cache shape mismatch")
    block_size = key_cache.shape[2]

    @triton.jit
    def _scatter(stage_ptr, key_ptr, value_ptr, blocks_ptr, START: tl.constexpr,
                 TOKENS: tl.constexpr, NUM_BLOCKS: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr, HEADS: tl.constexpr,
                 DIM: tl.constexpr, WIDTH_BLOCK: tl.constexpr):
        layer_kv = tl.program_id(0)
        token_slot = tl.program_id(1)
        width_block = tl.program_id(2)
        layer = layer_kv // 2
        is_value = layer_kv % 2
        logical_token = START + token_slot
        physical = tl.load(blocks_ptr + logical_token // BLOCK_SIZE)
        width = width_block * WIDTH_BLOCK + tl.arange(0, WIDTH_BLOCK)
        mask = width < HEADS * DIM
        stage_offset = ((layer_kv * TOKENS + token_slot) * HEADS * DIM + width)
        value = tl.load(stage_ptr + stage_offset, mask=mask)
        destination = (
            ((layer * NUM_BLOCKS + physical) * BLOCK_SIZE
             + logical_token % BLOCK_SIZE) * HEADS * DIM
            + width
        )
        tl.store(key_ptr + destination, value, mask=mask & (is_value == 0))
        tl.store(value_ptr + destination, value, mask=mask & (is_value == 1))

    width_block = 256
    _scatter[
        (layers * 2, tokens, triton.cdiv(heads * dim, width_block))
    ](
        staging,
        key_cache,
        value_cache,
        block_ids,
        START=start,
        TOKENS=tokens,
        NUM_BLOCKS=key_cache.shape[1],
        BLOCK_SIZE=block_size,
        HEADS=heads,
        DIM=dim,
        WIDTH_BLOCK=width_block,
    )
