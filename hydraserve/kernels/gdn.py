"""HydraServe Triton fused recurrent Gated DeltaNet kernel."""

from __future__ import annotations

import os


_LEGACY_GDN_KERNELS = os.environ.get("HYDRASERVE_GDN_KERNEL", "blocked") == "legacy"


def legacy_gdn_kernels_enabled() -> bool:
    """Return whether the process selected the pre-blocking GDN kernels."""
    return _LEGACY_GDN_KERNELS


def causal_depthwise_conv(
    hidden,
    weight,
    state,
    *,
    next_state=None,
    split_widths=None,
):
    """Causal depthwise convolution with explicit fixed-length state."""
    import torch
    import triton

    tensors = (hidden, weight, state)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("Triton causal conv requires contiguous CUDA tensors")
    if hidden.ndim != 3 or weight.ndim != 2:
        raise ValueError("hidden=[B,T,C], weight=[C,K] are required")
    batch, sequence, channels = hidden.shape
    if weight.shape[0] != channels or state.shape != (batch, channels, weight.shape[1]):
        raise ValueError("invalid causal conv state or weight shape")
    if split_widths is not None:
        split_widths = tuple(int(width) for width in split_widths)
        if len(split_widths) != 3 or sum(split_widths) != channels:
            raise ValueError("causal conv split widths must contain three channel partitions")
    output = (
        torch.empty_like(hidden)
        if split_widths is None or _LEGACY_GDN_KERNELS
        else None
    )
    if next_state is None:
        next_state = torch.empty_like(state)
    elif (
        next_state.shape != state.shape
        or next_state.device != state.device
        or next_state.dtype != state.dtype
        or not next_state.is_contiguous()
    ):
        raise ValueError("next convolution state must match the input state")
    if _LEGACY_GDN_KERNELS:
        _causal_conv_scalar_kernel[(batch * sequence, channels)](
            hidden,
            weight,
            state,
            output,
            sequence,
            channels,
            KERNEL=weight.shape[1],
        )
        _causal_conv_scalar_state_kernel[(batch, channels)](
            hidden,
            state,
            next_state,
            sequence,
            channels,
            KERNEL=weight.shape[1],
        )
    else:
        block_channels = 256
        channel_blocks = triton.cdiv(channels, block_channels)
        if split_widths is None:
            _causal_conv_blocked_kernel[(batch * sequence, channel_blocks)](
                hidden,
                weight,
                state,
                output,
                sequence,
                channels,
                KERNEL=weight.shape[1],
                BLOCK_CHANNELS=block_channels,
                num_warps=8,
            )
        else:
            first_width, second_width, third_width = split_widths
            output = (
                torch.empty(
                    batch,
                    sequence,
                    first_width,
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
                torch.empty(
                    batch,
                    sequence,
                    second_width,
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
                torch.empty(
                    batch,
                    sequence,
                    third_width,
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
            )
            _causal_conv_blocked_split_kernel[(batch * sequence, channel_blocks)](
                hidden,
                weight,
                state,
                output[0],
                output[1],
                output[2],
                sequence,
                channels,
                FIRST_WIDTH=first_width,
                SECOND_WIDTH=second_width,
                KERNEL=weight.shape[1],
                BLOCK_CHANNELS=block_channels,
                num_warps=8,
            )
        _causal_conv_blocked_state_kernel[(batch, channel_blocks)](
            hidden,
            state,
            next_state,
            sequence,
            channels,
            KERNEL=weight.shape[1],
            BLOCK_CHANNELS=block_channels,
            num_warps=8,
        )
    if split_widths is not None and _LEGACY_GDN_KERNELS:
        output = output.split(split_widths, dim=-1)
    return output, next_state


def gated_delta_recurrent(query, key, value, log_decay, beta, state):
    """Advance a GDN state for one or more tokens entirely in a Triton kernel."""
    import torch
    import triton

    tensors = (query, key, value, log_decay, beta, state)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("Triton GDN requires CUDA tensors")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Triton GDN currently requires contiguous tensors")
    if query.shape != key.shape or query.shape[:2] != value.shape[:2]:
        raise ValueError("incompatible GDN q/k/v shapes")
    batch, sequence, query_heads, key_dim = query.shape
    heads = value.shape[2]
    value_dim = value.shape[-1]
    if heads % query_heads:
        raise ValueError("GDN value heads must be divisible by query/key heads")
    if state.shape != (batch, heads, key_dim, value_dim):
        raise ValueError("invalid recurrent state shape")
    if log_decay.shape != (batch, sequence, heads) or beta.shape != log_decay.shape:
        raise ValueError("invalid decay or beta shape")
    output = torch.empty_like(value)
    block_k = triton.next_power_of_2(key_dim)
    block_v = 16
    grid = (batch, heads, triton.cdiv(value_dim, block_v))
    kernel = (
        _gdn_recurrent_kernel
        if _LEGACY_GDN_KERNELS or sequence != 1
        else _gdn_recurrent_decode_kernel
    )
    kernel[grid](
        query,
        key,
        value,
        log_decay,
        beta,
        state,
        output,
        sequence,
        query_heads,
        heads,
        key_dim,
        value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        HEAD_RATIO=heads // query_heads,
        SCALE=key_dim**-0.5,
        num_warps=4,
    )
    return output, state


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _causal_conv_scalar_kernel(
        hidden_ptr,
        weight_ptr,
        state_ptr,
        output_ptr,
        sequence,
        channels,
        KERNEL: tl.constexpr,
    ):
        token_batch = tl.program_id(0)
        channel = tl.program_id(1)
        batch_id = token_batch // sequence
        token = token_batch % sequence
        accumulator = 0.0
        for kernel_index in range(0, KERNEL):
            source_token = token - (KERNEL - 1 - kernel_index)
            from_input = source_token >= 0
            input_offset = (batch_id * sequence + source_token) * channels + channel
            state_index = KERNEL + source_token
            state_offset = (batch_id * channels + channel) * KERNEL + state_index
            value = tl.load(
                hidden_ptr + input_offset,
                mask=from_input,
                other=0.0,
            )
            prior = tl.load(
                state_ptr + state_offset,
                mask=~from_input,
                other=0.0,
            )
            coefficient = tl.load(weight_ptr + channel * KERNEL + kernel_index)
            accumulator += tl.where(from_input, value, prior).to(tl.float32) * coefficient.to(tl.float32)
        activated = accumulator * tl.sigmoid(accumulator)
        tl.store(output_ptr + (batch_id * sequence + token) * channels + channel, activated)

    @triton.jit
    def _causal_conv_scalar_state_kernel(
        hidden_ptr,
        state_ptr,
        next_state_ptr,
        sequence,
        channels,
        KERNEL: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        channel = tl.program_id(1)
        for state_index in range(0, KERNEL):
            source_token = sequence - KERNEL + state_index
            from_input = source_token >= 0
            input_offset = (batch_id * sequence + source_token) * channels + channel
            prior_index = KERNEL + source_token
            prior_offset = (batch_id * channels + channel) * KERNEL + prior_index
            value = tl.load(hidden_ptr + input_offset, mask=from_input, other=0.0)
            prior = tl.load(state_ptr + prior_offset, mask=~from_input, other=0.0)
            destination = (batch_id * channels + channel) * KERNEL + state_index
            tl.store(next_state_ptr + destination, tl.where(from_input, value, prior))

    @triton.jit
    def _causal_conv_blocked_kernel(
        hidden_ptr,
        weight_ptr,
        state_ptr,
        output_ptr,
        sequence,
        channels,
        KERNEL: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        token_batch = tl.program_id(0)
        channel_offsets = (
            tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        )
        channel_mask = channel_offsets < channels
        batch_id = token_batch // sequence
        token = token_batch % sequence
        accumulator = tl.zeros((BLOCK_CHANNELS,), tl.float32)
        for kernel_index in range(0, KERNEL):
            source_token = token - (KERNEL - 1 - kernel_index)
            from_input = source_token >= 0
            input_offsets = (
                (batch_id * sequence + source_token) * channels + channel_offsets
            )
            state_index = KERNEL + source_token
            state_offsets = (
                (batch_id * channels + channel_offsets) * KERNEL + state_index
            )
            values = tl.load(
                hidden_ptr + input_offsets,
                mask=channel_mask & from_input,
                other=0.0,
            )
            prior = tl.load(
                state_ptr + state_offsets,
                mask=channel_mask & ~from_input,
                other=0.0,
            )
            coefficients = tl.load(
                weight_ptr + channel_offsets * KERNEL + kernel_index,
                mask=channel_mask,
                other=0.0,
            )
            accumulator += (
                tl.where(from_input, values, prior).to(tl.float32)
                * coefficients.to(tl.float32)
            )
        activated = accumulator * tl.sigmoid(accumulator)
        output_offsets = token_batch * channels + channel_offsets
        tl.store(output_ptr + output_offsets, activated, mask=channel_mask)

    @triton.jit
    def _causal_conv_blocked_state_kernel(
        hidden_ptr,
        state_ptr,
        next_state_ptr,
        sequence,
        channels,
        KERNEL: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        channel_offsets = (
            tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        )
        channel_mask = channel_offsets < channels
        for state_index in range(0, KERNEL):
            source_token = sequence - KERNEL + state_index
            from_input = source_token >= 0
            input_offsets = (
                (batch_id * sequence + source_token) * channels + channel_offsets
            )
            prior_index = KERNEL + source_token
            prior_offsets = (
                (batch_id * channels + channel_offsets) * KERNEL + prior_index
            )
            values = tl.load(
                hidden_ptr + input_offsets,
                mask=channel_mask & from_input,
                other=0.0,
            )
            prior = tl.load(
                state_ptr + prior_offsets,
                mask=channel_mask & ~from_input,
                other=0.0,
            )
            destinations = (
                (batch_id * channels + channel_offsets) * KERNEL + state_index
            )
            tl.store(
                next_state_ptr + destinations,
                tl.where(from_input, values, prior),
                mask=channel_mask,
            )

    @triton.jit
    def _causal_conv_blocked_split_kernel(
        hidden_ptr,
        weight_ptr,
        state_ptr,
        first_ptr,
        second_ptr,
        third_ptr,
        sequence,
        channels,
        FIRST_WIDTH: tl.constexpr,
        SECOND_WIDTH: tl.constexpr,
        KERNEL: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        token_batch = tl.program_id(0)
        channel_offsets = (
            tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        )
        channel_mask = channel_offsets < channels
        batch_id = token_batch // sequence
        token = token_batch % sequence
        accumulator = tl.zeros((BLOCK_CHANNELS,), tl.float32)
        for kernel_index in range(0, KERNEL):
            source_token = token - (KERNEL - 1 - kernel_index)
            from_input = source_token >= 0
            input_offsets = (
                (batch_id * sequence + source_token) * channels + channel_offsets
            )
            state_index = KERNEL + source_token
            state_offsets = (
                (batch_id * channels + channel_offsets) * KERNEL + state_index
            )
            values = tl.load(
                hidden_ptr + input_offsets,
                mask=channel_mask & from_input,
                other=0.0,
            )
            prior = tl.load(
                state_ptr + state_offsets,
                mask=channel_mask & ~from_input,
                other=0.0,
            )
            coefficients = tl.load(
                weight_ptr + channel_offsets * KERNEL + kernel_index,
                mask=channel_mask,
                other=0.0,
            )
            accumulator += (
                tl.where(from_input, values, prior).to(tl.float32)
                * coefficients.to(tl.float32)
            )
        activated = accumulator * tl.sigmoid(accumulator)
        first_mask = channel_offsets < FIRST_WIDTH
        second_offsets = channel_offsets - FIRST_WIDTH
        second_mask = (second_offsets >= 0) & (second_offsets < SECOND_WIDTH)
        third_offsets = second_offsets - SECOND_WIDTH
        third_mask = (third_offsets >= 0) & channel_mask
        tl.store(
            first_ptr + token_batch * FIRST_WIDTH + channel_offsets,
            activated,
            mask=first_mask,
        )
        tl.store(
            second_ptr + token_batch * SECOND_WIDTH + second_offsets,
            activated,
            mask=second_mask,
        )
        third_width = channels - FIRST_WIDTH - SECOND_WIDTH
        tl.store(
            third_ptr + token_batch * third_width + third_offsets,
            activated,
            mask=third_mask,
        )

    @triton.jit
    def _gdn_recurrent_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        decay_ptr,
        beta_ptr,
        state_ptr,
        output_ptr,
        sequence,
        query_heads,
        heads,
        key_dim,
        value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_RATIO: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        query_head_id = head_id // HEAD_RATIO
        value_block = tl.program_id(2)
        key_offsets = tl.arange(0, BLOCK_K)
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        key_mask = key_offsets < key_dim
        value_mask = value_offsets < value_dim
        state_offsets = (
            ((batch_id * heads + head_id) * key_dim + key_offsets[:, None]) * value_dim
            + value_offsets[None, :]
        )
        state_mask = key_mask[:, None] & value_mask[None, :]
        recurrent = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(tl.float32)
        for token in tl.range(0, sequence, 1):
            qk_base = (
                (batch_id * sequence + token) * query_heads + query_head_id
            ) * key_dim
            value_base = ((batch_id * sequence + token) * heads + head_id) * value_dim
            scalar_base = (batch_id * sequence + token) * heads + head_id
            query = tl.load(query_ptr + qk_base + key_offsets, mask=key_mask, other=0.0).to(tl.float32)
            key = tl.load(key_ptr + qk_base + key_offsets, mask=key_mask, other=0.0).to(tl.float32)
            q_norm = tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6)
            k_norm = tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
            query = query * q_norm * SCALE
            key = key * k_norm
            decay = tl.exp(tl.load(decay_ptr + scalar_base).to(tl.float32))
            beta_value = tl.load(beta_ptr + scalar_base).to(tl.float32)
            recurrent *= decay
            prediction = tl.sum(recurrent * key[:, None], axis=0)
            value = tl.load(value_ptr + value_base + value_offsets, mask=value_mask, other=0.0).to(tl.float32)
            delta = (value - prediction) * beta_value
            recurrent += key[:, None] * delta[None, :]
            result = tl.sum(recurrent * query[:, None], axis=0)
            tl.store(output_ptr + value_base + value_offsets, result, mask=value_mask)
        tl.store(state_ptr + state_offsets, recurrent, mask=state_mask)

    @triton.jit
    def _gdn_recurrent_decode_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        decay_ptr,
        beta_ptr,
        state_ptr,
        output_ptr,
        sequence,
        query_heads,
        heads,
        key_dim,
        value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_RATIO: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        query_head_id = head_id // HEAD_RATIO
        value_block = tl.program_id(2)
        key_offsets = tl.arange(0, BLOCK_K)
        value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
        key_mask = key_offsets < key_dim
        value_mask = value_offsets < value_dim
        state_offsets = (
            ((batch_id * heads + head_id) * key_dim + key_offsets[:, None])
            * value_dim
            + value_offsets[None, :]
        )
        state_mask = key_mask[:, None] & value_mask[None, :]
        recurrent = tl.load(
            state_ptr + state_offsets, mask=state_mask, other=0.0
        ).to(tl.float32)
        qk_base = (batch_id * query_heads + query_head_id) * key_dim
        value_base = (batch_id * heads + head_id) * value_dim
        scalar_base = batch_id * heads + head_id
        query = tl.load(
            query_ptr + qk_base + key_offsets, mask=key_mask, other=0.0
        ).to(tl.float32)
        key = tl.load(
            key_ptr + qk_base + key_offsets, mask=key_mask, other=0.0
        ).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6) * SCALE
        key *= tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
        decay = tl.exp(tl.load(decay_ptr + scalar_base).to(tl.float32))
        beta_value = tl.load(beta_ptr + scalar_base).to(tl.float32)
        recurrent *= decay
        prediction = tl.sum(recurrent * key[:, None], axis=0)
        value = tl.load(
            value_ptr + value_base + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        delta = (value - prediction) * beta_value
        recurrent += key[:, None] * delta[None, :]
        result = tl.sum(recurrent * query[:, None], axis=0)
        tl.store(output_ptr + value_base + value_offsets, result, mask=value_mask)
        tl.store(state_ptr + state_offsets, recurrent, mask=state_mask)
except ImportError:
    _gdn_recurrent_kernel = None
