from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.config import LayerKind
from hydraserve.cache import (
    CostAwarePrefixPolicy,
    KVBlockManager,
    PagedKVCache,
    PrefixCache,
)
from hydraserve.model.runtime import QwenTextRuntime, RuntimeState
from hydraserve.model.weights import LANGUAGE_PREFIX, layer_prefix


def make_weights(model, *, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(19)

    def random(shape, scale=0.04):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype) * scale

    weights = {
        f"{LANGUAGE_PREFIX}.embed_tokens.weight": random((model.vocab_size, model.hidden_size)),
        f"{LANGUAGE_PREFIX}.norm.weight": random((model.hidden_size,), 0.01),
    }
    for layer_index, kind in enumerate(model.layer_types):
        prefix = layer_prefix(layer_index)
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": random((model.hidden_size,), 0.01),
                f"{prefix}.post_attention_layernorm.weight": random((model.hidden_size,), 0.01),
                f"{prefix}.mlp.gate_proj.weight": random((model.intermediate_size, model.hidden_size)),
                f"{prefix}.mlp.up_proj.weight": random((model.intermediate_size, model.hidden_size)),
                f"{prefix}.mlp.down_proj.weight": random((model.hidden_size, model.intermediate_size)),
            }
        )
        if kind is LayerKind.FULL_ATTENTION:
            attention = f"{prefix}.self_attn"
            weights.update(
                {
                    f"{attention}.q_proj.weight": random(
                        (model.num_attention_heads * model.head_dim * 2, model.hidden_size)
                    ),
                    f"{attention}.k_proj.weight": random(
                        (model.num_kv_heads * model.head_dim, model.hidden_size)
                    ),
                    f"{attention}.v_proj.weight": random(
                        (model.num_kv_heads * model.head_dim, model.hidden_size)
                    ),
                    f"{attention}.o_proj.weight": random(
                        (model.hidden_size, model.num_attention_heads * model.head_dim)
                    ),
                    f"{attention}.q_norm.weight": random((model.head_dim,), 0.01),
                    f"{attention}.k_norm.weight": random((model.head_dim,), 0.01),
                }
            )
        else:
            attention = f"{prefix}.linear_attn"
            weights.update(
                {
                    f"{attention}.in_proj_qkv.weight": random(
                        (model.linear_conv_width, model.hidden_size)
                    ),
                    f"{attention}.in_proj_z.weight": random(
                        (model.linear_value_width, model.hidden_size)
                    ),
                    f"{attention}.in_proj_b.weight": random(
                        (model.linear_num_value_heads, model.hidden_size)
                    ),
                    f"{attention}.in_proj_a.weight": random(
                        (model.linear_num_value_heads, model.hidden_size)
                    ),
                    f"{attention}.conv1d.weight": random(
                        (model.linear_conv_width, 1, model.linear_conv_kernel_dim)
                    ),
                    f"{attention}.A_log": torch.zeros(
                        model.linear_num_value_heads, device=device, dtype=torch.float32
                    ),
                    f"{attention}.dt_bias": torch.zeros(
                        model.linear_num_value_heads, device=device, dtype=torch.float32
                    ),
                    f"{attention}.norm.weight": torch.ones(
                        model.linear_value_head_dim, device=device, dtype=dtype
                    ),
                    f"{attention}.out_proj.weight": random(
                        (model.hidden_size, model.linear_value_width)
                    ),
                }
            )
    return weights


def test_whole_prefill_matches_token_by_token_decode(tiny_model) -> None:
    weights = make_weights(tiny_model)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    token_ids = torch.tensor([[3, 7, 11, 5, 2]])
    full_logits, full_state = runtime.forward(token_ids)
    state = RuntimeState()
    pieces = []
    for token in token_ids[0]:
        logits, state = runtime.forward(token.reshape(1, 1), state)
        pieces.append(logits)
    incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(incremental, full_logits, atol=2e-5, rtol=2e-5)
    assert state.sequence_length == full_state.sequence_length == token_ids.shape[1]
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(state.recurrent[layer], full_state.recurrent[layer])
        torch.testing.assert_close(state.convolution[layer], full_state.convolution[layer])


def test_independent_lm_head_is_used_for_logits(tiny_model) -> None:
    weights = make_weights(tiny_model)
    weights["lm_head.weight"] = torch.zeros_like(
        weights[f"{LANGUAGE_PREFIX}.embed_tokens.weight"]
    )
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    logits, _ = runtime.forward(torch.tensor([[3, 7]]))
    assert torch.count_nonzero(logits) == 0


def test_chunked_prefill_with_paged_history_matches_whole_prefill(tiny_model) -> None:
    weights = make_weights(tiny_model)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    token_ids = torch.tensor([[3, 7, 11, 5, 2]])
    expected, expected_state = runtime.forward(token_ids)
    cache = PagedKVCache(
        tiny_model,
        KVBlockManager(8, block_size=2),
        device="cpu",
        dtype=torch.float32,
    )
    cache.allocate(17, token_ids.shape[1])
    actual, actual_state = runtime.prefill(
        token_ids,
        chunk_size=2,
        paged_cache=cache,
        request_id=17,
    )
    torch.testing.assert_close(actual[:, -1], expected[:, -1], atol=2e-5, rtol=2e-5)
    assert actual_state.sequence_length == expected_state.sequence_length
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            actual_state.recurrent[layer], expected_state.recurrent[layer]
        )
        torch.testing.assert_close(
            actual_state.convolution[layer], expected_state.convolution[layer]
        )


def test_prefix_kv_hit_preserves_runtime_logits_and_gdn_state(tiny_model) -> None:
    weights = make_weights(tiny_model)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    prefix = PrefixCache(
        block_size=2,
        max_blocks=4,
        policy=CostAwarePrefixPolicy(minimum_frequency=1),
    )
    cache = PagedKVCache(
        tiny_model,
        KVBlockManager(12, block_size=2),
        device="cpu",
        dtype=torch.float32,
        prefix_cache=prefix,
    )
    first_ids = torch.tensor([[3, 7, 11, 5]])
    cache.allocate(1, 4, token_ids=first_ids[0].tolist())
    first_logits, first_state = runtime.prefill(
        first_ids, chunk_size=2, paged_cache=cache, request_id=1
    )
    cache.publish_prefix(1, first_ids[0].tolist())
    cached_blocks = cache.block_manager.get(1).block_ids[:2]
    cache.free(1)

    second_ids = torch.tensor([[3, 7, 11, 5, 2]])
    expected_logits, expected_state = runtime.prefill(
        second_ids, chunk_size=2
    )
    allocation = cache.allocate(2, 5, token_ids=second_ids[0].tolist())
    assert allocation.block_ids[:2] == cached_blocks
    assert cache.matched_prefix_tokens(2) == 4
    actual_logits, actual_state = runtime.prefill(
        second_ids, chunk_size=2, paged_cache=cache, request_id=2
    )
    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-5, rtol=2e-5)
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            actual_state.recurrent[layer], expected_state.recurrent[layer]
        )
        torch.testing.assert_close(
            actual_state.convolution[layer], expected_state.convolution[layer]
        )
    cache.free(2)
    assert actual_state.keys == actual_state.values == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_cuda_runtime_matches_cpu(tiny_model) -> None:
    cpu_weights = make_weights(tiny_model)
    gpu_weights = {
        name: tensor.to(
            device="cuda",
            dtype=torch.float32 if name.endswith((".A_log", ".dt_bias")) else torch.bfloat16,
        )
        for name, tensor in cpu_weights.items()
    }
    cpu = QwenTextRuntime(
        tiny_model, cpu_weights, use_triton=False, use_flash_attention=False
    )
    gpu = QwenTextRuntime(
        tiny_model, gpu_weights, use_triton=True, use_flash_attention=False
    )
    tokens = torch.tensor([[1, 9, 4, 3]])
    expected, _ = cpu.forward(tokens)
    actual, _ = gpu.forward(tokens.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_heterogeneous_batched_decode_matches_sequential(tiny_model) -> None:
    weights = make_weights(tiny_model, device="cuda", dtype=torch.bfloat16)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=True, use_flash_attention=False
    )

    def prepare():
        manager = KVBlockManager(32, block_size=4)
        cache = PagedKVCache(tiny_model, manager, device="cuda", dtype=torch.bfloat16)
        states = []
        for request_id, prompt in ((10, [1, 2, 3]), (11, [4, 5])):
            cache.allocate(request_id, len(prompt))
            _, state = runtime.forward(
                torch.tensor([prompt], device="cuda"),
                paged_cache=cache,
                request_id=request_id,
            )
            states.append(state)
        return cache, states

    batch_cache, batch_states = prepare()
    sequential_cache, sequential_states = prepare()
    for cache in (batch_cache, sequential_cache):
        cache.reserve_append(10)
        cache.reserve_append(11)
    tokens = torch.tensor([[7], [8]], device="cuda")
    recurrent_pointers = {
        (row, layer): state.recurrent[layer].data_ptr()
        for row, state in enumerate(batch_states)
        for layer in tiny_model.linear_layer_indices
    }
    convolution_pointers = {
        (row, layer): state.convolution[layer].data_ptr()
        for row, state in enumerate(batch_states)
        for layer in tiny_model.linear_layer_indices
    }
    metadata_calls = 0
    original_metadata = batch_cache.batch_metadata

    def counted_metadata(*args, **kwargs):
        nonlocal metadata_calls
        metadata_calls += 1
        return original_metadata(*args, **kwargs)

    batch_cache.batch_metadata = counted_metadata
    batched, _ = runtime.decode_batch(tokens, batch_states, batch_cache, (10, 11))
    assert metadata_calls == 1
    assert recurrent_pointers == {
        (row, layer): state.recurrent[layer].data_ptr()
        for row, state in enumerate(batch_states)
        for layer in tiny_model.linear_layer_indices
    }
    assert convolution_pointers == {
        (row, layer): state.convolution[layer].data_ptr()
        for row, state in enumerate(batch_states)
        for layer in tiny_model.linear_layer_indices
    }
    sequential = []
    for row, request_id in enumerate((10, 11)):
        logits, _ = runtime.forward(
            tokens[row : row + 1],
            sequential_states[row],
            paged_cache=sequential_cache,
            request_id=request_id,
        )
        sequential.append(logits)
    expected = torch.cat(sequential, dim=0)
    torch.testing.assert_close(batched, expected, atol=3e-2, rtol=3e-2)
