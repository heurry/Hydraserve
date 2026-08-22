from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.config import LayerKind
from hydraserve.cache import (
    CostAwarePrefixPolicy,
    GpuLinearStatePool,
    KVBlockManager,
    PagedKVCache,
    PrefixCache,
)
from hydraserve.model.runtime import QwenTextRuntime, RuntimeState
from hydraserve.model.weights import LANGUAGE_PREFIX, layer_prefix
from hydraserve.model.weights import BlockScaledFP8Weight, PackedInt4Weight


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


def test_prefill_reports_completed_chunk_boundaries(tiny_model) -> None:
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    completed = []
    runtime.prefill(
        torch.tensor([[3, 7, 11, 5, 2]]),
        chunk_size=2,
        chunk_callback=lambda start, end, state: completed.append(
            (start, end, state.sequence_length)
        ),
    )
    assert completed == [(0, 2, 2), (2, 4, 4), (4, 5, 5)]


def test_fused_projections_match_unfused_runtime(tiny_model) -> None:
    weights = make_weights(tiny_model)
    fused = QwenTextRuntime(
        tiny_model,
        weights,
        use_triton=False,
        use_flash_attention=False,
        fuse_projections=True,
    )
    unfused = QwenTextRuntime(
        tiny_model,
        weights,
        use_triton=False,
        use_flash_attention=False,
        fuse_projections=False,
    )
    full_layer = tiny_model.full_attention_layer_indices[0]
    attention = f"{layer_prefix(full_layer)}.self_attn"
    assert f"{attention}.qkv_proj.weight" in fused.weights
    assert f"{layer_prefix(0)}.mlp.gate_up_proj.weight" in fused.weights
    assert f"{layer_prefix(0)}.linear_attn.in_proj_qkvz.weight" in fused.weights
    assert f"{layer_prefix(0)}.linear_attn.in_proj_ba.weight" in fused.weights
    assert f"{attention}.qkv_proj.weight" not in unfused.weights

    token_ids = torch.tensor([[3, 7, 11, 5, 2]])
    actual, actual_state = fused.forward(token_ids)
    expected, expected_state = unfused.forward(token_ids)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            actual_state.recurrent[layer], expected_state.recurrent[layer]
        )
        torch.testing.assert_close(
            actual_state.convolution[layer], expected_state.convolution[layer]
        )

def test_torch_compile_opt_in_wraps_runtime_transactions(
    tiny_model, monkeypatch
) -> None:
    compiled = []
    invoked = []

    def fake_compile(function, **options):
        compiled.append((function.__name__, options))

        def wrapped(*args, **kwargs):
            invoked.append(function.__name__)
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setenv("HYDRASERVE_TORCH_COMPILE_BACKEND", "eager")
    monkeypatch.setenv("HYDRASERVE_TORCH_COMPILE_FULLGRAPH", "1")
    monkeypatch.setenv("HYDRASERVE_TORCH_COMPILE_DYNAMIC", "1")
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
        use_torch_compile=True,
    )

    logits, state = runtime.forward(torch.tensor([[3, 7, 11]]))

    assert logits.shape[:2] == (1, 3)
    assert state.sequence_length == 3
    assert invoked == ["_forward_transaction"]
    assert [name for name, _ in compiled] == [
        "_forward_transaction",
        "_decode_batch_transaction",
    ]
    assert all(
        options == {"backend": "eager", "fullgraph": True, "dynamic": True}
        for _, options in compiled
    )
    assert runtime._use_cuda_graphs() is False


def test_torch_compile_eager_backend_matches_uncompiled_forward(
    tiny_model, monkeypatch
) -> None:
    monkeypatch.setenv("HYDRASERVE_TORCH_COMPILE_BACKEND", "eager")
    compiled = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
        use_torch_compile=True,
    )
    eager = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
        use_torch_compile=False,
    )
    token_ids = torch.tensor([[3, 7, 11]])

    actual, actual_state = compiled.forward(token_ids)
    expected, expected_state = eager.forward(token_ids)

    torch.testing.assert_close(actual, expected)
    assert actual_state.sequence_length == expected_state.sequence_length
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            actual_state.recurrent[layer], expected_state.recurrent[layer]
        )
        torch.testing.assert_close(
            actual_state.convolution[layer], expected_state.convolution[layer]
        )

def test_runtime_preserves_native_logits_dtype(tiny_model, monkeypatch) -> None:
    weights = make_weights(tiny_model, dtype=torch.bfloat16)
    token_ids = torch.tensor([[3, 7]])

    monkeypatch.delenv("HYDRASERVE_FP32_LOGITS", raising=False)
    native_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    native_logits, _ = native_runtime.forward(token_ids)

    monkeypatch.setenv("HYDRASERVE_FP32_LOGITS", "1")
    fp32_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    fp32_logits, _ = fp32_runtime.forward(token_ids)

    assert native_logits.dtype == torch.bfloat16
    assert fp32_logits.dtype == torch.float32
    torch.testing.assert_close(native_logits.float(), fp32_logits)


def test_decode_positions_reuse_staging_buffers(tiny_model) -> None:
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    states = [RuntimeState(sequence_length=3), RuntimeState(sequence_length=7)]

    first = runtime._decode_positions(states)
    states[0].sequence_length = 4
    second = runtime._decode_positions(states)

    assert first.data_ptr() == second.data_ptr()
    assert second.tolist() == [[4], [7]]


def test_cuda_graph_capture_policy_defaults_and_validation(
    tiny_model, monkeypatch
) -> None:
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    monkeypatch.delenv("HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER", raising=False)
    monkeypatch.delenv("HYDRASERVE_CUDA_GRAPH_WARMUP_STEPS", raising=False)
    assert runtime._cuda_graph_capture_after() == 16
    assert runtime._cuda_graph_warmup_steps() == 1

    monkeypatch.setenv("HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER", "3")
    monkeypatch.setenv("HYDRASERVE_CUDA_GRAPH_WARMUP_STEPS", "2")
    assert runtime._cuda_graph_capture_after() == 3
    assert runtime._cuda_graph_warmup_steps() == 2

    monkeypatch.setenv("HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER", "0")
    with pytest.raises(ValueError, match="positive integer"):
        runtime._cuda_graph_capture_after()


def test_cuda_graph_capture_is_deferred_and_reuses_bucket_metadata(
    tiny_model, monkeypatch
) -> None:
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    monkeypatch.setenv("HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER", "3")
    table = torch.tensor([[2, -1]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)

    class Cache:
        def batch_metadata(self, request_ids, *, bucket_width=False):
            assert tuple(request_ids) == (7,)
            assert bucket_width
            return table, lengths

    calls = {"capture": 0, "transaction": 0}

    def capture(*args, **kwargs):
        calls["capture"] += 1
        return None

    def transaction(*args, **kwargs):
        calls["transaction"] += 1
        override = kwargs["paged_metadata_override"]
        assert override[0] is table
        assert override[1] is lengths
        return "eager"

    monkeypatch.setattr(runtime, "_capture_decode_graph", capture)
    monkeypatch.setattr(runtime, "_decode_batch_transaction", transaction)
    input_ids = torch.tensor([[3]])
    states = [RuntimeState(sequence_length=4)]

    assert runtime._decode_batch_graph(
        input_ids, states, Cache(), (7,), object()
    ) == "eager"
    assert runtime._decode_batch_graph(
        input_ids, states, Cache(), (7,), object()
    ) == "eager"
    assert calls == {"capture": 0, "transaction": 2}

    assert runtime._decode_batch_graph(
        input_ids, states, Cache(), (7,), object()
    ) == "eager"
    assert calls == {"capture": 1, "transaction": 3}


def test_fused_projections_reduce_linear_calls(tiny_model, monkeypatch) -> None:
    runtime = QwenTextRuntime(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    original_linear = runtime._linear
    calls = []

    def counted_linear(hidden, weight):
        calls.append(weight)
        return original_linear(hidden, weight)

    monkeypatch.setattr(runtime, "_linear", counted_linear)
    hidden = torch.randn(2, 3, tiny_model.hidden_size)
    full_layer = tiny_model.full_attention_layer_indices[0]
    runtime._full_attention_projections(
        hidden, runtime._layer_weights[full_layer].full_attention
    )
    assert len(calls) == 1

    calls.clear()
    runtime._linear_attention_projections(
        hidden, runtime._layer_weights[0].linear_attention
    )
    assert len(calls) == 2

    calls.clear()
    runtime._mlp(0, hidden)
    assert len(calls) == 2


def test_quantized_projection_weights_can_be_fused_without_requantizing() -> None:
    fp8_parts = tuple(
        BlockScaledFP8Weight(
            torch.full((128, 256), value, dtype=torch.float8_e4m3fn),
            torch.full((1, 2), value, dtype=torch.bfloat16),
            (128, 256),
        )
        for value in (1.0, 2.0)
    )
    fused_fp8 = QwenTextRuntime._concatenate_linear_weights(fp8_parts)
    assert isinstance(fused_fp8, BlockScaledFP8Weight)
    assert fused_fp8.shape == (256, 256)
    split_fp8 = QwenTextRuntime._split_linear_weight(fused_fp8, (128, 128))
    for expected, actual in zip(fp8_parts, split_fp8, strict=True):
        torch.testing.assert_close(actual.data, expected.data)
        torch.testing.assert_close(actual.scale_inv, expected.scale_inv)

    int4_parts = tuple(
        PackedInt4Weight(
            torch.full((8, 16), value, dtype=torch.int32),
            torch.full((8, 1), float(value), dtype=torch.bfloat16),
            torch.full((1, 1), value, dtype=torch.int32),
            (8, 128),
        )
        for value in (1, 2)
    )
    fused_int4 = QwenTextRuntime._concatenate_linear_weights(int4_parts)
    assert isinstance(fused_int4, PackedInt4Weight)
    assert fused_int4.shape == (16, 128)
    split_int4 = QwenTextRuntime._split_linear_weight(fused_int4, (8, 8))
    for expected, actual in zip(int4_parts, split_int4, strict=True):
        torch.testing.assert_close(actual.packed, expected.packed)
        torch.testing.assert_close(actual.scale, expected.scale)
        torch.testing.assert_close(actual.zero_point, expected.zero_point)


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


def test_checkpoint_loader_pairs_fp8_weights_with_inverse_scales(
    tiny_model, monkeypatch, tmp_path
) -> None:
    import hydraserve.model.runtime as runtime_module

    raw = make_weights(tiny_model)
    name = f"{layer_prefix(0)}.linear_attn.in_proj_qkv.weight"
    source = raw[name]
    scale = source.abs().max().clamp_min(1e-6) / 448.0
    raw[name] = (source / scale).to(torch.float8_e4m3fn)
    raw[f"{name}_scale_inv"] = scale.reshape(1, 1)

    class FakeLoader:
        def __init__(self, _):
            pass

        def keys(self, prefix=None):
            return tuple(
                sorted(key for key in raw if prefix is None or key.startswith(prefix))
            )

        def __contains__(self, key):
            return key in raw

        def tensor(self, key, *, device="cpu", dtype=None):
            value = raw[key]
            if dtype is not None:
                value = value.to(dtype=dtype)
            return value.to(device=device)

        def tensor_shape(self, key):
            return tuple(raw[key].shape)

    monkeypatch.setattr(runtime_module, "load_model_config", lambda _: tiny_model)
    monkeypatch.setattr(runtime_module, "ShardedSafeTensorLoader", FakeLoader)
    runtime = QwenTextRuntime.from_checkpoint(
        tmp_path,
        device="cpu",
        dtype=torch.float32,
        use_triton=False,
        use_flash_attention=False,
    )

    assert isinstance(runtime.weights[name], BlockScaledFP8Weight)
    assert runtime.weights[name].data.dtype == torch.float8_e4m3fn
    logits, state = runtime.forward(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, tiny_model.vocab_size)
    assert state.sequence_length == 3


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
def test_cpu_embedding_accepts_cpu_token_ids_without_gpu_round_trip(tiny_model) -> None:
    weights = make_weights(tiny_model)
    gpu_weights = {
        name: tensor.to(
            device="cuda",
            dtype=torch.float32
            if name.endswith((".A_log", ".dt_bias"))
            else torch.bfloat16,
        )
        for name, tensor in weights.items()
    }
    embedding_name = "model.language_model.embed_tokens.weight"
    gpu_weights["lm_head.weight"] = gpu_weights[embedding_name].clone()
    baseline = QwenTextRuntime(
        tiny_model,
        gpu_weights,
        use_triton=True,
        use_flash_attention=False,
        device="cuda",
    )
    mixed_weights = dict(gpu_weights)
    mixed_weights[embedding_name] = weights[embedding_name].to(torch.bfloat16)
    mixed = QwenTextRuntime(
        tiny_model,
        mixed_weights,
        use_triton=True,
        use_flash_attention=False,
        device="cuda",
    )
    token_ids = torch.tensor([[1, 9, 4, 3]], device="cpu")

    expected, _ = baseline.forward(token_ids.cuda())
    actual, _ = mixed.forward(token_ids)

    assert mixed.input_device.type == "cpu"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


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
    state_pool = GpuLinearStatePool(
        2, tiny_model, device="cuda", workspace_capacity=2
    )
    batch_states = [
        state_pool.install(request_id, state)
        for request_id, state in zip((10, 11), batch_states, strict=True)
    ]
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pooled_decode_does_not_commit_when_output_projection_fails(
    tiny_model, monkeypatch
) -> None:
    weights = make_weights(tiny_model, device="cuda", dtype=torch.bfloat16)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=True, use_flash_attention=False
    )
    cache = PagedKVCache(
        tiny_model,
        KVBlockManager(16, block_size=4),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache.allocate(7, 3, reserve_tokens=4)
    _, standalone = runtime.forward(
        torch.tensor([[1, 2, 3]], device="cuda"),
        paged_cache=cache,
        request_id=7,
    )
    pool = GpuLinearStatePool(1, tiny_model, device="cuda", workspace_capacity=1)
    state = pool.install(7, standalone)
    recurrent_before = {
        layer: value.clone() for layer, value in state.recurrent.items()
    }
    convolution_before = {
        layer: value.clone() for layer, value in state.convolution.items()
    }
    cache.reserve_append(7)
    bad_output = torch.empty(
        tiny_model.vocab_size, tiny_model.hidden_size + 1,
        device="cuda", dtype=torch.bfloat16,
    )
    monkeypatch.setattr(runtime, "_output_weight", lambda: bad_output)

    with pytest.raises(RuntimeError):
        runtime.decode_batch(
            torch.tensor([[4]], device="cuda"), [state], cache, (7,)
        )

    assert state.sequence_length == 3
    for layer in tiny_model.linear_layer_indices:
        torch.testing.assert_close(state.recurrent[layer], recurrent_before[layer])
        torch.testing.assert_close(
            state.convolution[layer], convolution_before[layer]
        )
