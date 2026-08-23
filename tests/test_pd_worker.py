from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import HostPrefixCache, KVBlockManager, PagedKVCache
from hydraserve.engine import (
    CentralScheduler,
    DecodeWorker,
    PrefillWorker,
    RequestState,
)
from hydraserve.model.runtime import QwenTextRuntime
from hydraserve.transfer import InMemoryTransferBackend, TransferMode, TransferPipeline
from hydraserve.engine.pd_worker import adaptive_transfer_chunk_size
from tests.test_runtime import make_weights


def test_partial_pd_workers_recompute_kv_and_restore_gdn(tiny_model) -> None:
    weights = make_weights(tiny_model)
    prefill_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    decode_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    pipeline = TransferPipeline(
        InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    )
    request = CentralScheduler().submit([1, 3, 5, 7], max_new_tokens=3)
    prefill = PrefillWorker(prefill_runtime, pipeline).process(request)
    assert prefill.state_token_count == len(request.token_ids) - 1
    cache = PagedKVCache(
        tiny_model,
        KVBlockManager(16, block_size=4),
        device="cpu",
        dtype=torch.float32,
    )
    prepared = DecodeWorker(decode_runtime, pipeline, cache).receive_and_prepare(
        request, chunk_size=2
    )
    assert request.state is RequestState.READY
    assert request.generated_token_ids == [prefill.first_token_id]
    assert prepared.state.sequence_length == len(request.token_ids)
    for layer_index in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            prepared.state.recurrent[layer_index], prefill.state.recurrent[layer_index]
        )
        torch.testing.assert_close(
            prepared.state.convolution[layer_index],
            prefill.state.convolution[layer_index].float(),
        )
    table, lengths = cache.batch_metadata((request.request_id,))
    assert lengths.tolist() == [len(request.token_ids)]
    expected_kv_tokens = len(request.token_ids) + request.max_new_tokens - 1
    assert table.shape == (
        1,
        cache.block_manager.blocks_required(expected_kv_tokens),
    )


def test_adaptive_transfer_chunk_is_page_aligned_and_bounded(tiny_model) -> None:
    selected = adaptive_transfer_chunk_size(
        tiny_model, 4096, 16, target_bytes=4096
    )
    assert selected % 16 == 0
    assert 16 <= selected <= 4096
    assert adaptive_transfer_chunk_size(tiny_model, 32, 16) == 32
    assert adaptive_transfer_chunk_size(
        tiny_model,
        4096,
        16,
        target_bytes=4096,
        transfer_mode=TransferMode.INT8_TRANSFER,
    ) >= selected


def test_quantized_pd_workers_install_kv_without_recompute(tiny_model) -> None:
    weights = make_weights(tiny_model)
    prefill_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    decode_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    pipeline = TransferPipeline(
        InMemoryTransferBackend(TransferMode.QUANTIZED_TRANSFER, bandwidth_gbps=14)
    )
    prefill_cache = PagedKVCache(
        tiny_model,
        KVBlockManager(16, block_size=4),
        device="cpu",
        dtype=torch.float32,
    )
    decode_cache = PagedKVCache(
        tiny_model,
        KVBlockManager(16, block_size=4),
        device="cpu",
        dtype=torch.float32,
    )
    request = CentralScheduler().submit([2, 4, 6, 8], max_new_tokens=3)
    result = PrefillWorker(
        prefill_runtime, pipeline, prefill_cache
    ).process(request)
    prepared = DecodeWorker(
        decode_runtime, pipeline, decode_cache
    ).receive_and_prepare(request)
    assert prepared.first_token_id == result.first_token_id
    layer = tiny_model.full_attention_layer_indices[0]
    source_key, source_value = prefill_cache.read(request.request_id, layer)
    target_key, target_value = decode_cache.read(request.request_id, layer)
    torch.testing.assert_close(target_key, source_key, atol=0.15, rtol=0.15)
    torch.testing.assert_close(target_value, source_value, atol=0.15, rtol=0.15)


def test_chunked_prefill_streams_and_installs_kv(tiny_model) -> None:
    weights = make_weights(tiny_model)
    prefill_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    decode_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    pipeline = TransferPipeline(
        InMemoryTransferBackend(TransferMode.QUANTIZED_TRANSFER)
    )
    prefill_cache = PagedKVCache(
        tiny_model, KVBlockManager(16, 2), device="cpu", dtype=torch.float32
    )
    decode_cache = PagedKVCache(
        tiny_model, KVBlockManager(16, 2), device="cpu", dtype=torch.float32
    )
    request = CentralScheduler().submit([2, 4, 6, 8, 10], max_new_tokens=2)
    result = PrefillWorker(prefill_runtime, pipeline, prefill_cache).process(
        request, chunk_size=2, streamed_transfer=True
    )
    prepared = DecodeWorker(decode_runtime, pipeline, decode_cache).receive_and_prepare(
        request, streamed_transfer=True
    )
    assert prepared.first_token_id == result.first_token_id
    layer = tiny_model.full_attention_layer_indices[0]
    source_key, _ = prefill_cache.read(request.request_id, layer)
    target_key, _ = decode_cache.read(request.request_id, layer)
    torch.testing.assert_close(target_key, source_key, atol=0.15, rtol=0.15)


def test_prefill_yields_at_each_runtime_chunk_boundary(tiny_model) -> None:
    weights = make_weights(tiny_model)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    pipeline = TransferPipeline(
        InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    )
    request = CentralScheduler().submit([2, 4, 6, 8, 10], max_new_tokens=2)
    yields = []

    result = PrefillWorker(runtime, pipeline).process(
        request,
        chunk_size=2,
        chunk_yield_callback=lambda: yields.append(len(yields)) or 1,
    )

    # n-1 prefill produces two chunks, then the final prompt token is replayed.
    assert len(yields) == 3
    assert result.chunk_preemptions == 3


def test_hicache_restores_repeated_prefix_without_second_kv_transfer(tiny_model) -> None:
    weights = make_weights(tiny_model)
    prefill_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    decode_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    pipeline = TransferPipeline(InMemoryTransferBackend(TransferMode.FULL_TRANSFER))
    prefill_cache = PagedKVCache(
        tiny_model, KVBlockManager(32, 2), device="cpu", dtype=torch.float32
    )
    decode_cache = PagedKVCache(
        tiny_model, KVBlockManager(32, 2), device="cpu", dtype=torch.float32
    )
    host_cache = HostPrefixCache(1 << 20)
    decode_worker = DecodeWorker(
        decode_runtime, pipeline, decode_cache, host_cache=host_cache
    )
    tokens = [1, 3, 5, 7]

    scheduler = CentralScheduler()
    first = scheduler.submit(tokens, max_new_tokens=2)
    PrefillWorker(prefill_runtime, pipeline, prefill_cache).process(first)
    decode_worker.receive_and_prepare(first)
    assert host_cache.contains(tiny_model.name, tokens)
    decode_cache.free(first.request_id)

    second = scheduler.submit(tokens, max_new_tokens=2)
    PrefillWorker(prefill_runtime, pipeline, prefill_cache).process(
        second, reuse_host_kv=True
    )
    prepared = decode_worker.receive_and_prepare(second)
    assert prepared.first_token_id is not None
    assert host_cache.stats().hits == 1


def test_hicache_radix_hit_streams_only_uncached_suffix(tiny_model) -> None:
    weights = make_weights(tiny_model)
    prefill_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    decode_runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    backend = InMemoryTransferBackend(TransferMode.FULL_TRANSFER)
    pipeline = TransferPipeline(backend)
    prefill_cache = PagedKVCache(
        tiny_model, KVBlockManager(32, 2), device="cpu", dtype=torch.float32
    )
    decode_cache = PagedKVCache(
        tiny_model, KVBlockManager(32, 2), device="cpu", dtype=torch.float32
    )
    host_cache = HostPrefixCache(1 << 20, block_size=2)
    prefill_worker = PrefillWorker(prefill_runtime, pipeline, prefill_cache)
    decode_worker = DecodeWorker(
        decode_runtime, pipeline, decode_cache, host_cache=host_cache
    )
    scheduler = CentralScheduler()

    first = scheduler.submit([1, 3, 5, 7], max_new_tokens=2)
    prefill_worker.process(first, chunk_size=2, streamed_transfer=True)
    decode_worker.receive_and_prepare(first, streamed_transfer=True)
    decode_cache.free(first.request_id)

    second = scheduler.submit([1, 3, 5, 7, 9, 11], max_new_tokens=2)
    assert host_cache.longest_prefix_tokens(tiny_model.name, second.token_ids) == 4
    prefill_worker.process(
        second,
        chunk_size=2,
        streamed_transfer=True,
        host_prefix_tokens=4,
    )
    chunk_keys = {
        key for (_, key) in backend._messages if f"request:{second.request_id}:chunks:" in key
    }
    assert f"request:{second.request_id}:chunks:4:5" in chunk_keys
    assert f"request:{second.request_id}:chunks:5:6" in chunk_keys
    assert not any(":0:" in key or ":2:" in key for key in chunk_keys)
    decode_worker.receive_and_prepare(
        second,
        streamed_transfer=True,
        host_prefix_tokens=4,
    )
    assert host_cache.contains(tiny_model.name, second.token_ids)


def test_n_minus_one_replay_drift_is_observed_but_prefill_token_is_authoritative(
    tiny_model,
) -> None:
    weights = make_weights(tiny_model)
    runtime = QwenTextRuntime(
        tiny_model, weights, use_triton=False, use_flash_attention=False
    )
    backend = InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    pipeline = TransferPipeline(backend)
    request = CentralScheduler().submit([1, 3, 5, 7], max_new_tokens=2)
    result = PrefillWorker(runtime, pipeline).process(request)
    envelope = backend._messages[(1, f"request:{request.request_id}:bundle")]
    authoritative = (result.first_token_id + 1) % tiny_model.vocab_size
    envelope["descriptor"]["first_token_id"] = authoritative
    cache = PagedKVCache(
        tiny_model,
        KVBlockManager(16, block_size=4),
        device="cpu",
        dtype=torch.float32,
    )
    prepared = DecodeWorker(runtime, pipeline, cache).receive_and_prepare(request)
    assert not prepared.replay_consistent
    assert prepared.first_token_id == authoritative
    assert request.generated_token_ids == [authoritative]
