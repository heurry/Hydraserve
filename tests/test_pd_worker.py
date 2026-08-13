from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache
from hydraserve.engine import (
    CentralScheduler,
    DecodeWorker,
    PrefillWorker,
    RequestState,
)
from hydraserve.model.runtime import QwenTextRuntime
from hydraserve.transfer import InMemoryTransferBackend, TransferMode, TransferPipeline
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
    prepared = DecodeWorker(decode_runtime, pipeline, cache).receive_and_prepare(request)
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
