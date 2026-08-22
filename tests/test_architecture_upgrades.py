from __future__ import annotations

import numpy as np
import pytest

from hydraserve.cache import HostPrefixCache
from hydraserve.engine.dp_graph_sync import pad_dp_batch, synchronize_dp_token_count
from hydraserve.engine.zmq_proxy import ZMQWaveRouter
from hydraserve.transfer import (
    BootstrapClient,
    BootstrapRegistry,
    BootstrapServer,
    NetworkBootstrapClient,
    StateHandlerRegistry,
    StateType,
    compute_head_slice_params,
)


def test_host_prefix_cache_is_bounded_lru() -> None:
    cache = HostPrefixCache(max_bytes=16)
    first = np.arange(4, dtype=np.float32)
    second = first + 10
    assert cache.put("model", (1, 2), first)
    np.testing.assert_array_equal(cache.get("model", (1, 2)), first)
    assert cache.put("model", (3, 4), second)
    assert cache.get("model", (1, 2)) is None
    stats = cache.stats()
    assert stats.entries == 1 and stats.evictions == 1


def test_host_prefix_cache_returns_longest_block_aligned_prefix_without_copy() -> None:
    cache = HostPrefixCache(max_bytes=1 << 20, block_size=2)
    payload = np.arange(2 * 2 * 6 * 2, dtype=np.uint16).reshape(2, 2, 6, 2)
    assert cache.put("model", (1, 2, 3, 4, 5, 6), payload)

    match = cache.match("model", (1, 2, 3, 4, 5, 6, 7, 8))
    assert match.matched_tokens == 6
    assert match.payload is cache.get("model", (1, 2, 3, 4, 5, 6))
    np.testing.assert_array_equal(match.payload, payload)
    assert cache.longest_prefix_tokens("model", (1, 2, 9, 9)) == 2


def test_host_prefix_cache_drops_partial_page_tail() -> None:
    cache = HostPrefixCache(max_bytes=1 << 20, block_size=4)
    payload = np.arange(2 * 2 * 6, dtype=np.uint16).reshape(2, 2, 6, 1, 1)
    assert cache.put("model", (1, 2, 3, 4, 5, 6), payload)
    exact = cache.match("model", (1, 2, 3, 4, 5, 6))
    assert exact.matched_tokens == 6
    match = cache.match("model", (1, 2, 3, 4, 9, 10))
    assert match.matched_tokens == 4
    assert match.payload.shape[2] == 4


def test_host_prefix_cache_matches_shared_path_before_divergent_tail() -> None:
    cache = HostPrefixCache(max_bytes=1 << 20, block_size=2)
    payload = np.arange(2 * 2 * 6, dtype=np.uint16).reshape(2, 2, 6, 1, 1)
    assert cache.put("model", (1, 2, 3, 4, 5, 6), payload)
    match = cache.match("model", (1, 2, 3, 4, 9, 10))
    assert match.matched_tokens == 4
    np.testing.assert_array_equal(match.payload, payload[:, :, :4])
    assert np.shares_memory(match.payload, payload)


def test_host_prefix_cache_pin_prevents_admission_restore_eviction() -> None:
    cache = HostPrefixCache(max_bytes=16, block_size=1)
    first = np.arange(4, dtype=np.float32)
    assert cache.put("model", (1,), first)
    lease = cache.pin("model", (1, 9))
    assert lease.matched_tokens == 1
    assert not cache.put("model", (2,), first + 10)
    cache.unpin(lease)
    assert cache.put("model", (2,), first + 10)


def test_bootstrap_metadata_is_one_shot() -> None:
    client = BootstrapClient(BootstrapRegistry())
    client.publish(7, "kv_chunks", {"ranges": [[0, 4]]})
    assert client.consume(7, "kv_chunks", timeout=0.1) == {"ranges": [[0, 4]]}
    with pytest.raises(TimeoutError):
        client.consume(7, "kv_chunks", timeout=0.001)


def test_network_bootstrap_keeps_metadata_off_data_plane() -> None:
    try:
        server_context = BootstrapServer()
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    with server_context as server:
        sender = NetworkBootstrapClient(server.address)
        receiver = NetworkBootstrapClient(server.address)
        sender.publish(8, "topology", {"src": 0, "dst": 1})
        assert receiver.consume(8, "topology", timeout=0.2) == {
            "src": 0,
            "dst": 1,
        }


def test_tp_head_slices_support_replication() -> None:
    assert compute_head_slice_params(8, 0, 4) == (0, 2)
    assert compute_head_slice_params(8, 3, 4) == (6, 2)
    assert compute_head_slice_params(4, 0, 4, replication_factor=2) == (0, 2)
    assert compute_head_slice_params(4, 1, 4, replication_factor=2) == (0, 2)
    assert compute_head_slice_params(4, 2, 4, replication_factor=2) == (2, 2)


def test_state_type_dispatch_is_extensible() -> None:
    registry = StateHandlerRegistry()
    registry.register(StateType.DSA_KV, lambda value: value + 1)
    assert registry.dispatch(StateType.DSA_KV, 4) == 5
    with pytest.raises(NotImplementedError):
        registry.dispatch(StateType.MLA_KV, 4)


def test_dp_padding_without_distributed_process_group() -> None:
    torch = pytest.importorskip("torch")
    plan = synchronize_dp_token_count(2)
    assert plan.synchronized_tokens == 2 and plan.padding_tokens == 0
    padded, mask = pad_dp_batch(torch.tensor([[1], [2]]), 4, pad_token_id=9)
    assert padded.tolist() == [[1], [2], [9], [9]]
    assert mask.tolist() == [True, True, False, False]


def test_zmq_wave_router_balances_and_synchronizes() -> None:
    router = ZMQWaveRouter((b"a", b"b"))
    router.mark_ready(b"a", 0)
    router.mark_ready(b"b", 0)
    first = router.choose()
    second = router.choose()
    assert {first, second} == {b"a", b"b"}
    assert router.wave == 1
    assert router.choose() is None
    router.mark_ready(b"a", 1)
    router.mark_ready(b"b", 1)
    assert router.choose() in {b"a", b"b"}
