from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager
from hydraserve.engine import (
    CentralScheduler,
    ContinuousBatchExecutor,
    ContinuousBatchScheduler,
    RequestState,
)


class FakeRuntime:
    device = torch.device("cpu")

    def decode_batch(self, input_ids, states, paged_cache, request_ids):
        logits = torch.zeros(len(request_ids), 1, 32)
        for row, token in enumerate(input_ids[:, 0]):
            logits[row, 0, int(token) + 1] = 1
        return logits, states


def _ready_request(scheduler, central, blocks, token_ids, max_new_tokens=1):
    request = central.submit(token_ids, max_new_tokens=max_new_tokens)
    blocks.allocate(request.request_id, len(token_ids))
    scheduler.add(request)
    scheduler.next_prefill_batch()
    scheduler.mark_prefill_complete(request.request_id)
    return request


def test_executor_connects_iteration_to_sampler() -> None:
    blocks = KVBlockManager(16, block_size=4)
    central = CentralScheduler()
    scheduler = ContinuousBatchScheduler(blocks)
    first = _ready_request(scheduler, central, blocks, [1, 2])
    second = _ready_request(scheduler, central, blocks, [3, 4])
    executor = ContinuousBatchExecutor(scheduler, FakeRuntime(), paged_cache=object())
    executor.register_state(first.request_id, object())
    executor.register_state(second.request_id, object())
    batch, sampled = executor.step()
    assert batch.request_ids == (first.request_id, second.request_id)
    assert sampled == (3, 5)
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.FINISHED
    assert blocks.num_free_blocks == blocks.num_blocks
    empty_batch, sampled = executor.step()
    assert empty_batch.request_ids == ()
    assert sampled == ()


def test_executor_failure_releases_inflight_requests() -> None:
    class FailingRuntime(FakeRuntime):
        def decode_batch(self, *args, **kwargs):
            raise RuntimeError("kernel failed")

    blocks = KVBlockManager(8, block_size=4)
    central = CentralScheduler()
    scheduler = ContinuousBatchScheduler(blocks)
    request = _ready_request(scheduler, central, blocks, [1, 2])
    executor = ContinuousBatchExecutor(scheduler, FailingRuntime(), object())
    executor.register_state(request.request_id, object())
    with pytest.raises(RuntimeError, match="kernel failed"):
        executor.step()
    assert request.state is RequestState.FAILED
    assert blocks.num_free_blocks == blocks.num_blocks
