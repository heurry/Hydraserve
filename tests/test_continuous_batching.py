from hydraserve.cache import KVBlockManager
from hydraserve.engine import CentralScheduler, ContinuousBatchScheduler, RequestState


def test_new_requests_join_between_decode_iterations() -> None:
    blocks = KVBlockManager(64, block_size=4)
    central = CentralScheduler()
    first = central.submit([1, 2, 3], max_new_tokens=2)
    second = central.submit([4, 5], max_new_tokens=2)
    blocks.allocate(first.request_id, len(first.token_ids))
    blocks.allocate(second.request_id, len(second.token_ids))
    scheduler = ContinuousBatchScheduler(
        blocks, max_decode_sequences=4, max_prefill_tokens=4, prefill_chunk_size=4
    )
    scheduler.add(first)
    item = scheduler.next_prefill_batch()
    assert [entry.request_id for entry in item] == [first.request_id]
    scheduler.mark_prefill_complete(first.request_id)
    batch = scheduler.next_decode_batch()
    assert batch.request_ids == (first.request_id,)
    scheduler.commit_decode_tokens(batch, (10,))

    scheduler.add(second)
    scheduler.next_prefill_batch()
    scheduler.mark_prefill_complete(second.request_id)
    batch = scheduler.next_decode_batch()
    assert batch.request_ids == (first.request_id, second.request_id)
    scheduler.commit_decode_tokens(batch, (11, 12))
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.RUNNING


def test_preempt_releases_blocks_and_resume_reallocates() -> None:
    blocks = KVBlockManager(16, block_size=4)
    request = CentralScheduler().submit([1, 2, 3, 4], max_new_tokens=3)
    blocks.allocate(request.request_id, 4)
    scheduler = ContinuousBatchScheduler(blocks)
    scheduler.add(request)
    scheduler.next_prefill_batch()
    scheduler.mark_prefill_complete(request.request_id)
    batch = scheduler.next_decode_batch()
    scheduler.commit_decode_tokens(batch, (9,))
    scheduler.preempt(request.request_id)
    assert request.state is RequestState.PREEMPTED
    assert blocks.num_free_blocks == 16
    scheduler.resume(request.request_id)
    assert request.state is RequestState.READY
    assert blocks.num_free_blocks == 14
