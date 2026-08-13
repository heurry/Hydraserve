import pytest

from hydraserve.engine import FairDecodeScheduler, ServingRequest


def _request(request_id: int, priority: int) -> ServingRequest:
    return ServingRequest(request_id, (1,), 32, priority=priority)


def test_higher_priority_gets_more_service_but_aging_prevents_starvation() -> None:
    scheduler = FairDecodeScheduler()
    low = _request(1, 0)
    high = _request(2, 3)
    selected = [scheduler.select((low, high), 1)[0].request_id for _ in range(24)]
    assert selected.count(high.request_id) > selected.count(low.request_id)
    assert low.request_id in selected


def test_scheduler_prunes_finished_accounts_and_rejects_bad_priority() -> None:
    scheduler = FairDecodeScheduler()
    request = _request(1, 0)
    scheduler.select((request,), 1)
    scheduler.forget(request.request_id)
    bad = _request(2, scheduler.config.max_priority + 1)
    with pytest.raises(ValueError, match="priority"):
        scheduler.select((bad,), 1)
