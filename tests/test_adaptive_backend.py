from __future__ import annotations

from threading import RLock

from hydraserve.engine import (
    AdaptiveGenerationBackend,
    AdmissionDecision,
    BackendCapacity,
    ServingRequest,
)
from hydraserve.router import AdaptiveRouter, CostAwareRouter, Route, RouteReason


class FakeAdaptiveBackend(AdaptiveGenerationBackend):
    def __init__(self) -> None:
        self.router = AdaptiveRouter()
        self._route_decisions = {}
        self._route_lock = RLock()
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
        self._prefill_healthy = True
        self.snapshot = BackendCapacity(100, 100, 10, 10)
        self.calls = []
        self.timeout_pd = False

    def capacity(self):
        return self.snapshot

    def _prefill_available(self):
        return self._prefill_healthy

    def _reserve_decode(self, request, *, force_rpc=False):
        self.calls.append(("reserve", request.request_id, force_rpc))
        return AdmissionDecision.accept()

    def _prefill_collocated(self, request):
        self.calls.append(("collocated", request.request_id))
        return 11

    def _prefill_pd(self, request):
        self.calls.append(("pd", request.request_id))
        if self.timeout_pd:
            raise TimeoutError("prefill worker timed out")
        return 22

    def release(self, request_id):
        with self._route_lock:
            self._route_decisions.pop(request_id, None)


def test_adaptive_backend_routes_each_admitted_request() -> None:
    backend = FakeAdaptiveBackend()
    short = ServingRequest(1, tuple(range(32)), 4)
    long = ServingRequest(2, tuple(range(9000)), 4)
    assert backend.prefill(short) == 11
    assert backend.prefill(long) == 22
    assert backend.route_for(1).route is Route.COLLOCATED
    assert backend.route_for(2).route is Route.PD_DISAGGREGATED
    assert backend.routing_stats().collocated == 1
    assert backend.routing_stats().pd_disaggregated == 1


def test_route_is_immutable_after_capacity_changes() -> None:
    backend = FakeAdaptiveBackend()
    request = ServingRequest(7, tuple(range(9000)), 4)
    assert backend.admit(request).admitted
    assert backend.route_for(7).route is Route.PD_DISAGGREGATED
    backend.snapshot = BackendCapacity(100, 1, 10, 1)
    assert backend.admit(request).admitted
    assert backend.route_for(7).route is Route.PD_DISAGGREGATED


def test_ambiguous_pd_timeout_quarantines_route_for_later_requests() -> None:
    backend = FakeAdaptiveBackend()
    backend.timeout_pd = True
    request = ServingRequest(9, tuple(range(9000)), 4)
    import pytest

    with pytest.raises(TimeoutError):
        backend.prefill(request)
    stats = backend.routing_stats()
    assert stats.pd_failures == 1
    assert not stats.prefill_healthy
    assert stats.pd_disaggregated == 0
    later = ServingRequest(10, tuple(range(9000)), 4)
    backend.timeout_pd = False
    assert backend.prefill(later) == 11
    assert backend.route_for(10).route is Route.COLLOCATED


def test_cost_route_is_bound_observed_and_exposed_on_request() -> None:
    backend = FakeAdaptiveBackend()
    backend.router = CostAwareRouter()
    request = ServingRequest(12, tuple(range(9_000)), 4)
    assert backend.prefill(request) == 11
    assert request.route == Route.COLLOCATED.value
    assert request.route_reason == RouteReason.COST_MODEL_COLLOCATED.value
    assert request.worker_id == 0
    assert request.route_collocated_cost_ms is not None
    assert request.route_pd_cost_ms > request.route_collocated_cost_ms
    assert request.route_estimated_savings_ms < 0
    stats = backend.routing_cost_stats()
    assert stats.collocated_observations == 1
    assert stats.pd_observations == 0
