from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Event, Lock, RLock, Thread
from queue import Queue

import pytest

from hydraserve.engine import (
    AdmissionDecision,
    BackendCapacity,
    MultiWorkerGenerationBackend,
    PartialDecodeError,
    PDClusterConfig,
    ServingRequest,
    WorkerUnavailableError,
    WorkerStateLostError,
)
from hydraserve.router import (
    AdaptiveRouter,
    DecodeWorkerRegistry,
    DecodeWorkerSnapshot,
    RouterConfig,
)


class FakeMultiWorkerBackend(MultiWorkerGenerationBackend):
    def __init__(self, *, prefix_affinity=None):
        self.config = SimpleNamespace(
            prefill_devices=("cuda:0",),
            decode_devices=("cuda:1", "cuda:2"),
            block_size=4,
            cache_tokens_per_worker=40,
            max_state_slots_per_worker=4,
            prefix_cache_blocks=0,
            kv_headroom_blocks=0,
            pd_schedule="load-aware",
        )
        self.router = AdaptiveRouter(
            RouterConfig(
                short_prompt_tokens=4, long_prompt_tokens=8, force_pd_tokens=16
            )
        )
        self.prefix_affinity = prefix_affinity
        self.registry = DecodeWorkerRegistry(
            tuple(
                DecodeWorkerSnapshot(
                    index,
                    f"cuda:{index + 1}",
                    BackendCapacity(10, 10, 4, 4),
                )
                for index in range(2)
            )
        )
        self._reserved_blocks = [dict(), dict()]
        self._route_decisions = {}
        self._lost_requests = set()
        self._state_lock = RLock()
        self._prefill_healthy = [True]
        self._prefill_pending = [0]
        self._collocated_count = 0
        self._pd_count = 0
        self._pd_failures = 0
        self._closed = False
        self._prefill_locks = [Lock()]
        self._prefill_processes = [SimpleNamespace(is_alive=lambda: True)]
        self._prefill_commands = [Queue()]
        self._prefill_responses = [Queue()]
        self._prefill_recovering = [False]
        self._prefill_recovery_threads = [None]
        self._prefill_recovery_attempts = [0]
        self._prefill_recovery_successes = [0]
        self._prefill_recovery_failures = [0]
        self._prefill_round_robin = 0
        self._prefill_serve_round_robin = 0
        self._decode_round_robin = 0
        self._prefill_bound = {}
        self._recovery_stop = Event()
        self._recovering_workers = set()
        self._recovery_threads = {}
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._recovery_failures = 0
        self.max_worker_restarts = 3
        self.worker_restart_backoff_s = 0
        self.rpc_calls = []
        self.rpc_commands = []

    def _reserve_on(self, worker_id, request):
        self.rpc_calls.append(("reserve", worker_id, request.request_id))
        self._reserved_blocks[worker_id][request.request_id] = self._required_blocks(
            request
        )
        return AdmissionDecision.accept()

    def _collocated_prefill(self, worker_id, request):
        self.rpc_calls.append(("collocated", worker_id, request.request_id))
        return request.request_id + 100

    def _decode_rpc(
        self, worker_id, command, expected_op, request_id=None, *, dispatched=None
    ):
        if dispatched is not None:
            dispatched.set()
        self.rpc_commands.append((worker_id, dict(command)))
        self.rpc_calls.append(
            (expected_op, worker_id, tuple(command.get("request_ids", ())))
        )
        if expected_op == "decode":
            ids = tuple(command["request_ids"])
            return {"op": "decode", "request_ids": ids, "token_ids": ids}
        return {"op": expected_op, "request_id": request_id}

    def _prefill_serving_rpc(self, index, command, expected_op, request_id=None):
        # W4 prefill-worker collocated serving is not exercised by these unit
        # tests; always decline so collocated requests fall through to decode
        # workers (the pre-W4 path under test).
        self.rpc_calls.append(("prefill_serve", index, command.get("op")))
        if expected_op == "admission":
            return {"op": "admission", "admitted": False, "retryable": False}
        return {"op": expected_op, "request_id": request_id}


def test_cluster_config_rejects_duplicate_or_overlapping_devices() -> None:
    with pytest.raises(ValueError, match="unique"):
        PDClusterConfig("model", ("cuda:1", "cuda:1"))
    with pytest.raises(ValueError, match="distinct"):
        PDClusterConfig("model", ("cuda:0",))


def test_cluster_config_propagates_decode_workspace_capacity() -> None:
    config = PDClusterConfig(
        "model",
        ("cuda:1",),
        max_state_slots_per_worker=12,
        max_decode_batch_size_per_worker=5,
    )
    worker = config.worker_config(0)
    assert worker.max_state_slots == 12
    assert worker.max_decode_batch_size == 5


def test_cluster_config_validates_conditional_and_prefill_short_policies() -> None:
    config = PDClusterConfig(
        "model",
        ("cuda:1", "cuda:2", "cuda:3"),
        conditional_pd_tokens=8192,
        prefill_short_policy="never",
        prefill_preempt_max_ops=4,
    )
    assert config.conditional_pd_tokens == 8192
    assert config.prefill_config(0).prefill_preempt_max_ops == 4
    with pytest.raises(ValueError, match="prefill_short_policy"):
        PDClusterConfig("model", ("cuda:1",), prefill_short_policy="always")
    assert config.effective_hybrid_prefill_reserve_tokens == 32_768
    assert (
        PDClusterConfig(
            "model",
            ("cuda:1",),
            hybrid_prefill_reserve_tokens=4096,
        ).effective_hybrid_prefill_reserve_tokens
        == 4096
    )
    with pytest.raises(ValueError, match="hybrid prefill reserve"):
        PDClusterConfig("model", ("cuda:1",), hybrid_prefill_reserve_tokens=-2)


def test_work_conserving_short_admission_balances_hybrid_with_decode_pool() -> None:
    class HybridServingBackend(FakeMultiWorkerBackend):
        def _prefill_serving_rpc(self, index, command, expected_op, request_id=None):
            self.rpc_calls.append(("prefill_serve", index, command.get("op")))
            return {
                "op": expected_op,
                "request_id": request_id,
                "admitted": True,
            }

    backend = HybridServingBackend()
    backend.config.conditional_pd_tokens = 8
    backend.config.prefill_short_policy = "work-conserving"
    backend.config.hybrid_prefill_reserve_tokens = 0
    requests = [ServingRequest(100 + index, (1, 2, 3, 4), 2) for index in range(6)]

    assert all(backend.admit(request).admitted for request in requests)
    pools = [request.worker_pool for request in requests]
    assert pools.count("prefill") == 2
    assert pools.count("decode") == 4


def test_long_prefill_pending_temporarily_removes_hybrid_from_short_pool() -> None:
    class HybridServingBackend(FakeMultiWorkerBackend):
        def _prefill_serving_rpc(self, index, command, expected_op, request_id=None):
            self.rpc_calls.append(("prefill_serve", index, command.get("op")))
            return {
                "op": expected_op,
                "request_id": request_id,
                "admitted": True,
            }

    backend = HybridServingBackend()
    backend.config.conditional_pd_tokens = 8
    backend.config.prefill_short_policy = "work-conserving"
    backend.config.hybrid_prefill_reserve_tokens = 0

    assert backend.capacity() == BackendCapacity(30, 30, 12, 12)
    assert backend._bind_long_prefill(900) == 0
    assert backend.hybrid_role_states == ("prefill_pending",)
    assert backend.capacity() == BackendCapacity(20, 20, 8, 8)
    short = ServingRequest(901, (1, 2, 3, 4), 2)
    assert backend.admit(short).admitted
    assert short.worker_pool == "decode"
    assert not any(call[0] == "prefill_serve" for call in backend.rpc_calls)

    backend._release_long_prefill_binding(900)
    assert backend.hybrid_role_states == ("decode",)
    assert backend.capacity() == BackendCapacity(30, 30, 12, 12)


def test_long_prefill_role_returns_to_decode_after_rpc() -> None:
    class RoleBackend(FakeMultiWorkerBackend):
        def _prefill_rpc_call(self, index, command, *, long_operation):
            assert long_operation
            assert self.hybrid_role_states[index] == "prefill_active"
            return {
                "op": "prefill",
                "request_id": command["request_id"],
                "worker_index": command["worker_index"],
                "token_id": 7,
            }

    backend = RoleBackend()
    backend._bind_long_prefill(902)
    result = backend._prefill_rpc(
        {"op": "prefill", "request_id": 902, "worker_index": 0}, 902
    )
    assert result["token_id"] == 7
    assert backend.hybrid_role_states == ("decode",)


def test_conditional_route_keeps_short_on_decode_and_sends_long_to_pd() -> None:
    backend = FakeMultiWorkerBackend()
    backend.config.conditional_pd_tokens = 8
    backend.config.prefill_short_policy = "never"

    short = ServingRequest(40, tuple(range(4)), 2)
    long = ServingRequest(41, tuple(range(8)), 2)
    assert backend.admit(short).admitted
    assert backend.admit(long).admitted
    assert short.route == "collocated"
    assert short.route_reason == "conditional_short_collocated"
    assert long.route == "pd_disaggregated"
    assert long.route_reason == "conditional_long_pd"
    assert not any(call[0] == "prefill_serve" for call in backend.rpc_calls)


def test_conditional_routes_use_independent_prefill_executor_groups() -> None:
    backend = FakeMultiWorkerBackend()
    assert backend.prefill_parallelism == 3
    assert backend.prefill_executor_limits == {"prefill": 1, "decode": 2}

    short = ServingRequest(42, (1,), 1)
    short.route = "collocated"
    long = ServingRequest(43, tuple(range(8)), 1)
    long.route = "pd_disaggregated"
    assert backend.prefill_executor_group(short) == "decode"
    assert backend.prefill_executor_group(long) == "prefill"


def test_prefill_rpc_multiplexes_out_of_order_short_response() -> None:
    backend = FakeMultiWorkerBackend()
    backend.operation_timeout = 2.0
    results = {}

    long_thread = Thread(
        target=lambda: results.setdefault(
            "long",
            backend._prefill_rpc_call(
                0, {"op": "prefill", "request_id": 1}, long_operation=True
            ),
        )
    )
    long_thread.start()
    long_command = backend._prefill_commands[0].get(timeout=1)

    short_thread = Thread(
        target=lambda: results.setdefault(
            "short",
            backend._prefill_rpc_call(
                0,
                {"op": "decode", "request_ids": (2,)},
                long_operation=False,
            ),
        )
    )
    short_thread.start()
    short_command = backend._prefill_commands[0].get(timeout=1)

    backend._prefill_responses[0].put(
        {
            "rpc_id": short_command["rpc_id"],
            "op": "decode",
            "request_ids": (2,),
            "token_ids": (7,),
        }
    )
    short_thread.join(1)
    assert not short_thread.is_alive()
    assert long_thread.is_alive()

    backend._prefill_responses[0].put(
        {
            "rpc_id": long_command["rpc_id"],
            "op": "prefill",
            "request_id": 1,
            "token_id": 9,
        }
    )
    long_thread.join(1)
    assert not long_thread.is_alive()
    assert results["short"]["token_ids"] == (7,)
    assert results["long"]["token_id"] == 9


def test_pd_prefill_arms_decode_receiver_before_starting_producer() -> None:
    class ReceiverFirstBackend(FakeMultiWorkerBackend):
        def __init__(self):
            super().__init__()
            self.receiver_armed = Event()
            self.producer_started = Event()
            self.operation_timeout = 2.0
            self._pd_executor = ThreadPoolExecutor(max_workers=1)

        def _decode_rpc(
            self, worker_id, command, expected_op, request_id=None, *, dispatched=None
        ):
            if expected_op != "prepare":
                return super()._decode_rpc(
                    worker_id,
                    command,
                    expected_op,
                    request_id,
                    dispatched=dispatched,
                )
            assert not self.producer_started.is_set()
            self.receiver_armed.set()
            if dispatched is not None:
                dispatched.set()
            return {
                "op": "prepare",
                "request_id": request_id,
                "token_id": 9,
                "replay_consistent": True,
            }

        def _prefill_rpc(self, command, request_id):
            self.producer_started.set()
            assert self.receiver_armed.is_set()
            return {
                "op": "prefill",
                "request_id": request_id,
                "worker_index": command["worker_index"],
                "token_id": 9,
            }

    backend = ReceiverFirstBackend()
    try:
        request = ServingRequest(90, tuple(range(20)), 2)
        assert backend.prefill(request) == 9
        assert request.route == "pd_disaggregated"
    finally:
        backend._pd_executor.shutdown(wait=True)


def test_multi_worker_admission_uses_prefix_affinity_and_binds_route() -> None:
    backend = FakeMultiWorkerBackend(
        prefix_affinity=lambda request, worker_id: (
            len(request.token_ids) if worker_id == 1 else 0
        )
    )
    request = ServingRequest(5, tuple(range(3)), 2)
    assert backend.prefill(request) == 105
    assert backend.worker_for(5) == 1
    assert request.route == "collocated"
    assert ("reserve", 1, 5) in backend.rpc_calls


def test_multi_worker_decode_groups_by_worker_and_preserves_input_order() -> None:
    backend = FakeMultiWorkerBackend()
    requests = tuple(ServingRequest(index, (index + 1,), 2) for index in range(4))
    for request, worker_id in zip(requests, (0, 1, 0, 1), strict=True):
        backend.registry.bind(request.request_id, worker_id)
    assert backend.decode(requests) == (0, 1, 2, 3)
    decode_calls = [call for call in backend.rpc_calls if call[0] == "decode"]
    assert set(decode_calls) == {
        ("decode", 0, (0, 2)),
        ("decode", 1, (1, 3)),
    }


def test_multi_worker_reports_physical_decode_batch_widths() -> None:
    backend = FakeMultiWorkerBackend()
    requests = tuple(ServingRequest(index, (index + 1,), 2) for index in range(3))
    backend.registry.bind(requests[0].request_id, 0)
    backend.registry.bind(requests[1].request_id, 0)
    backend._prefill_bound[requests[2].request_id] = 0

    assert backend.decode_batch_sizes(requests) == {0: 2, 1: 2, 2: 1}


def test_multi_worker_decode_isolates_failed_worker_group() -> None:
    class OneWorkerFails(FakeMultiWorkerBackend):
        def _decode_rpc(
            self, worker_id, command, expected_op, request_id=None, *, dispatched=None
        ):
            if dispatched is not None:
                dispatched.set()
            if expected_op == "decode" and worker_id == 1:
                raise RuntimeError("worker 1 failed")
            return super()._decode_rpc(worker_id, command, expected_op, request_id)

    backend = OneWorkerFails()
    requests = tuple(ServingRequest(index, (index + 1,), 2) for index in range(4))
    for request, worker_id in zip(requests, (0, 1, 0, 1), strict=True):
        backend.registry.bind(request.request_id, worker_id)
    with pytest.raises(PartialDecodeError) as raised:
        backend.decode(requests)
    assert raised.value.token_ids == {0: 0, 2: 2}
    assert set(raised.value.errors) == {1, 3}
    assert all(
        "worker 1 failed" in str(error) for error in raised.value.errors.values()
    )


def test_multi_worker_admission_defers_when_cluster_is_full() -> None:
    backend = FakeMultiWorkerBackend()
    for worker in backend.registry.snapshots():
        backend.registry.update_capacity(worker.worker_id, BackendCapacity(10, 0, 4, 0))
    decision = backend.admit(ServingRequest(99, (1,), 1))
    assert not decision.admitted and decision.retryable


def test_multi_worker_preemption_rebinds_and_sends_exact_recovery_replay() -> None:
    backend = FakeMultiWorkerBackend()
    request = ServingRequest(9, (1, 2, 3), 6, generated_token_ids=[10, 11, 12])
    assert backend.admit(request).admitted
    original_worker = backend.worker_for(request.request_id)

    backend.preempt(request.request_id)
    assert request.request_id not in backend._reserved_blocks[original_worker]
    decision = backend.recover(request)

    assert decision.admitted
    worker_id = backend.worker_for(request.request_id)
    recovery = [
        command
        for recorded_worker, command in backend.rpc_commands
        if recorded_worker == worker_id and command.get("op") == "recover"
    ]
    assert len(recovery) == 1
    assert recovery[0]["token_ids"] == (1, 2, 3)
    assert recovery[0]["generated_token_ids"] == (10, 11, 12)
    assert recovery[0]["replay_token_ids"] == (1, 2, 3, 10, 11)
    backend.release(request.request_id)


def test_prefill_bound_short_recovery_stays_on_prefill_worker() -> None:
    class PrefillServingBackend(FakeMultiWorkerBackend):
        def _prefill_serving_rpc(self, index, command, expected_op, request_id=None):
            self.rpc_commands.append((f"prefill-{index}", dict(command)))
            if expected_op == "admission":
                return {
                    "op": "admission",
                    "request_id": request_id,
                    "admitted": True,
                }
            return {"op": expected_op, "request_id": request_id}

    backend = PrefillServingBackend()
    request = ServingRequest(12, (1, 2, 3), 5, generated_token_ids=[7, 8])
    assert backend.admit(request).admitted
    assert request.request_id in backend._prefill_bound
    assert backend.admit(request).admitted
    admission_commands = [
        command
        for worker, command in backend.rpc_commands
        if str(worker).startswith("prefill-") and command.get("op") == "reserve"
    ]
    assert len(admission_commands) == 1

    backend.preempt(request.request_id)
    assert backend.recover(request).admitted
    recovery = [
        command
        for worker, command in backend.rpc_commands
        if str(worker).startswith("prefill-") and command.get("op") == "recover"
    ]
    assert len(recovery) == 1
    assert recovery[0]["replay_token_ids"] == (1, 2, 3, 7)


def test_dead_decode_worker_is_removed_and_recovery_is_scheduled() -> None:
    class DeadProcess:
        def is_alive(self):
            return False

    backend = FakeMultiWorkerBackend()
    backend._decode_processes = [DeadProcess(), DeadProcess()]
    backend._decode_locks = [Lock(), Lock()]
    backend._decode_commands = [Queue(), Queue()]
    backend._decode_responses = [Queue(), Queue()]
    scheduled = []
    backend._schedule_decode_recovery = scheduled.append

    with pytest.raises(WorkerUnavailableError, match="not running"):
        MultiWorkerGenerationBackend._decode_rpc(
            backend, 0, {"op": "decode", "request_ids": (1,)}, "decode"
        )
    assert not backend.registry.snapshots()[0].healthy
    assert scheduled == [0]


def test_failed_worker_invalidates_all_bindings_and_marks_state_recoverable() -> None:
    backend = FakeMultiWorkerBackend()
    requests = (ServingRequest(20, (1,), 3), ServingRequest(21, (2,), 3))
    for request in requests:
        backend.registry.bind(request.request_id, 0)
        backend._reserved_blocks[0][request.request_id] = 1
    invalidated = backend._invalidate_worker(0)

    assert invalidated == (20, 21)
    assert backend.registry.snapshots()[0].active_requests == 0
    assert not backend.registry.snapshots()[0].healthy
    assert backend._reserved_blocks[0] == {}
    for request in requests:
        with pytest.raises(KeyError):
            backend.registry.worker_for(request.request_id)
        error = WorkerStateLostError("lost")
        assert backend.is_recoverable_decode_error(request.request_id, error)


def test_decode_reports_previously_invalidated_request_as_partial_failure() -> None:
    backend = FakeMultiWorkerBackend()
    healthy = ServingRequest(30, (1,), 3)
    lost = ServingRequest(31, (2,), 3)
    backend.registry.bind(healthy.request_id, 1)
    backend._lost_requests.add(lost.request_id)

    with pytest.raises(PartialDecodeError) as raised:
        backend.decode((healthy, lost))

    assert raised.value.token_ids == {healthy.request_id: healthy.request_id}
    assert isinstance(raised.value.errors[lost.request_id], WorkerStateLostError)


def test_worker_recovery_retries_with_backoff_and_restores_health() -> None:
    class RecoveringBackend(FakeMultiWorkerBackend):
        def __init__(self):
            super().__init__()
            self.restart_calls = 0

        def _restart_decode_worker_once(self, worker_id):
            self.restart_calls += 1
            if self.restart_calls == 1:
                raise RuntimeError("startup failed")
            self.registry.set_health(worker_id, True)

    backend = RecoveringBackend()
    backend.registry.set_health(0, False)
    backend._recovering_workers.add(0)
    backend._recover_decode_worker(0)
    stats = backend.recovery_stats()
    assert backend.restart_calls == 2
    assert stats.attempts == 2
    assert stats.successes == 1
    assert stats.failures == 1
    assert stats.healthy_workers == 2
    assert stats.recovering_workers == ()


def test_dead_prefill_worker_fails_fast_and_schedules_recovery() -> None:
    class DeadProcess:
        def is_alive(self):
            return False

    backend = FakeMultiWorkerBackend()
    backend._prefill_processes = [DeadProcess()]
    backend._prefill_commands = [Queue()]
    backend._prefill_responses = [Queue()]
    backend.operation_timeout = 30
    scheduled = []
    backend._schedule_prefill_recovery = lambda index: scheduled.append(True)

    with pytest.raises(WorkerUnavailableError, match="prefill worker 0 is not running"):
        backend._prefill_rpc({"op": "prefill"}, 1)

    assert not backend.routing_stats().prefill_healthy
    assert backend.routing_stats().pd_failures == 1
    assert scheduled == [True]


def test_prefill_recovery_retries_and_restores_pd_routing_health() -> None:
    class RecoveringPrefill(FakeMultiWorkerBackend):
        def __init__(self):
            super().__init__()
            self.restart_calls = 0

        def _restart_prefill_worker_once(self, index):
            self.restart_calls += 1
            if self.restart_calls == 1:
                raise RuntimeError("prefill startup failed")

    backend = RecoveringPrefill()
    backend._prefill_healthy = [False]
    backend._prefill_recovering = [True]
    backend._recover_prefill_worker(0)
    stats = backend.prefill_recovery_stats()

    assert backend.restart_calls == 2
    assert stats.healthy
    assert stats.attempts == 2
    assert stats.successes == 1
    assert stats.failures == 1
    assert not stats.recovering


def test_unhealthy_prefill_worker_forces_new_request_to_collocated_route() -> None:
    backend = FakeMultiWorkerBackend()
    backend._prefill_healthy = [False]
    request = ServingRequest(88, tuple(range(20)), 2)

    assert backend.admit(request).admitted
    assert request.route == "collocated"
    assert request.route_reason == "prefill_unavailable"
    backend.release(request.request_id)


def test_pick_prefill_worker_prefers_least_loaded() -> None:
    backend = FakeMultiWorkerBackend()
    backend._prefill_processes = [None, None]
    backend._prefill_healthy = [True, True]
    backend._prefill_round_robin = 0
    backend._prefill_pending = [2, 0]
    assert backend._pick_prefill_worker() == 1
    backend._prefill_pending = [0, 3]
    assert backend._pick_prefill_worker() == 0
    backend._prefill_healthy = [False, True]
    backend._prefill_pending = [0, 5]
    assert backend._pick_prefill_worker() == 1


def test_prefill_dispatch_claims_prevent_concurrent_worker_herding() -> None:
    backend = FakeMultiWorkerBackend()
    backend._prefill_processes = [None, None]
    backend._prefill_healthy = [True, True]
    backend._prefill_pending = [0, 0]
    backend._prefill_dispatch_claims = [0, 0]

    first = backend._claim_prefill_worker()
    second = backend._claim_prefill_worker()
    assert (first, second) == (0, 1)
    backend._release_prefill_dispatch_claim(first)
    backend._release_prefill_dispatch_claim(second)
    assert backend._prefill_dispatch_claims == [0, 0]


def test_prefill_short_admission_claims_idle_worker_atomically() -> None:
    backend = FakeMultiWorkerBackend()
    backend._prefill_dispatch_claims = [0]

    first = backend._pick_serve_prefill_worker()
    second = backend._pick_serve_prefill_worker()
    assert first == 0
    assert second is None
    backend._release_prefill_dispatch_claim(first)
    third = backend._pick_serve_prefill_worker()
    assert third == 0
    backend._release_prefill_dispatch_claim(third)
