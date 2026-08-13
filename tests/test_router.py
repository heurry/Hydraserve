import json

import pytest

from hydraserve.router import (
    AdaptiveRouter,
    CostAwareRouter,
    CostRouterConfig,
    LatencyCurve,
    Route,
    RouteReason,
)


def test_router_thresholds() -> None:
    router = AdaptiveRouter()
    assert router.route(1024, 0.0) is Route.COLLOCATED
    assert router.route(8192, 0.2) is Route.PD_DISAGGREGATED
    assert router.route(8192, 0.9) is Route.COLLOCATED
    assert router.route(32768, 1.0, decode_has_slot=False) is Route.PD_DISAGGREGATED


def test_router_exposes_stable_decision_reason() -> None:
    router = AdaptiveRouter()
    short = router.decide(100, 0.1, True)
    assert short.route is Route.COLLOCATED
    assert short.reason is RouteReason.SHORT_PROMPT
    saturated = router.decide(9000, 0.9, True)
    assert saturated.route is Route.COLLOCATED
    assert saturated.reason is RouteReason.DECODE_SATURATED
    long = router.decide(9000, 0.2, True)
    assert long.route is Route.PD_DISAGGREGATED
    assert long.reason is RouteReason.LONG_PROMPT_PD


def test_partial_transfer_cost_prior_rejects_false_8k_crossover() -> None:
    router = CostAwareRouter()
    decision = router.decide(9_000, 0.2, True)
    assert decision.route is Route.COLLOCATED
    assert decision.reason is RouteReason.COST_MODEL_COLLOCATED
    assert decision.collocated_cost_ms < decision.pd_cost_ms
    assert decision.estimated_savings_ms < 0


def test_cost_router_selects_pd_only_after_risk_adjusted_margin() -> None:
    router = CostAwareRouter(
        CostRouterConfig(
            collocated=LatencyCurve(100, 1.0),
            pd_disaggregated=LatencyCurve(20, 0.4),
            minimum_pd_prompt_tokens=16,
            minimum_savings_ms=10,
            minimum_savings_ratio=0.1,
            pd_uncertainty_multiplier=1.0,
        )
    )
    decision = router.decide(100, 0.5)
    assert decision.route is Route.PD_DISAGGREGATED
    assert decision.reason is RouteReason.COST_MODEL_PD
    assert decision.estimated_savings_ms == pytest.approx(140)


def test_online_observation_can_correct_an_optimistic_pd_prior() -> None:
    router = CostAwareRouter(
        CostRouterConfig(
            collocated=LatencyCurve(100, 0),
            pd_disaggregated=LatencyCurve(50, 0),
            minimum_pd_prompt_tokens=1,
            minimum_savings_ms=1,
            minimum_savings_ratio=0,
            pd_uncertainty_multiplier=1,
            ewma_alpha=1,
        )
    )
    assert router.route(1024, 0) is Route.PD_DISAGGREGATED
    router.observe(Route.PD_DISAGGREGATED, 1024, 300, 0)
    corrected = router.decide(1024, 0)
    assert corrected.route is Route.COLLOCATED
    assert corrected.pd_cost_ms == pytest.approx(200)  # correction is clipped at 4x
    assert router.stats().pd_observations == 1


def test_router_profile_loads_and_rejects_incomplete_json(tmp_path) -> None:
    profile = tmp_path / "router.json"
    profile.write_text(
        json.dumps(
            {
                "collocated": {"fixed_ms": 100, "linear_ms_per_token": 1},
                "pd_disaggregated": {
                    "fixed_ms": 10,
                    "linear_ms_per_token": 0.1,
                },
                "minimum_pd_prompt_tokens": 8,
                "pd_uncertainty_multiplier": 1,
            }
        ),
        encoding="utf-8",
    )
    assert CostAwareRouter.from_json(profile).route(32, 0) is Route.PD_DISAGGREGATED
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        CostAwareRouter.from_json(profile)


def test_cost_router_hysteresis_prevents_boundary_flapping() -> None:
    router = CostAwareRouter(
        CostRouterConfig(
            collocated=LatencyCurve(100, 0, decode_load_scale=1),
            pd_disaggregated=LatencyCurve(100, 0),
            minimum_pd_prompt_tokens=1,
            minimum_savings_ms=1,
            minimum_savings_ratio=0,
            pd_uncertainty_multiplier=1,
            hysteresis_ms=10,
            hysteresis_ratio=0,
        )
    )
    assert router.route(1024, 0) is Route.COLLOCATED
    held_collocated = router.decide(1024, 0.05)
    assert held_collocated.route is Route.COLLOCATED
    assert held_collocated.reason is RouteReason.COST_MODEL_HOLD_COLLOCATED
    assert router.route(1024, 0.2) is Route.PD_DISAGGREGATED
    held_pd = router.decide(1024, 0.005)
    assert held_pd.route is Route.PD_DISAGGREGATED
    assert held_pd.reason is RouteReason.COST_MODEL_HOLD_PD


def test_profile_drift_fails_closed_and_is_reported() -> None:
    router = CostAwareRouter(
        CostRouterConfig(
            collocated=LatencyCurve(100, 0),
            pd_disaggregated=LatencyCurve(50, 0),
            minimum_pd_prompt_tokens=1,
            minimum_savings_ms=1,
            minimum_savings_ratio=0,
            pd_uncertainty_multiplier=1,
            ewma_alpha=1,
            drift_ratio_threshold=1.5,
            drift_min_observations=3,
        )
    )
    for _ in range(3):
        router.observe(Route.COLLOCATED, 1024, 200, 0)
    decision = router.decide(1024, 0)
    assert decision.route is Route.COLLOCATED
    assert decision.reason is RouteReason.COST_MODEL_DRIFT
    assert router.stats().collocated_drifted_buckets == (10,)
