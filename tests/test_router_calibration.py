from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydraserve.router import (
    CalibrationPoint,
    CostAwareRouter,
    CostRouterConfig,
    Route,
    build_router_profile,
    fit_latency_curve,
    load_calibration_points,
)


def _points(fixed: float, linear: float, quadratic: float):
    return tuple(
        CalibrationPoint(tokens, fixed + linear * tokens + quadratic * tokens * tokens)
        for tokens in (32, 256, 2048, 8192)
    )


def test_nonnegative_curve_fit_recovers_synthetic_latency() -> None:
    fitted = fit_latency_curve(_points(12.0, 0.04, 0.0002))
    assert fitted.curve.fixed_ms == pytest.approx(12.0, rel=1e-7)
    assert fitted.curve.linear_ms_per_token == pytest.approx(0.04, rel=1e-7)
    assert fitted.curve.quadratic_ms_per_token2 == pytest.approx(0.0002, rel=1e-7)
    assert fitted.diagnostics.rmse_ms < 1e-8
    assert fitted.diagnostics.unique_prompt_lengths == 4
    assert fitted.diagnostics.loaded_samples == 0


def test_curve_fit_separates_decode_load_externality() -> None:
    baseline = _points(20, 0.1, 0.0001)
    loaded = tuple(
        CalibrationPoint(
            point.prompt_tokens,
            point.ttft_ms * 1.5,
            0.5,
        )
        for point in baseline
    )
    fitted = fit_latency_curve((*baseline, *loaded))
    assert fitted.curve.decode_load_scale == pytest.approx(1.0, rel=1e-7)
    assert fitted.diagnostics.loaded_samples == 4
    assert fitted.diagnostics.rmse_ms < 1e-8


def test_curve_fit_requires_prompt_length_coverage() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        fit_latency_curve((CalibrationPoint(32, 10), CalibrationPoint(64, 20)))


def test_benchmark_loader_skips_failed_results_and_validates_shape(tmp_path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(
        json.dumps(
            {
                "results": [
                    {"prompt_tokens": 32, "ttft_ms": 10, "error": None},
                    {"prompt_tokens": 64, "ttft_ms": 20, "error": "OOM"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_calibration_points((source,)) == (CalibrationPoint(32, 10),)
    source.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no results array"):
        load_calibration_points((source,))


def test_fitted_profile_metadata_is_loadable_by_cost_router(tmp_path) -> None:
    profile = build_router_profile(
        _points(50, 0.5, 0.0004),
        _points(10, 0.1, 0.00005),
        minimum_pd_prompt_tokens=16,
        pd_uncertainty_multiplier=1,
    )
    assert profile["metadata"]["fit"]["collocated"]["samples"] == 4
    CostRouterConfig.from_dict(profile)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert CostAwareRouter.from_json(path).route(2048, 0) is Route.PD_DISAGGREGATED


def test_checked_in_profile_matches_builtin_partial_prior() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/router/rtx3090-4b-shm-partial.json"
    )
    configured = CostAwareRouter.from_json(path).decide(9_000, 0)
    builtin = CostAwareRouter().decide(9_000, 0)
    assert configured.route is builtin.route is Route.COLLOCATED
    assert configured.collocated_cost_ms == pytest.approx(builtin.collocated_cost_ms)
    assert configured.pd_cost_ms == pytest.approx(builtin.pd_cost_ms)
