"""Offline fitting for hardware/model/transport-specific route profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from math import isfinite
from pathlib import Path
from typing import Iterable

import numpy as np

from hydraserve.router.adaptive_router import LatencyCurve


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    prompt_tokens: int
    ttft_ms: float
    decode_load: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.prompt_tokens <= 0
            or self.ttft_ms <= 0
            or not isfinite(self.ttft_ms)
            or not 0 <= self.decode_load <= 1
        ):
            raise ValueError("calibration points require positive finite values")


@dataclass(frozen=True, slots=True)
class CurveFitDiagnostics:
    samples: int
    unique_prompt_lengths: int
    minimum_prompt_tokens: int
    maximum_prompt_tokens: int
    rmse_ms: float
    loaded_samples: int


@dataclass(frozen=True, slots=True)
class FittedLatencyCurve:
    curve: LatencyCurve
    diagnostics: CurveFitDiagnostics


def load_calibration_points(paths: Iterable[str | Path]) -> tuple[CalibrationPoint, ...]:
    points: list[CalibrationPoint] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ValueError(f"benchmark output has no results array: {path}")
        for result in results:
            if not isinstance(result, dict) or result.get("error") is not None:
                continue
            try:
                point = CalibrationPoint(
                    int(result["prompt_tokens"]),
                    float(result["ttft_ms"]),
                    float(result.get("route_decode_load") or 0.0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid benchmark result in {path}: {exc}") from exc
            points.append(point)
    if not points:
        raise ValueError("calibration inputs contain no successful requests")
    return tuple(points)


def fit_latency_curve(points: Iterable[CalibrationPoint]) -> FittedLatencyCurve:
    values = tuple(points)
    baseline_values = tuple(point for point in values if point.decode_load <= 0.05)
    lengths = np.asarray(
        [point.prompt_tokens for point in baseline_values], dtype=np.float64
    )
    latency = np.asarray([point.ttft_ms for point in baseline_values], dtype=np.float64)
    unique_lengths = np.unique(lengths)
    if unique_lengths.size < 3:
        raise ValueError(
            "latency fitting requires at least three distinct low-load prompt lengths"
        )

    scale = float(lengths.max())
    normalized = lengths / scale
    design = np.column_stack(
        (np.ones_like(normalized), normalized, normalized * normalized)
    )
    coefficients, predictions = _nonnegative_least_squares(design, latency)
    fixed, normalized_linear, normalized_quadratic = coefficients
    base_curve = LatencyCurve(
        fixed_ms=float(fixed),
        linear_ms_per_token=float(normalized_linear / scale),
        quadratic_ms_per_token2=float(normalized_quadratic / (scale * scale)),
    )
    loaded_values = tuple(point for point in values if point.decode_load > 0.05)
    load_scale = _fit_decode_load_scale(base_curve, loaded_values)
    curve = LatencyCurve(
        base_curve.fixed_ms,
        base_curve.linear_ms_per_token,
        base_curve.quadratic_ms_per_token2,
        load_scale,
    )
    all_predictions = np.asarray(
        [curve.predict(point.prompt_tokens, point.decode_load) for point in values]
    )
    all_latency = np.asarray([point.ttft_ms for point in values])
    rmse = float(np.sqrt(np.mean(np.square(all_predictions - all_latency))))
    return FittedLatencyCurve(
        curve,
        CurveFitDiagnostics(
            samples=len(values),
            unique_prompt_lengths=int(unique_lengths.size),
            minimum_prompt_tokens=min(point.prompt_tokens for point in values),
            maximum_prompt_tokens=max(point.prompt_tokens for point in values),
            rmse_ms=rmse,
            loaded_samples=len(loaded_values),
        ),
    )


def build_router_profile(
    collocated_points: Iterable[CalibrationPoint],
    pd_points: Iterable[CalibrationPoint],
    *,
    minimum_pd_prompt_tokens: int = 256,
    minimum_savings_ms: float = 5.0,
    minimum_savings_ratio: float = 0.05,
    pd_uncertainty_multiplier: float = 1.10,
    ewma_alpha: float = 0.2,
    hysteresis_ms: float = 5.0,
    hysteresis_ratio: float = 0.02,
    drift_ratio_threshold: float = 1.5,
    drift_min_observations: int = 5,
    fail_closed_on_drift: bool = True,
) -> dict:
    collocated = fit_latency_curve(collocated_points)
    pd = fit_latency_curve(pd_points)
    return {
        "collocated": asdict(collocated.curve),
        "pd_disaggregated": asdict(pd.curve),
        "minimum_pd_prompt_tokens": minimum_pd_prompt_tokens,
        "minimum_savings_ms": minimum_savings_ms,
        "minimum_savings_ratio": minimum_savings_ratio,
        "pd_uncertainty_multiplier": pd_uncertainty_multiplier,
        "ewma_alpha": ewma_alpha,
        "hysteresis_ms": hysteresis_ms,
        "hysteresis_ratio": hysteresis_ratio,
        "drift_ratio_threshold": drift_ratio_threshold,
        "drift_min_observations": drift_min_observations,
        "fail_closed_on_drift": fail_closed_on_drift,
        "metadata": {
            "fit": {
                "collocated": asdict(collocated.diagnostics),
                "pd_disaggregated": asdict(pd.diagnostics),
            },
            "latency_metric": "ttft_ms",
            "recommended_input": (
                "warmed C1 baselines plus optional loaded traces carrying "
                "route_decode_load"
            ),
        },
    }


def _nonnegative_least_squares(
    design: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the three-variable NNLS problem by enumerating active sets."""

    best_coefficients = None
    best_predictions = None
    best_error = float("inf")
    columns = range(design.shape[1])
    for size in range(1, design.shape[1] + 1):
        for active in combinations(columns, size):
            partial, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
            if np.any(partial < 0):
                continue
            coefficients = np.zeros(design.shape[1], dtype=np.float64)
            coefficients[list(active)] = partial
            predictions = design @ coefficients
            error = float(np.sum(np.square(predictions - target)))
            if error < best_error:
                best_coefficients = coefficients
                best_predictions = predictions
                best_error = error
    if best_coefficients is None or best_predictions is None:
        raise RuntimeError("nonnegative latency fit has no feasible solution")
    return best_coefficients, best_predictions


def _fit_decode_load_scale(
    curve: LatencyCurve, points: tuple[CalibrationPoint, ...]
) -> float:
    if not points:
        return 0.0
    estimates = []
    for point in points:
        baseline = curve.predict(point.prompt_tokens, 0.0)
        estimate = (point.ttft_ms / max(baseline, 1e-6) - 1.0) / point.decode_load
        estimates.append(max(0.0, estimate))
    return min(10.0, float(np.median(np.asarray(estimates, dtype=np.float64))))
