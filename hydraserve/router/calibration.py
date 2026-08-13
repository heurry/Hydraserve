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

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.ttft_ms <= 0 or not isfinite(self.ttft_ms):
            raise ValueError("calibration points require positive finite values")


@dataclass(frozen=True, slots=True)
class CurveFitDiagnostics:
    samples: int
    unique_prompt_lengths: int
    minimum_prompt_tokens: int
    maximum_prompt_tokens: int
    rmse_ms: float


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
                    int(result["prompt_tokens"]), float(result["ttft_ms"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid benchmark result in {path}: {exc}") from exc
            points.append(point)
    if not points:
        raise ValueError("calibration inputs contain no successful requests")
    return tuple(points)


def fit_latency_curve(points: Iterable[CalibrationPoint]) -> FittedLatencyCurve:
    values = tuple(points)
    lengths = np.asarray([point.prompt_tokens for point in values], dtype=np.float64)
    latency = np.asarray([point.ttft_ms for point in values], dtype=np.float64)
    unique_lengths = np.unique(lengths)
    if unique_lengths.size < 3:
        raise ValueError("latency fitting requires at least three distinct prompt lengths")

    scale = float(lengths.max())
    normalized = lengths / scale
    design = np.column_stack(
        (np.ones_like(normalized), normalized, normalized * normalized)
    )
    coefficients, predictions = _nonnegative_least_squares(design, latency)
    fixed, normalized_linear, normalized_quadratic = coefficients
    curve = LatencyCurve(
        fixed_ms=float(fixed),
        linear_ms_per_token=float(normalized_linear / scale),
        quadratic_ms_per_token2=float(normalized_quadratic / (scale * scale)),
    )
    rmse = float(np.sqrt(np.mean(np.square(predictions - latency))))
    return FittedLatencyCurve(
        curve,
        CurveFitDiagnostics(
            samples=len(values),
            unique_prompt_lengths=int(unique_lengths.size),
            minimum_prompt_tokens=int(lengths.min()),
            maximum_prompt_tokens=int(lengths.max()),
            rmse_ms=rmse,
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
        "metadata": {
            "fit": {
                "collocated": asdict(collocated.diagnostics),
                "pd_disaggregated": asdict(pd.diagnostics),
            },
            "latency_metric": "ttft_ms",
            "recommended_input": "concurrency-1 warmed benchmark outputs",
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
