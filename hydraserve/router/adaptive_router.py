from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from threading import RLock


class Route(str, Enum):
    COLLOCATED = "collocated"
    PD_DISAGGREGATED = "pd_disaggregated"


class RouteReason(str, Enum):
    SHORT_PROMPT = "short_prompt"
    MEDIUM_PROMPT = "medium_prompt"
    LONG_PROMPT_PD = "long_prompt_pd"
    FORCED_LONG_PROMPT = "forced_long_prompt"
    DECODE_SATURATED = "decode_saturated"
    NO_DECODE_SLOT = "no_decode_slot"
    PREFILL_UNAVAILABLE = "prefill_unavailable"
    COST_MODEL_COLLOCATED = "cost_model_collocated"
    COST_MODEL_PD = "cost_model_pd"
    COST_MODEL_CONSERVATIVE = "cost_model_conservative"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    reason: RouteReason
    prompt_tokens: int
    decode_load: float
    decode_has_slot: bool
    collocated_cost_ms: float | None = None
    pd_cost_ms: float | None = None
    estimated_savings_ms: float | None = None
    cost_model_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RouterConfig:
    short_prompt_tokens: int = 2_048
    long_prompt_tokens: int = 8_192
    force_pd_tokens: int = 32_768
    decode_load_limit: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.short_prompt_tokens < self.long_prompt_tokens <= self.force_pd_tokens:
            raise ValueError("prompt thresholds must be strictly ordered")
        if not 0 < self.decode_load_limit <= 1:
            raise ValueError("decode_load_limit must be in (0, 1]")


class AdaptiveRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
    ) -> Route:
        return self.decide(prompt_tokens, decode_load, decode_has_slot).route

    def decide(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
    ) -> RouteDecision:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        if prompt_tokens >= self.config.force_pd_tokens:
            route, reason = Route.PD_DISAGGREGATED, RouteReason.FORCED_LONG_PROMPT
        elif prompt_tokens < self.config.short_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.SHORT_PROMPT
        elif not decode_has_slot:
            route, reason = Route.COLLOCATED, RouteReason.NO_DECODE_SLOT
        elif (
            prompt_tokens >= self.config.long_prompt_tokens
            and decode_load < self.config.decode_load_limit
        ):
            route, reason = Route.PD_DISAGGREGATED, RouteReason.LONG_PROMPT_PD
        elif prompt_tokens >= self.config.long_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.DECODE_SATURATED
        else:
            route, reason = Route.COLLOCATED, RouteReason.MEDIUM_PROMPT
        return RouteDecision(
            route=route,
            reason=reason,
            prompt_tokens=prompt_tokens,
            decode_load=decode_load,
            decode_has_slot=decode_has_slot,
        )


@dataclass(frozen=True, slots=True)
class LatencyCurve:
    """Quadratic prompt-cost prior with an explicit decode-load multiplier."""

    fixed_ms: float
    linear_ms_per_token: float
    quadratic_ms_per_token2: float = 0.0
    decode_load_scale: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.fixed_ms,
            self.linear_ms_per_token,
            self.quadratic_ms_per_token2,
            self.decode_load_scale,
        ) < 0:
            raise ValueError("latency curve coefficients cannot be negative")

    def predict(self, prompt_tokens: int, decode_load: float) -> float:
        if prompt_tokens <= 0 or not 0 <= decode_load <= 1:
            raise ValueError("invalid latency-curve input")
        base = (
            self.fixed_ms
            + self.linear_ms_per_token * prompt_tokens
            + self.quadratic_ms_per_token2 * prompt_tokens * prompt_tokens
        )
        return base * (1.0 + self.decode_load_scale * decode_load)


@dataclass(frozen=True, slots=True)
class CostRouterConfig:
    collocated: LatencyCurve
    pd_disaggregated: LatencyCurve
    minimum_pd_prompt_tokens: int = 256
    minimum_savings_ms: float = 2.0
    minimum_savings_ratio: float = 0.05
    pd_uncertainty_multiplier: float = 1.10
    ewma_alpha: float = 0.2

    def __post_init__(self) -> None:
        if self.minimum_pd_prompt_tokens <= 0:
            raise ValueError("minimum PD prompt length must be positive")
        if self.minimum_savings_ms < 0:
            raise ValueError("minimum route savings cannot be negative")
        if not 0 <= self.minimum_savings_ratio < 1:
            raise ValueError("minimum savings ratio must be in [0, 1)")
        if self.pd_uncertainty_multiplier < 1:
            raise ValueError("PD uncertainty multiplier cannot be below one")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")

    @classmethod
    def partial_transfer_default(cls) -> "CostRouterConfig":
        """Conservative 4B/3090 SHM prior, calibrated by the local B-vs-D run.

        PARTIAL_TRANSFER repeats prompt execution on the decode worker. Its prior
        must not assume that a length threshold alone makes PD beneficial.
        """

        return cls(
            collocated=LatencyCurve(
                21.802638304077778,
                0.7440074621754906,
                0.00027073184949249845,
            ),
            pd_disaggregated=LatencyCurve(
                252.91923530140815,
                0.6801430442154653,
                0.00044979155493816183,
            ),
            minimum_pd_prompt_tokens=256,
            minimum_savings_ms=5.0,
            minimum_savings_ratio=0.05,
            pd_uncertainty_multiplier=1.10,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "CostRouterConfig":
        if not isinstance(payload, dict):
            raise ValueError("router profile must be a JSON object")
        expected = {"collocated", "pd_disaggregated"}
        missing = expected - payload.keys()
        if missing:
            raise ValueError(f"router profile is missing {sorted(missing)}")
        try:
            collocated = LatencyCurve(**payload["collocated"])
            pd_disaggregated = LatencyCurve(**payload["pd_disaggregated"])
            options = {
                key: value
                for key, value in payload.items()
                if key not in expected and key != "metadata"
            }
            return cls(collocated, pd_disaggregated, **options)
        except TypeError as exc:
            raise ValueError(f"invalid router profile: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RouteCostStats:
    collocated_observations: int
    pd_observations: int
    collocated_correction: float
    pd_correction: float


@dataclass(slots=True)
class _CostObservation:
    correction: float = 1.0
    count: int = 0


class CostAwareRouter:
    """Risk-aware route selection with online, prompt-bucketed calibration."""

    def __init__(self, config: CostRouterConfig | None = None) -> None:
        self.config = config or CostRouterConfig.partial_transfer_default()
        self._observations: dict[tuple[Route, int], _CostObservation] = {}
        self._lock = RLock()

    @classmethod
    def from_json(cls, path: str | Path) -> "CostAwareRouter":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(CostRouterConfig.from_dict(payload))

    def route(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
    ) -> Route:
        return self.decide(prompt_tokens, decode_load, decode_has_slot).route

    def decide(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
    ) -> RouteDecision:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        collocated = self._predict(Route.COLLOCATED, prompt_tokens, decode_load)
        pd = self._predict(Route.PD_DISAGGREGATED, prompt_tokens, decode_load)
        risk_adjusted_pd = pd * self.config.pd_uncertainty_multiplier
        savings = collocated - risk_adjusted_pd
        required_savings = max(
            self.config.minimum_savings_ms,
            collocated * self.config.minimum_savings_ratio,
        )
        if not decode_has_slot:
            route, reason = Route.COLLOCATED, RouteReason.NO_DECODE_SLOT
        elif prompt_tokens < self.config.minimum_pd_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.COST_MODEL_CONSERVATIVE
        elif savings >= required_savings:
            route, reason = Route.PD_DISAGGREGATED, RouteReason.COST_MODEL_PD
        else:
            route, reason = Route.COLLOCATED, RouteReason.COST_MODEL_COLLOCATED
        return RouteDecision(
            route,
            reason,
            prompt_tokens,
            decode_load,
            decode_has_slot,
            collocated,
            risk_adjusted_pd,
            savings,
            self._confidence(prompt_tokens),
        )

    def observe(
        self,
        route: Route | str,
        prompt_tokens: int,
        elapsed_ms: float,
        decode_load: float,
    ) -> None:
        route = Route(route)
        if prompt_tokens <= 0 or elapsed_ms <= 0 or not 0 <= decode_load <= 1:
            raise ValueError("invalid route-cost observation")
        curve = self._curve(route)
        baseline = curve.predict(prompt_tokens, decode_load)
        ratio = min(4.0, max(0.25, elapsed_ms / max(baseline, 1e-6)))
        key = (route, self._bucket(prompt_tokens))
        with self._lock:
            observation = self._observations.setdefault(key, _CostObservation())
            alpha = 1.0 if observation.count == 0 else self.config.ewma_alpha
            observation.correction = (
                alpha * ratio + (1.0 - alpha) * observation.correction
            )
            observation.count += 1

    def stats(self) -> RouteCostStats:
        with self._lock:
            def summarize(route: Route) -> tuple[int, float]:
                values = [
                    observation
                    for (observed_route, _), observation in self._observations.items()
                    if observed_route is route
                ]
                count = sum(value.count for value in values)
                if count == 0:
                    return 0, 1.0
                correction = sum(
                    value.correction * value.count for value in values
                ) / count
                return count, correction

            collocated = summarize(Route.COLLOCATED)
            pd = summarize(Route.PD_DISAGGREGATED)
            return RouteCostStats(collocated[0], pd[0], collocated[1], pd[1])

    def _predict(self, route: Route, prompt_tokens: int, decode_load: float) -> float:
        baseline = self._curve(route).predict(prompt_tokens, decode_load)
        key = (route, self._bucket(prompt_tokens))
        with self._lock:
            observation = self._observations.get(key)
            correction = 1.0 if observation is None else observation.correction
        return baseline * correction

    def _confidence(self, prompt_tokens: int) -> float:
        bucket = self._bucket(prompt_tokens)
        with self._lock:
            counts = [
                self._observations.get((route, bucket), _CostObservation()).count
                for route in Route
            ]
        return min(1.0, min(counts) / 5.0)

    def _curve(self, route: Route) -> LatencyCurve:
        if route is Route.COLLOCATED:
            return self.config.collocated
        return self.config.pd_disaggregated

    @staticmethod
    def _bucket(prompt_tokens: int) -> int:
        return prompt_tokens.bit_length() - 1
