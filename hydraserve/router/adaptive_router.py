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
    PREFILL_SATURATED = "prefill_saturated"
    NO_DECODE_SLOT = "no_decode_slot"
    PREFILL_UNAVAILABLE = "prefill_unavailable"
    CONDITIONAL_SHORT_COLLOCATED = "conditional_short_collocated"
    CONDITIONAL_LONG_PD = "conditional_long_pd"
    COST_MODEL_COLLOCATED = "cost_model_collocated"
    COST_MODEL_PD = "cost_model_pd"
    COST_MODEL_CONSERVATIVE = "cost_model_conservative"
    COST_MODEL_HOLD_COLLOCATED = "cost_model_hold_collocated"
    COST_MODEL_HOLD_PD = "cost_model_hold_pd"
    COST_MODEL_DRIFT = "cost_model_drift"


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
    prefill_queue_ahead_ms: float = 0.0
    prefill_load: float = 0.0


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
        prefill_queue_ahead_ms: float = 0.0,
        prefill_load: float = 0.0,
    ) -> Route:
        return self.decide(
            prompt_tokens,
            decode_load,
            decode_has_slot,
            prefill_queue_ahead_ms,
            prefill_load,
        ).route

    def decide(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
        prefill_queue_ahead_ms: float = 0.0,
        prefill_load: float = 0.0,
    ) -> RouteDecision:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        if prefill_queue_ahead_ms < 0:
            raise ValueError("prefill queue cost cannot be negative")
        if not 0 <= prefill_load <= 1:
            raise ValueError("prefill_load must be in [0, 1]")
        if prompt_tokens >= self.config.force_pd_tokens:
            route, reason = Route.PD_DISAGGREGATED, RouteReason.FORCED_LONG_PROMPT
        elif prompt_tokens < self.config.short_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.SHORT_PROMPT
        elif not decode_has_slot:
            route, reason = Route.COLLOCATED, RouteReason.NO_DECODE_SLOT
        elif (
            prompt_tokens >= self.config.long_prompt_tokens
            and decode_load < self.config.decode_load_limit
            and prefill_load < self.config.decode_load_limit
        ):
            route, reason = Route.PD_DISAGGREGATED, RouteReason.LONG_PROMPT_PD
        elif prompt_tokens >= self.config.long_prompt_tokens:
            route, reason = (
                Route.COLLOCATED,
                RouteReason.PREFILL_SATURATED
                if prefill_load >= self.config.decode_load_limit
                else RouteReason.DECODE_SATURATED,
            )
        else:
            route, reason = Route.COLLOCATED, RouteReason.MEDIUM_PROMPT
        return RouteDecision(
            route=route,
            reason=reason,
            prompt_tokens=prompt_tokens,
            decode_load=decode_load,
            decode_has_slot=decode_has_slot,
            prefill_queue_ahead_ms=prefill_queue_ahead_ms,
            prefill_load=prefill_load,
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
    hysteresis_ms: float = 5.0
    hysteresis_ratio: float = 0.02
    drift_ratio_threshold: float = 1.5
    drift_min_observations: int = 5
    fail_closed_on_drift: bool = True
    force_pd_tokens: int = 0
    prefill_load_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_pd_prompt_tokens <= 0:
            raise ValueError("minimum PD prompt length must be positive")
        if self.force_pd_tokens < 0:
            raise ValueError("force-PD prompt threshold cannot be negative")
        if self.prefill_load_scale < 0:
            raise ValueError("prefill load scale cannot be negative")
        if self.minimum_savings_ms < 0:
            raise ValueError("minimum route savings cannot be negative")
        if not 0 <= self.minimum_savings_ratio < 1:
            raise ValueError("minimum savings ratio must be in [0, 1)")
        if self.pd_uncertainty_multiplier < 1:
            raise ValueError("PD uncertainty multiplier cannot be below one")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")
        if self.hysteresis_ms < 0 or not 0 <= self.hysteresis_ratio < 1:
            raise ValueError("invalid route hysteresis")
        if self.drift_ratio_threshold <= 1 or self.drift_min_observations <= 0:
            raise ValueError("invalid route-profile drift policy")

    @classmethod
    def partial_transfer_default(cls) -> "CostRouterConfig":
        """Conservative 4B/3090 SHM prior, calibrated by the local B-vs-D run.

        PARTIAL_TRANSFER repeats prompt execution on the decode worker. Its prior
        must not assume that a length threshold alone makes PD beneficial.
        """

        return cls(
            collocated=LatencyCurve(
                28.075501691917207,
                0.3720540365424588,
                0.00043369855004635356,
                0.1571279209918884,
            ),
            pd_disaggregated=LatencyCurve(
                216.26650429730614,
                1.478002962857347,
                0.00036159295511324613,
                0.0,
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
    collocated_drifted_buckets: tuple[int, ...] = ()
    pd_drifted_buckets: tuple[int, ...] = ()


@dataclass(slots=True)
class _CostObservation:
    correction: float = 1.0
    count: int = 0


class CostAwareRouter:
    """Risk-aware route selection with online, prompt-bucketed calibration."""

    def __init__(self, config: CostRouterConfig | None = None) -> None:
        self.config = config or CostRouterConfig.partial_transfer_default()
        self._observations: dict[tuple[Route, int], _CostObservation] = {}
        self._last_routes: dict[int, Route] = {}
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
        prefill_queue_ahead_ms: float = 0.0,
        prefill_load: float = 0.0,
    ) -> Route:
        return self.decide(
            prompt_tokens,
            decode_load,
            decode_has_slot,
            prefill_queue_ahead_ms,
            prefill_load,
        ).route

    def decide(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
        prefill_queue_ahead_ms: float = 0.0,
        prefill_load: float = 0.0,
    ) -> RouteDecision:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        if prefill_queue_ahead_ms < 0:
            raise ValueError("prefill queue cost cannot be negative")
        if not 0 <= prefill_load <= 1:
            raise ValueError("prefill_load must be in [0, 1]")
        collocated_service = self._predict(
            Route.COLLOCATED, prompt_tokens, decode_load
        )
        pd_service = self._predict(Route.PD_DISAGGREGATED, prompt_tokens, decode_load)
        # A busy prefill pool inflates the PD path's effective cost: the
        # request would wait behind in-flight long prefills.
        risk_adjusted_pd_service = (
            pd_service
            * self.config.pd_uncertainty_multiplier
            * (1.0 + self.config.prefill_load_scale * prefill_load)
        )
        savings = collocated_service - risk_adjusted_pd_service
        required_savings = max(
            self.config.minimum_savings_ms,
            collocated_service * self.config.minimum_savings_ratio,
        )
        bucket = self._bucket(prompt_tokens)
        if not decode_has_slot:
            route, reason = Route.COLLOCATED, RouteReason.NO_DECODE_SLOT
        elif (
            self.config.force_pd_tokens
            and prompt_tokens >= self.config.force_pd_tokens
        ):
            route, reason = Route.PD_DISAGGREGATED, RouteReason.FORCED_LONG_PROMPT
        elif prompt_tokens < self.config.minimum_pd_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.COST_MODEL_CONSERVATIVE
        else:
            with self._lock:
                drifted = self._bucket_is_drifted(bucket)
                previous = self._last_routes.get(bucket)
                hysteresis = max(
                    self.config.hysteresis_ms,
                    collocated_service * self.config.hysteresis_ratio,
                )
                if drifted and self.config.fail_closed_on_drift:
                    route, reason = Route.COLLOCATED, RouteReason.COST_MODEL_DRIFT
                elif (
                    previous is Route.COLLOCATED
                    and savings >= required_savings
                    and savings < required_savings + hysteresis
                ):
                    route = Route.COLLOCATED
                    reason = RouteReason.COST_MODEL_HOLD_COLLOCATED
                elif (
                    previous is Route.PD_DISAGGREGATED
                    and savings < required_savings
                    and savings >= required_savings - hysteresis
                ):
                    route = Route.PD_DISAGGREGATED
                    reason = RouteReason.COST_MODEL_HOLD_PD
                elif savings >= required_savings:
                    route, reason = Route.PD_DISAGGREGATED, RouteReason.COST_MODEL_PD
                else:
                    route = Route.COLLOCATED
                    reason = RouteReason.COST_MODEL_COLLOCATED
                self._last_routes[bucket] = route
        return RouteDecision(
            route,
            reason,
            prompt_tokens,
            decode_load,
            decode_has_slot,
            collocated_service + prefill_queue_ahead_ms,
            risk_adjusted_pd_service + prefill_queue_ahead_ms,
            savings,
            self._confidence(prompt_tokens),
            prefill_queue_ahead_ms,
            prefill_load,
        )

    def observe(
        self,
        route: Route | str,
        prompt_tokens: int,
        elapsed_ms: float,
        decode_load: float,
        prefill_queue_ahead_ms: float = 0.0,
    ) -> None:
        route = Route(route)
        if (
            prompt_tokens <= 0
            or elapsed_ms <= 0
            or not 0 <= decode_load <= 1
            or prefill_queue_ahead_ms < 0
        ):
            raise ValueError("invalid route-cost observation")
        curve = self._curve(route)
        baseline = curve.predict(prompt_tokens, decode_load)
        # Backend timing begins when its executor task starts; it is already
        # queue-free. The queue estimate is only for end-to-end TTFT attribution.
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
            collocated_drift = self._drifted_buckets(Route.COLLOCATED)
            pd_drift = self._drifted_buckets(Route.PD_DISAGGREGATED)
            return RouteCostStats(
                collocated[0],
                pd[0],
                collocated[1],
                pd[1],
                collocated_drift,
                pd_drift,
            )

    def reset_online_state(self) -> None:
        """Drop learned corrections/hysteresis without changing the profile."""

        with self._lock:
            self._observations.clear()
            self._last_routes.clear()

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

    def _bucket_is_drifted(self, bucket: int) -> bool:
        return any(self._observation_is_drifted(route, bucket) for route in Route)

    def _drifted_buckets(self, route: Route) -> tuple[int, ...]:
        return tuple(
            sorted(
                bucket
                for observed_route, bucket in self._observations
                if observed_route is route
                and self._observation_is_drifted(route, bucket)
            )
        )

    def _observation_is_drifted(self, route: Route, bucket: int) -> bool:
        observation = self._observations.get((route, bucket))
        if observation is None or observation.count < self.config.drift_min_observations:
            return False
        threshold = self.config.drift_ratio_threshold
        return (
            observation.correction > threshold
            or observation.correction < 1.0 / threshold
        )

    @staticmethod
    def _bucket(prompt_tokens: int) -> int:
        return prompt_tokens.bit_length() - 1
