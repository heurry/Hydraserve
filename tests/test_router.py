from hydraserve.router import AdaptiveRouter, Route, RouteReason


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
