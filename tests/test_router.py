from hydraserve.router import AdaptiveRouter, Route


def test_router_thresholds() -> None:
    router = AdaptiveRouter()
    assert router.route(1024, 0.0) is Route.COLLOCATED
    assert router.route(8192, 0.2) is Route.PD_DISAGGREGATED
    assert router.route(8192, 0.9) is Route.COLLOCATED
    assert router.route(32768, 1.0, decode_has_slot=False) is Route.PD_DISAGGREGATED
