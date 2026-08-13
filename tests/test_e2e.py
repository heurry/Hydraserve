from __future__ import annotations

import numpy as np

from hydraserve.cache import KVBlockManager, LinearState, LinearStatePool
from hydraserve.engine import CentralScheduler, DecodeEngine, PrefillEngine, RequestState
from hydraserve.router import Route
from hydraserve.transfer import (
    HybridStateBundle,
    InMemoryTransferBackend,
    TransferMode,
    TransferPipeline,
)


def test_partial_transfer_vertical_slice(tiny_model) -> None:
    scheduler = CentralScheduler()
    request = scheduler.submit(list(range(9000)), max_new_tokens=10)
    assert request.route is Route.PD_DISAGGREGATED

    backend = InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    pipeline = TransferPipeline(backend)
    recurrent = LinearState(
        np.ones(tiny_model.ssm_state_shape, dtype=np.float32),
        np.ones(tiny_model.conv_state_shape, dtype=np.float32),
    )
    output = PrefillEngine(pipeline).transfer_prefilled_state(
        request,
        HybridStateBundle(recurrent),
        tiny_model,
        first_token_id=101,
        n_minus_one=True,
    )
    assert sum(chunk.size for chunk in output.chunks) == len(request.token_ids) - 1
    assert request.state is RequestState.TRANSFER_PENDING

    recomputed: list[tuple[int, ...]] = []

    def recompute(token_ids: tuple[int, ...]) -> np.ndarray:
        recomputed.append(token_ids)
        return np.zeros((len(token_ids), tiny_model.num_kv_heads, tiny_model.head_dim), dtype=np.float32)

    decoder = DecodeEngine(
        pipeline,
        KVBlockManager(num_blocks=1024),
        LinearStatePool(2, tiny_model.ssm_state_shape, tiny_model.conv_state_shape),
        kv_recompute=recompute,
    )
    installed = decoder.receive_and_install(request)
    assert recomputed == [request.token_ids]
    assert installed.seeded_token_id == 101
    assert request.generated_token_ids == [101]
    assert request.state is RequestState.READY
    decoder.start(request)
    decoder.finish(request)
    assert request.state is RequestState.FINISHED
