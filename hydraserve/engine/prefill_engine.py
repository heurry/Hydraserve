from __future__ import annotations

from dataclasses import dataclass

from hydraserve.engine.chunked_prefill import ChunkedPrefillScheduler, PrefillChunk
from hydraserve.engine.scheduler import Request, RequestState
from hydraserve.transfer.descriptor import StateTransferDescriptor
from hydraserve.transfer.pipeline import HybridStateBundle, TransferPipeline


@dataclass(frozen=True, slots=True)
class PrefillOutput:
    descriptor: StateTransferDescriptor
    chunks: tuple[PrefillChunk, ...]


class PrefillEngine:
    """Coordinates chunking and extraction; numerical execution is adapter-owned."""

    def __init__(
        self,
        pipeline: TransferPipeline,
        chunk_scheduler: ChunkedPrefillScheduler | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.chunk_scheduler = chunk_scheduler or ChunkedPrefillScheduler()

    def transfer_prefilled_state(
        self,
        request: Request,
        state: HybridStateBundle,
        model,
        *,
        first_token_id: int | None,
        n_minus_one: bool = True,
    ) -> PrefillOutput:
        request.transition(RequestState.PREFILL_RUNNING)
        chunks = self.chunk_scheduler.split(len(request.token_ids), n_minus_one=n_minus_one)
        request.transition(RequestState.TRANSFER_PENDING)
        descriptor = self.pipeline.send(
            request.request_id,
            model,
            len(request.token_ids),
            state,
            first_token_id=first_token_id,
        )
        return PrefillOutput(descriptor, chunks)
