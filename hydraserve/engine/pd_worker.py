from __future__ import annotations

from dataclasses import dataclass

from hydraserve.engine.scheduler import Request, RequestState
from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer.pipeline import TransferPipeline
from hydraserve.transfer.runtime_codec import RuntimeStateCodec
from hydraserve.transfer.descriptor import TransferMode


@dataclass(slots=True)
class PrefillResult:
    first_token_id: int
    state: RuntimeState
    state_token_count: int


@dataclass(slots=True)
class DecodePrepared:
    first_token_id: int | None
    state: RuntimeState


class PrefillWorker:
    """GPU worker that runs prompt prefill and publishes transferable GDN state."""

    def __init__(self, runtime, pipeline: TransferPipeline, paged_cache=None) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache

    def process(self, request: Request, *, n_minus_one: bool = True) -> PrefillResult:
        import torch

        request.transition(RequestState.PREFILL_RUNNING)
        use_n_minus_one = n_minus_one and len(request.token_ids) > 1
        split = len(request.token_ids) - 1 if use_n_minus_one else len(request.token_ids)
        if self.paged_cache is not None:
            self.paged_cache.allocate(request.request_id, len(request.token_ids))
        if (
            self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
            and self.paged_cache is None
        ):
            raise RuntimeError("full/quantized transfer requires a prefill Paged KV cache")
        prefix_ids = torch.tensor(
            [request.token_ids[:split]], device=self.runtime.device, dtype=torch.long
        )
        with torch.inference_mode():
            logits, state = self.runtime.forward(
                prefix_ids,
                paged_cache=self.paged_cache,
                request_id=request.request_id if self.paged_cache is not None else None,
            )
        bundle = RuntimeStateCodec.extract(self.runtime.config, state)
        if use_n_minus_one:
            last_id = torch.tensor(
                [[request.token_ids[-1]]], device=self.runtime.device, dtype=torch.long
            )
            with torch.inference_mode():
                logits, state = self.runtime.forward(
                    last_id,
                    state,
                    paged_cache=self.paged_cache,
                    request_id=request.request_id if self.paged_cache is not None else None,
                )
        if self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER:
            bundle.kv_cache = RuntimeStateCodec.extract_kv(
                self.runtime.config, self.paged_cache, request.request_id
            )
        first_token = int(logits[0, -1].argmax())
        request.transition(RequestState.TRANSFER_PENDING)
        self.pipeline.send(
            request.request_id,
            self.runtime.config,
            len(request.token_ids),
            bundle,
            first_token_id=first_token,
            state_token_count=split,
        )
        return PrefillResult(first_token, state, split)


class DecodeWorker:
    """GPU worker that recomputes KV for PARTIAL mode and installs transferred state."""

    def __init__(self, runtime, pipeline: TransferPipeline, paged_cache) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache

    def receive_and_prepare(
        self, request: Request, *, timeout: float | None = None
    ) -> DecodePrepared:
        import torch

        if request.state is not RequestState.TRANSFER_PENDING:
            raise RuntimeError("request is not awaiting a PD transfer")
        descriptor, bundle = self.pipeline.receive(request.request_id, timeout=timeout)
        self.paged_cache.allocate(request.request_id, len(request.token_ids))
        token_ids = torch.tensor(
            [request.token_ids], device=self.runtime.device, dtype=torch.long
        )
        try:
            if descriptor.mode is TransferMode.PARTIAL_TRANSFER:
                with torch.inference_mode():
                    self.runtime.forward(
                        token_ids,
                        paged_cache=self.paged_cache,
                        request_id=request.request_id,
                    )
            else:
                if bundle.kv_cache is None:
                    raise RuntimeError("full/quantized transfer did not include KV")
                RuntimeStateCodec.install_kv(
                    self.runtime.config,
                    self.paged_cache,
                    request.request_id,
                    bundle.kv_cache,
                )
            state = RuntimeStateCodec.install(
                self.runtime.config,
                descriptor,
                bundle,
                device=self.runtime.device,
            )
            if descriptor.state_token_count == descriptor.prompt_length - 1:
                last_prompt_token = torch.tensor(
                    [[request.token_ids[-1]]],
                    device=self.runtime.device,
                    dtype=torch.long,
                )
                with torch.inference_mode():
                    replay_logits, state = self.runtime.forward(
                        last_prompt_token,
                        state,
                        paged_cache=self.paged_cache,
                        request_id=request.request_id,
                    )
                replay_token = int(replay_logits[0, -1].argmax())
                if descriptor.first_token_id is not None and replay_token != descriptor.first_token_id:
                    raise RuntimeError(
                        f"N-1 replay token mismatch: prefill={descriptor.first_token_id}, "
                        f"decode={replay_token}"
                    )
        except Exception:
            self.paged_cache.free(request.request_id)
            request.transition(RequestState.FAILED)
            raise
        if descriptor.first_token_id is not None:
            request.generated_token_ids.append(descriptor.first_token_id)
        request.transition(RequestState.READY)
        return DecodePrepared(descriptor.first_token_id, state)
