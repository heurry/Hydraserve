from __future__ import annotations

from dataclasses import dataclass

from hydraserve.engine.scheduler import Request, RequestState
from hydraserve.engine.sampling import TokenSample, sample_logits
from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer.pipeline import TransferPipeline
from hydraserve.transfer.runtime_codec import RuntimeStateCodec
from hydraserve.transfer.descriptor import TransferMode


@dataclass(slots=True)
class PrefillResult:
    first_token_id: int
    state: RuntimeState
    state_token_count: int
    sample: TokenSample | None = None


@dataclass(slots=True)
class DecodePrepared:
    first_token_id: int | None
    state: RuntimeState
    replay_consistent: bool = True


class PrefillWorker:
    """GPU worker that runs prompt prefill and publishes transferable GDN state."""

    def __init__(self, runtime, pipeline: TransferPipeline, paged_cache=None) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache

    def process(
        self,
        request: Request,
        *,
        n_minus_one: bool = True,
        chunk_size: int = 4096,
    ) -> PrefillResult:
        import torch

        request.transition(RequestState.PREFILL_RUNNING)
        use_n_minus_one = n_minus_one and len(request.token_ids) > 1
        split = len(request.token_ids) - 1 if use_n_minus_one else len(request.token_ids)
        if self.paged_cache is not None:
            self.paged_cache.allocate(
                request.request_id,
                len(request.token_ids),
                token_ids=request.token_ids,
            )
        if (
            self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
            and self.paged_cache is None
        ):
            raise RuntimeError("full/quantized transfer requires a prefill Paged KV cache")
        prefix_ids = torch.tensor(
            [request.token_ids[:split]],
            device=getattr(self.runtime, "input_device", self.runtime.device),
            dtype=torch.long,
        )
        with torch.inference_mode():
            logits, state = self.runtime.prefill(
                prefix_ids,
                chunk_size=chunk_size,
                paged_cache=self.paged_cache,
                request_id=request.request_id if self.paged_cache is not None else None,
            )
        bundle = RuntimeStateCodec.extract(self.runtime.config, state)
        if use_n_minus_one:
            last_id = torch.tensor(
                [[request.token_ids[-1]]],
                device=getattr(self.runtime, "input_device", self.runtime.device),
                dtype=torch.long,
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
        sample = sample_logits(
            logits[:, -1],
            (request.token_ids,),
            (request.sampling_params,),
            steps=(0,),
        )[0]
        first_token = sample.token_id
        request.transition(RequestState.TRANSFER_PENDING)
        self.pipeline.send(
            request.request_id,
            self.runtime.config,
            len(request.token_ids),
            bundle,
            first_token_id=first_token,
            state_token_count=split,
        )
        return PrefillResult(first_token, state, split, sample)


class DecodeWorker:
    """GPU worker that recomputes KV for PARTIAL mode and installs transferred state."""

    def __init__(self, runtime, pipeline: TransferPipeline, paged_cache) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache

    def receive_and_prepare(
        self,
        request: Request,
        *,
        timeout: float | None = None,
        preallocated: bool = False,
        chunk_size: int = 4096,
    ) -> DecodePrepared:
        import torch

        if request.state is not RequestState.TRANSFER_PENDING:
            raise RuntimeError("request is not awaiting a PD transfer")
        if chunk_size <= 0:
            raise ValueError("decode recompute chunk size must be positive")
        try:
            total_tokens = len(request.token_ids) + max(0, request.max_new_tokens - 1)
            if preallocated:
                allocation = self.paged_cache.block_manager.get(request.request_id)
                if allocation.num_tokens != len(request.token_ids):
                    raise RuntimeError("preallocated KV logical length does not match prompt")
                if allocation.reserved_tokens < total_tokens:
                    raise RuntimeError("preallocated KV capacity does not cover maximum output")
            else:
                self.paged_cache.allocate(
                    request.request_id,
                    len(request.token_ids),
                    reserve_tokens=total_tokens,
                    token_ids=request.token_ids,
                )
            descriptor, bundle = self.pipeline.receive(request.request_id, timeout=timeout)
            token_ids = torch.tensor(
                [request.token_ids],
                device=getattr(self.runtime, "input_device", self.runtime.device),
                dtype=torch.long,
            )
            if descriptor.mode is TransferMode.PARTIAL_TRANSFER:
                with torch.inference_mode():
                    self.runtime.prefill(
                        token_ids,
                        chunk_size=chunk_size,
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
            replay_consistent = True
            if descriptor.state_token_count == descriptor.prompt_length - 1:
                last_prompt_token = torch.tensor(
                    [[request.token_ids[-1]]],
                    device=getattr(self.runtime, "input_device", self.runtime.device),
                    dtype=torch.long,
                )
                with torch.inference_mode():
                    replay_logits, state = self.runtime.forward(
                        last_prompt_token,
                        state,
                        paged_cache=self.paged_cache,
                        request_id=request.request_id,
                    )
                replay_token = sample_logits(
                    replay_logits[:, -1],
                    (request.token_ids,),
                    (request.sampling_params,),
                    steps=(0,),
                )[0].token_id
                replay_consistent = (
                    descriptor.first_token_id is None
                    or replay_token == descriptor.first_token_id
                )
            self.paged_cache.publish_prefix(request.request_id, request.token_ids)
        except Exception:
            self.paged_cache.free(request.request_id)
            request.transition(RequestState.FAILED)
            raise
        if descriptor.first_token_id is not None:
            request.generated_token_ids.append(descriptor.first_token_id)
        request.transition(RequestState.READY)
        return DecodePrepared(descriptor.first_token_id, state, replay_consistent)
