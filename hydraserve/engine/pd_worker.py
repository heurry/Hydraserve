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
        streamed_transfer: bool = False,
        reuse_host_kv: bool = False,
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
        stream_kv = (
            streamed_transfer
            and not reuse_host_kv
            and self.paged_cache is not None
            and self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
        )
        chunk_ranges = tuple(
            (start, min(start + chunk_size, split))
            for start in range(0, split, chunk_size)
        )
        if use_n_minus_one:
            chunk_ranges += ((split, split + 1),)
        if stream_kv:
            self.pipeline.begin_chunked_send(
                request.request_id,
                self.runtime.config,
                len(request.token_ids),
                chunk_ranges,
            )

        def publish_kv_chunk(start, end, _state) -> None:
            if not stream_kv:
                return
            payload = RuntimeStateCodec.extract_kv_range(
                self.runtime.config,
                self.paged_cache,
                request.request_id,
                start,
                end,
                mode=self.pipeline.backend.transfer_mode,
            )
            self.pipeline.send_kv_chunk(request.request_id, start, end, payload)

        with torch.inference_mode():
            logits, state = self.runtime.prefill(
                prefix_ids,
                chunk_size=chunk_size,
                paged_cache=self.paged_cache,
                request_id=request.request_id if self.paged_cache is not None else None,
                chunk_callback=publish_kv_chunk,
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
            publish_kv_chunk(split, split + 1, state)
        if (
            self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
            and not stream_kv
            and not reuse_host_kv
        ):
            bundle.kv_cache = RuntimeStateCodec.extract_kv(
                self.runtime.config, self.paged_cache, request.request_id,
                mode=self.pipeline.backend.transfer_mode,
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
            streamed_kv_ranges=chunk_ranges if stream_kv else (),
            host_cache_hit=reuse_host_kv,
        )
        return PrefillResult(first_token, state, split, sample)


class DecodeWorker:
    """GPU worker that recomputes KV for PARTIAL mode and installs transferred state."""

    def __init__(
        self, runtime, pipeline: TransferPipeline, paged_cache, host_cache=None
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache
        self.host_cache = host_cache

    def receive_and_prepare(
        self,
        request: Request,
        *,
        timeout: float | None = None,
        preallocated: bool = False,
        chunk_size: int = 4096,
        streamed_transfer: bool = False,
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
            received_ranges = ()
            if streamed_transfer:
                received_ranges = self.pipeline.begin_chunked_receive(
                    request.request_id, timeout=timeout
                )
                for start, end in received_ranges:
                    payload = self.pipeline.receive_kv_chunk(
                        request.request_id, start, end, timeout=timeout
                    )
                    RuntimeStateCodec.install_kv_range(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        payload,
                        start=start,
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
                if descriptor.host_cache_hit:
                    if self.host_cache is None:
                        raise RuntimeError("host KV restore requested without HiCache L2")
                    payload = self.host_cache.get(
                        self.runtime.config.name, request.token_ids
                    )
                    if payload is None:
                        raise RuntimeError("host KV cache entry disappeared before restore")
                    RuntimeStateCodec.install_kv(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        payload,
                    )
                elif descriptor.streamed_kv:
                    if tuple(descriptor.kv_chunk_ranges) != tuple(received_ranges):
                        raise RuntimeError("received KV chunks do not match final descriptor")
                elif bundle.kv_cache is None:
                    raise RuntimeError("full/quantized transfer did not include KV")
                else:
                    RuntimeStateCodec.install_kv(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        bundle.kv_cache,
                    )
                if self.host_cache is not None and not descriptor.host_cache_hit:
                    cached = RuntimeStateCodec.extract_kv(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        mode=TransferMode.FULL_TRANSFER,
                    )
                    self.host_cache.put(
                        self.runtime.config.name, request.token_ids, cached
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
