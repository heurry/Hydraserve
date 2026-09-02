from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from hydraserve.engine.scheduler import Request, RequestState
from hydraserve.engine.sampling import TokenSample, sample_logits
from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer.backend import TransferCancelledError
from hydraserve.transfer.pipeline import TransferPipeline
from hydraserve.transfer.runtime_codec import RuntimeStateCodec
from hydraserve.transfer.descriptor import TransferMode


@dataclass(slots=True)
class PrefillResult:
    first_token_id: int
    state: RuntimeState
    state_token_count: int
    sample: TokenSample | None = None
    chunk_preemptions: int = 0


@dataclass(slots=True)
class DecodePrepared:
    first_token_id: int | None
    state: RuntimeState
    replay_consistent: bool = True


def adaptive_transfer_chunk_size(
    model,
    requested_chunk_size: int,
    block_size: int,
    *,
    target_bytes: int = 8 << 20,
    transfer_mode: TransferMode = TransferMode.FULL_TRANSFER,
) -> int:
    """Choose a page-aligned chunk whose BF16 KV payload is near target size."""
    if min(requested_chunk_size, block_size, target_bytes) <= 0:
        raise ValueError("transfer chunk limits must be positive")
    elements_per_token = (
        model.num_full_attention_layers
        * 2
        * model.num_kv_heads
        * model.head_dim
    )
    if transfer_mode is TransferMode.INT8_TRANSFER:
        bytes_per_token = elements_per_token + (elements_per_token + 63) // 64 * 4
    elif transfer_mode is TransferMode.QUANTIZED_TRANSFER:
        bytes_per_token = (elements_per_token + 1) // 2 + (
            elements_per_token + 63
        ) // 64 * 4
    else:
        bytes_per_token = elements_per_token * 2
    bytes_per_token = max(1, bytes_per_token)
    target_tokens = max(block_size, target_bytes // bytes_per_token)
    target_tokens = max(
        block_size,
        ((target_tokens + block_size // 2) // block_size) * block_size,
    )
    configured = max(block_size, requested_chunk_size)
    configured = max(block_size, (configured // block_size) * block_size)
    return min(configured, target_tokens)


class PrefillWorker:
    """GPU worker that runs prompt prefill and publishes transferable GDN state."""

    def __init__(self, runtime, pipeline: TransferPipeline, paged_cache=None) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache
        self._transfer_executor = None
        self._transfer_stream = None

    def process(
        self,
        request: Request,
        *,
        n_minus_one: bool = True,
        chunk_size: int = 4096,
        streamed_transfer: bool = False,
        reuse_host_kv: bool = False,
        host_prefix_tokens: int = 0,
        transfer_target_bytes: int = 8 << 20,
        max_inflight_chunks: int = 2,
        chunk_yield_callback=None,
    ) -> PrefillResult:
        import torch

        request.transition(RequestState.PREFILL_RUNNING)
        if reuse_host_kv:
            host_prefix_tokens = len(request.token_ids)
        if not 0 <= host_prefix_tokens <= len(request.token_ids):
            raise ValueError("host prefix length must be within the prompt")
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
            and host_prefix_tokens < len(request.token_ids)
            and self.paged_cache is not None
            and self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
        )
        if 0 < host_prefix_tokens < len(request.token_ids) and not stream_kv:
            raise ValueError("a partial host KV hit requires streamed suffix transfer")
        block_size = (
            self.paged_cache.block_manager.block_size
            if self.paged_cache is not None
            else 1
        )
        effective_chunk_size = (
            adaptive_transfer_chunk_size(
                self.runtime.config,
                chunk_size,
                block_size,
                target_bytes=transfer_target_bytes,
                transfer_mode=self.pipeline.backend.transfer_mode,
            )
            if stream_kv
            else chunk_size
        )
        produced_ranges = tuple(
            (start, min(start + effective_chunk_size, split))
            for start in range(0, split, effective_chunk_size)
        )
        if use_n_minus_one:
            produced_ranges += ((split, split + 1),)
        chunk_ranges = tuple(
            (max(start, host_prefix_tokens), end)
            for start, end in produced_ranges
            if end > host_prefix_tokens
        )
        if stream_kv:
            self.pipeline.begin_chunked_send(
                request.request_id,
                self.runtime.config,
                len(request.token_ids),
                chunk_ranges,
                prefix_tokens=host_prefix_tokens,
            )

        if max_inflight_chunks <= 0:
            raise ValueError("max_inflight_chunks must be positive")
        transfer_executor = None
        transfer_stream = None
        pending_transfers = []
        cooperative_ops = 0
        if stream_kv and self.paged_cache.device.type == "cuda":
            if self._transfer_executor is None:
                self._transfer_executor = ThreadPoolExecutor(max_workers=1)
                self._transfer_stream = torch.cuda.Stream(device=self.paged_cache.device)
            transfer_executor = self._transfer_executor
            transfer_stream = self._transfer_stream

        def extract_and_send(start, end, ready_event=None) -> None:
            if transfer_stream is None:
                payload = RuntimeStateCodec.extract_kv_range(
                    self.runtime.config,
                    self.paged_cache,
                    request.request_id,
                    start,
                    end,
                    mode=self.pipeline.backend.transfer_mode,
                )
            else:
                with torch.cuda.device(self.paged_cache.device), torch.cuda.stream(
                    transfer_stream
                ):
                    transfer_stream.wait_event(ready_event)
                    payload = RuntimeStateCodec.extract_kv_range(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        start,
                        end,
                        mode=self.pipeline.backend.transfer_mode,
                    )
            self.pipeline.send_kv_chunk(request.request_id, start, end, payload)

        def publish_kv_chunk(start, end, _state) -> None:
            nonlocal cooperative_ops
            if stream_kv:
                transfer_start = max(start, host_prefix_tokens)
                if end > transfer_start:
                    if transfer_executor is None:
                        extract_and_send(transfer_start, end)
                    else:
                        ready = torch.cuda.Event()
                        ready.record(torch.cuda.current_stream(self.paged_cache.device))
                        pending_transfers.append(
                            transfer_executor.submit(
                                extract_and_send, transfer_start, end, ready
                            )
                        )
                        if len(pending_transfers) >= max_inflight_chunks:
                            pending_transfers.pop(0).result()
            # The model state and page table are consistent at every runtime
            # chunk boundary.  Let the process run bounded short decode work
            # before the next long-prefill chunk is launched.
            if chunk_yield_callback is not None:
                cooperative_ops += int(chunk_yield_callback() or 0)

        try:
            with torch.inference_mode():
                logits, state = self.runtime.prefill(
                    prefix_ids,
                    chunk_size=effective_chunk_size,
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
                        request_id=(
                            request.request_id
                            if self.paged_cache is not None
                            else None
                        ),
                    )
                publish_kv_chunk(split, split + 1, state)
        finally:
            for future in pending_transfers:
                future.result()
        if (
            self.pipeline.backend.transfer_mode is not TransferMode.PARTIAL_TRANSFER
            and not stream_kv
            and host_prefix_tokens == 0
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
            host_cache_hit=host_prefix_tokens == len(request.token_ids),
            host_prefix_tokens=host_prefix_tokens,
        )
        return PrefillResult(
            first_token, state, split, sample, cooperative_ops
        )


class DecodeWorker:
    """GPU worker that recomputes KV for PARTIAL mode and installs transferred state."""

    def __init__(
        self, runtime, pipeline: TransferPipeline, paged_cache, host_cache=None
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.paged_cache = paged_cache
        self.host_cache = host_cache
        self._install_stream = None
        if paged_cache.device.type == "cuda":
            import torch

            self._install_stream = torch.cuda.Stream(device=paged_cache.device)

    def receive_and_prepare(
        self,
        request: Request,
        *,
        timeout: float | None = None,
        preallocated: bool = False,
        chunk_size: int = 4096,
        streamed_transfer: bool = False,
        host_prefix_tokens: int = 0,
        host_match=None,
        receiver_armed_callback: Callable[[], None] | None = None,
        cancel_event=None,
    ) -> DecodePrepared:
        import torch

        if request.state is not RequestState.TRANSFER_PENDING:
            raise RuntimeError("request is not awaiting a PD transfer")
        if chunk_size <= 0:
            raise ValueError("decode recompute chunk size must be positive")
        install_context = None
        if self._install_stream is not None:
            install_context = torch.cuda.stream(self._install_stream)
            install_context.__enter__()

        def finish_install_stream() -> None:
            if install_context is None:
                return
            ready = torch.cuda.Event()
            ready.record(self._install_stream)
            install_context.__exit__(None, None, None)
            # The receive thread waits, while the decode worker's default
            # stream remains free to execute active batches concurrently.
            ready.synchronize()

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
            received_chunks = []
            if host_prefix_tokens:
                if self.host_cache is None:
                    raise RuntimeError("host prefix restore requested without HiCache L2")
                if host_match is None:
                    host_match = self.host_cache.match(
                        self.runtime.config.name, request.token_ids
                    )
                if host_match.matched_tokens != host_prefix_tokens:
                    raise RuntimeError("reserved host KV prefix changed before restore")
                RuntimeStateCodec.install_kv_range(
                    self.runtime.config,
                    self.paged_cache,
                    request.request_id,
                    host_match.payload,
                    start=0,
                )
            # Signal readiness only after the reservation and optional host
            # prefix restore have succeeded. At this boundary the next action
            # is the blocking receive, so a producer cannot fill the bounded
            # ring before a viable consumer exists.
            if cancel_event is not None and cancel_event.is_set():
                raise TransferCancelledError("decode prepare was cancelled before receive")
            if receiver_armed_callback is not None:
                receiver_armed_callback()
            if streamed_transfer:
                received_ranges = self.pipeline.begin_chunked_receive(
                    request.request_id,
                    timeout=timeout,
                    cancel_event=cancel_event,
                )
                for start, end in received_ranges:
                    payload = self.pipeline.receive_kv_chunk(
                        request.request_id,
                        start,
                        end,
                        timeout=timeout,
                        cancel_event=cancel_event,
                    )
                    RuntimeStateCodec.install_kv_range(
                        self.runtime.config,
                        self.paged_cache,
                        request.request_id,
                        payload,
                        start=start,
                    )
                    received_chunks.append(payload)
            descriptor, bundle = self.pipeline.receive(
                request.request_id,
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if descriptor.host_cache_hit and host_match is None:
                if self.host_cache is None:
                    raise RuntimeError("host KV restore requested without HiCache L2")
                host_prefix_tokens = descriptor.host_prefix_tokens
                host_match = self.host_cache.match(
                    self.runtime.config.name, request.token_ids
                )
                if host_match.matched_tokens != host_prefix_tokens:
                    raise RuntimeError("host KV cache entry disappeared before restore")
                RuntimeStateCodec.install_kv_range(
                    self.runtime.config,
                    self.paged_cache,
                    request.request_id,
                    host_match.payload,
                    start=0,
                )
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
                    if host_match is None:
                        raise RuntimeError("host KV descriptor arrived without a restore")
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
                if descriptor.host_prefix_tokens != host_prefix_tokens:
                    raise RuntimeError("host prefix length does not match final descriptor")
                if self.host_cache is not None and not descriptor.host_cache_hit:
                    import numpy as np

                    parts = []
                    if host_match is not None:
                        parts.append(host_match.payload)
                    if descriptor.streamed_kv:
                        parts.extend(received_chunks)
                    elif bundle.kv_cache is not None:
                        parts.append(bundle.kv_cache)
                    if parts:
                        from hydraserve.cache import (
                            Int4Tensor,
                            Int8Tensor,
                            dequantize_int4,
                            dequantize_int8,
                        )

                        normalized = [
                            dequantize_int4(part)
                            if isinstance(part, Int4Tensor)
                            else dequantize_int8(part)
                            if isinstance(part, Int8Tensor)
                            else part
                            for part in parts
                        ]
                        # Reuse the CPU transfer buffers directly. This removes
                        # the former full GPU->CPU readback after installation.
                        # Lossy wire chunks are dequantized from the received
                        # host payload, never read back from decode GPU memory.
                        cached = (
                            normalized[0]
                            if len(normalized) == 1
                            else np.concatenate(normalized, axis=2)
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
            finish_install_stream()
            self.paged_cache.free(request.request_id)
            request.transition(RequestState.FAILED)
            raise
        finish_install_stream()
        if descriptor.first_token_id is not None:
            request.generated_token_ids.append(descriptor.first_token_id)
        request.transition(RequestState.READY)
        return DecodePrepared(descriptor.first_token_id, state, replay_consistent)
