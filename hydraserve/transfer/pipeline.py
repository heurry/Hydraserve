from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from hydraserve.cache.kv_quantizer import (
    Int4Tensor,
    Int8Tensor,
    PagedInt8KVTensor,
    quantize_int4,
    quantize_int8,
)
from hydraserve.cache.state_pool import LinearState
from hydraserve.config import ModelConfig
from hydraserve.transfer.backend import TransferBackend
from hydraserve.transfer.descriptor import (
    RegionDescriptor,
    RegionType,
    StateTransferDescriptor,
    TransferMode,
)


@dataclass(slots=True)
class HybridStateBundle:
    recurrent: LinearState
    kv_cache: np.ndarray | Int4Tensor | Int8Tensor | None = None


class TransferPipeline:
    """Builds and transfers mode-correct hybrid state bundles."""

    def __init__(
        self,
        backend: TransferBackend,
        src_gpu: int = 0,
        dst_gpu: int = 1,
        *,
        src_tp_rank: int = 0,
        dst_tp_rank: int = 0,
        tp_world_size: int = 1,
        bootstrap=None,
    ) -> None:
        if src_gpu == dst_gpu:
            raise ValueError("prefill and decode endpoints must differ")
        self.backend = backend
        self.src_gpu = src_gpu
        self.dst_gpu = dst_gpu
        self.src_tp_rank = src_tp_rank
        self.dst_tp_rank = dst_tp_rank
        self.tp_world_size = tp_world_size
        self.bootstrap = bootstrap
        self._receive_keys_lock = Lock()
        self._receive_keys: dict[int, set[str]] = {}
        if tp_world_size <= 0 or not (
            0 <= src_tp_rank < tp_world_size and 0 <= dst_tp_rank < tp_world_size
        ):
            raise ValueError("invalid TP transfer topology")

    def send(
        self,
        request_id: int,
        model: ModelConfig,
        prompt_length: int,
        state: HybridStateBundle,
        first_token_id: int | None = None,
        state_token_count: int | None = None,
        streamed_kv_ranges: tuple[tuple[int, int], ...] = (),
        host_cache_hit: bool = False,
        host_prefix_tokens: int = 0,
    ) -> StateTransferDescriptor:
        mode = self.backend.transfer_mode
        recurrent = state.recurrent
        if recurrent.ssm_state.shape != model.ssm_state_shape:
            raise ValueError("SSM state does not match model configuration")
        if recurrent.conv_state.shape != model.conv_state_shape:
            raise ValueError("conv state does not match model configuration")

        regions = [
            self._region(RegionType.LINEAR_SSM, model.linear_layer_indices, recurrent.ssm_state),
            self._region(RegionType.LINEAR_CONV, model.linear_layer_indices, recurrent.conv_state),
        ]
        payload: dict[str, Any] = {
            RegionType.LINEAR_SSM.value: recurrent.ssm_state,
            RegionType.LINEAR_CONV.value: recurrent.conv_state,
        }
        if mode is not TransferMode.PARTIAL_TRANSFER:
            if state.kv_cache is None and not streamed_kv_ranges and not host_cache_hit:
                raise ValueError(f"{mode.value} transfer requires a KV cache")
            if streamed_kv_ranges or host_cache_hit:
                kv_shape = (
                    model.num_full_attention_layers,
                    2,
                    prompt_length,
                    model.num_kv_heads,
                    model.head_dim,
                )
                if mode is TransferMode.QUANTIZED_TRANSFER:
                    dtype, quantized = "int4", True
                elif mode is TransferMode.INT8_TRANSFER:
                    dtype, quantized = "int8", True
                else:
                    dtype, quantized = "uint16", False
                kv_payload = None
            elif mode is TransferMode.QUANTIZED_TRANSFER:
                kv_payload = (
                    state.kv_cache if isinstance(state.kv_cache, Int4Tensor)
                    else quantize_int4(state.kv_cache)
                )
                kv_shape = kv_payload.shape
                dtype, quantized = "int4", True
            elif mode is TransferMode.INT8_TRANSFER:
                kv_payload = (
                    state.kv_cache
                    if isinstance(state.kv_cache, (Int8Tensor, PagedInt8KVTensor))
                    else quantize_int8(state.kv_cache)
                )
                kv_shape = kv_payload.shape
                dtype, quantized = "int8", True
            else:
                kv_payload = state.kv_cache
                kv_shape = state.kv_cache.shape
                dtype, quantized = str(state.kv_cache.dtype), False
            regions.append(
                RegionDescriptor(
                    RegionType.FULL_ATTN_KV,
                    model.full_attention_layer_indices,
                    tuple(kv_shape),
                    dtype,
                    quantized,
                    self.src_gpu,
                    self.dst_gpu,
                    self.src_tp_rank,
                    self.dst_tp_rank,
                    self.tp_world_size,
                )
            )
            if kv_payload is not None:
                payload[RegionType.FULL_ATTN_KV.value] = kv_payload

        descriptor = StateTransferDescriptor(
            request_id=request_id,
            model_name=model.name,
            prompt_length=prompt_length,
            first_token_id=first_token_id,
            mode=mode,
            regions=tuple(regions),
            state_token_count=state_token_count,
            streamed_kv=bool(streamed_kv_ranges),
            kv_chunk_ranges=streamed_kv_ranges,
            host_cache_hit=host_cache_hit,
            host_prefix_tokens=host_prefix_tokens,
        )
        key = self._key(request_id)
        self.backend.send(
            f"{key}:bundle",
            {"descriptor": descriptor.to_dict(), "payload": payload},
            self.dst_gpu,
        )
        return descriptor

    def begin_chunked_send(
        self,
        request_id: int,
        model: ModelConfig,
        prompt_length: int,
        chunk_ranges: tuple[tuple[int, int], ...],
        *,
        prefix_tokens: int = 0,
    ) -> None:
        """Publish a manifest before prefill so decode can receive immediately."""
        if self.backend.transfer_mode is TransferMode.PARTIAL_TRANSFER:
            raise ValueError("partial transfer has no KV chunks")
        if not 0 <= prefix_tokens < prompt_length:
            raise ValueError("chunked transfer prefix must leave a suffix")
        if not chunk_ranges or chunk_ranges[0][0] != prefix_tokens:
            raise ValueError("chunk ranges must start after the cached prefix")
        previous = prefix_tokens
        for start, end in chunk_ranges:
            if start != previous or end <= start or end > prompt_length:
                raise ValueError("chunk ranges must be contiguous")
            previous = end
        if previous != prompt_length:
            raise ValueError("chunk ranges must cover the prompt")
        manifest = {
                "request_id": request_id,
                "model_name": model.name,
                "prompt_length": prompt_length,
                "mode": self.backend.transfer_mode.value,
                "prefix_tokens": prefix_tokens,
                "ranges": [list(item) for item in chunk_ranges],
            }
        if self.bootstrap is None:
            self.backend.send(
                f"{self._key(request_id)}:chunks:manifest",
                manifest,
                self.dst_gpu,
            )
        else:
            self.bootstrap.publish(request_id, "kv_chunks", manifest)

    def begin_chunked_receive(
        self,
        request_id: int,
        *,
        timeout: float | None = None,
        cancel_event=None,
    ) -> tuple[tuple[int, int], ...]:
        manifest = (
            self.backend.receive(
                f"{self._key(request_id)}:chunks:manifest",
                self.dst_gpu,
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if self.bootstrap is None
            else self.bootstrap.consume(
                request_id, "kv_chunks", timeout=timeout
            )
        )
        if int(manifest.get("request_id", -1)) != request_id:
            raise RuntimeError("received KV chunk manifest for the wrong request")
        if TransferMode(manifest["mode"]) is not self.backend.transfer_mode:
            raise RuntimeError("KV chunk manifest transfer mode mismatch")
        ranges = tuple((int(item[0]), int(item[1])) for item in manifest["ranges"])
        with self._receive_keys_lock:
            self._receive_keys[request_id] = {
                f"{self._key(request_id)}:chunks:{start}:{end}"
                for start, end in ranges
            } | {f"{self._key(request_id)}:bundle"}
        return ranges

    def send_kv_chunk(self, request_id: int, start: int, end: int, payload) -> None:
        mode = self.backend.transfer_mode
        if mode is TransferMode.PARTIAL_TRANSFER:
            raise ValueError("partial transfer has no KV chunks")
        if mode is TransferMode.QUANTIZED_TRANSFER and not isinstance(payload, Int4Tensor):
            payload = quantize_int4(payload)
        if mode is TransferMode.INT8_TRANSFER and not isinstance(
            payload, (Int8Tensor, PagedInt8KVTensor)
        ):
            payload = quantize_int8(payload)
        self.backend.send(
            f"{self._key(request_id)}:chunks:{start}:{end}",
            payload,
            self.dst_gpu,
        )

    def receive_kv_chunk(
        self,
        request_id: int,
        start: int,
        end: int,
        *,
        timeout: float | None = None,
        cancel_event=None,
    ):
        return self.backend.receive(
            f"{self._key(request_id)}:chunks:{start}:{end}",
            self.dst_gpu,
            timeout=timeout,
            cancel_event=cancel_event,
        )

    def receive(
        self,
        request_id: int,
        timeout: float | None = None,
        *,
        cancel_event=None,
    ) -> tuple[StateTransferDescriptor, HybridStateBundle]:
        key = self._key(request_id)
        envelope = self.backend.receive(
            f"{key}:bundle",
            self.dst_gpu,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        if not isinstance(envelope, dict) or set(envelope) != {"descriptor", "payload"}:
            raise RuntimeError("received an invalid state-transfer envelope")
        descriptor = StateTransferDescriptor.from_dict(envelope["descriptor"])
        payload = envelope["payload"]
        if descriptor.request_id != request_id:
            raise RuntimeError("received descriptor for the wrong request")
        with self._receive_keys_lock:
            self._receive_keys.pop(request_id, None)
        recurrent = LinearState(
            payload[RegionType.LINEAR_SSM.value], payload[RegionType.LINEAR_CONV.value]
        )
        return descriptor, HybridStateBundle(
            recurrent=recurrent, kv_cache=payload.get(RegionType.FULL_ATTN_KV.value)
        )

    def cancel_receive(self, request_id: int) -> None:
        """Wake a receiver blocked on the metadata control-plane handshake."""
        bootstrap_error = None
        if self.bootstrap is not None:
            try:
                self.bootstrap.cancel(request_id, "kv_chunks")
            except Exception as exc:  # data-plane cleanup must still run
                bootstrap_error = exc
        with self._receive_keys_lock:
            keys = self._receive_keys.pop(request_id, set())
        keys.update(
            {
                f"{self._key(request_id)}:chunks:manifest",
                f"{self._key(request_id)}:bundle",
            }
        )
        discard = getattr(self.backend, "discard", None)
        if discard is not None:
            for key in keys:
                discard(key, self.dst_gpu)
        if bootstrap_error is not None:
            raise bootstrap_error

    def _region(
        self, region_type: RegionType, layer_indices: tuple[int, ...], tensor: np.ndarray
    ) -> RegionDescriptor:
        return RegionDescriptor(
            region_type,
            layer_indices,
            tuple(tensor.shape),
            str(tensor.dtype),
            False,
            self.src_gpu,
            self.dst_gpu,
            self.src_tp_rank,
            self.dst_tp_rank,
            self.tp_world_size,
        )

    @staticmethod
    def _key(request_id: int) -> str:
        return f"request:{request_id}"
