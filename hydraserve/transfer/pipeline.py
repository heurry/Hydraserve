from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hydraserve.cache.kv_quantizer import Int4Tensor, quantize_int4
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
    kv_cache: np.ndarray | Int4Tensor | None = None


class TransferPipeline:
    """Builds and transfers mode-correct hybrid state bundles."""

    def __init__(self, backend: TransferBackend, src_gpu: int = 0, dst_gpu: int = 1) -> None:
        if src_gpu == dst_gpu:
            raise ValueError("prefill and decode endpoints must differ")
        self.backend = backend
        self.src_gpu = src_gpu
        self.dst_gpu = dst_gpu

    def send(
        self,
        request_id: int,
        model: ModelConfig,
        prompt_length: int,
        state: HybridStateBundle,
        first_token_id: int | None = None,
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
            if state.kv_cache is None:
                raise ValueError(f"{mode.value} transfer requires a KV cache")
            if mode is TransferMode.QUANTIZED_TRANSFER:
                kv_payload = (
                    state.kv_cache if isinstance(state.kv_cache, Int4Tensor)
                    else quantize_int4(state.kv_cache)
                )
                kv_shape = kv_payload.shape
                dtype, quantized = "int4", True
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
                )
            )
            payload[RegionType.FULL_ATTN_KV.value] = kv_payload

        descriptor = StateTransferDescriptor(
            request_id=request_id,
            model_name=model.name,
            prompt_length=prompt_length,
            first_token_id=first_token_id,
            mode=mode,
            regions=tuple(regions),
        )
        key = self._key(request_id)
        self.backend.send(f"{key}:descriptor", descriptor.to_dict(), self.dst_gpu)
        self.backend.send(f"{key}:payload", payload, self.dst_gpu)
        return descriptor

    def receive(self, request_id: int, timeout: float | None = None) -> tuple[StateTransferDescriptor, HybridStateBundle]:
        key = self._key(request_id)
        raw_descriptor = self.backend.receive(
            f"{key}:descriptor", self.dst_gpu, timeout=timeout
        )
        payload = self.backend.receive(f"{key}:payload", self.dst_gpu, timeout=timeout)
        descriptor = StateTransferDescriptor.from_dict(raw_descriptor)
        if descriptor.request_id != request_id:
            raise RuntimeError("received descriptor for the wrong request")
        recurrent = LinearState(
            payload[RegionType.LINEAR_SSM.value], payload[RegionType.LINEAR_CONV.value]
        )
        return descriptor, HybridStateBundle(
            recurrent=recurrent, kv_cache=payload.get(RegionType.FULL_ATTN_KV.value)
        )

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
        )

    @staticmethod
    def _key(request_id: int) -> str:
        return f"request:{request_id}"
