"""Bridge between model-runtime state dictionaries and transfer regions."""

from __future__ import annotations

from threading import RLock

from hydraserve.cache.state_pool import LinearState
from hydraserve.config import ModelConfig
from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer.descriptor import StateTransferDescriptor, TransferMode
from hydraserve.transfer.pipeline import HybridStateBundle
from hydraserve.cache.kv_quantizer import (
    Int4Tensor,
    Int8Tensor,
    PagedInt8KVTensor,
    dequantize_int4,
    dequantize_int8,
    dequantize_int8_torch,
    quantize_int8_torch,
)


class _PinnedStagingPool:
    """Per-process reusable pinned buffers for synchronous transport handoff."""

    def __init__(self) -> None:
        self._buffers = {}
        self._lock = RLock()

    def stage(self, tensor, *, slot: int = 0):
        import torch

        if not tensor.is_cuda:
            return tensor.cpu()
        key = (tuple(tensor.shape), tensor.dtype, tensor.device.index, slot)
        with self._lock:
            destination = self._buffers.get(key)
            if destination is None:
                destination = torch.empty(
                    tensor.shape,
                    device="cpu",
                    dtype=tensor.dtype,
                    pin_memory=True,
                )
                self._buffers[key] = destination
            destination.copy_(tensor, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(tensor.device))
            # Wait only for this D2H copy. Unlike stream.synchronize(), later
            # unrelated work submitted to the transfer stream is not drained.
            ready.synchronize()
            return destination


_PINNED_STAGING = _PinnedStagingPool()


class RuntimeStateCodec:
    @staticmethod
    def extract(model: ModelConfig, state: RuntimeState) -> HybridStateBundle:
        """Synchronously stage one request's GDN state as contiguous CPU FP32."""
        import torch

        missing_recurrent = set(model.linear_layer_indices) - state.recurrent.keys()
        missing_convolution = set(model.linear_layer_indices) - state.convolution.keys()
        if missing_recurrent or missing_convolution:
            raise ValueError(
                f"runtime state is incomplete: recurrent={sorted(missing_recurrent)}, "
                f"conv={sorted(missing_convolution)}"
            )
        recurrent_layers = []
        convolution_layers = []
        for layer_index in model.linear_layer_indices:
            recurrent = state.recurrent[layer_index]
            convolution = state.convolution[layer_index]
            if recurrent.shape != (1, *model.ssm_state_shape[1:]):
                raise ValueError(f"unexpected recurrent shape at layer {layer_index}")
            if convolution.shape != (1, *model.conv_state_shape[1:]):
                raise ValueError(f"unexpected conv shape at layer {layer_index}")
            recurrent_layers.append(recurrent[0].float())
            convolution_layers.append(convolution[0].float())
        recurrent_tensor = torch.stack(recurrent_layers).contiguous()
        convolution_tensor = torch.stack(convolution_layers).contiguous()
        recurrent_tensor, convolution_tensor = RuntimeStateCodec._stage_to_host(
            recurrent_tensor, convolution_tensor
        )
        return HybridStateBundle(
            recurrent=LinearState(
                recurrent_tensor.numpy(), convolution_tensor.numpy()
            )
        )

    @staticmethod
    def extract_kv(model: ModelConfig, paged_cache, request_id: int, *, mode: TransferMode = TransferMode.FULL_TRANSFER):
        """Gather full-attention KV as ``[layers, 2, tokens, heads, dim]``.

        FULL transfer ships BF16 raw bits viewed as uint16 (half the FP32 size,
        lossless round-trip); QUANTIZED still needs a floating-point tensor for
        ``quantize_int4`` on the pipeline side.
        """
        import torch

        if mode is TransferMode.INT8_TRANSFER:
            allocation = paged_cache.block_manager.get(request_id)
            raw = RuntimeStateCodec._extract_raw_int8_kv(
                model, paged_cache, request_id, 0, allocation.num_tokens
            )
            if raw is not None:
                return raw

        layers = []
        for layer_index in model.full_attention_layer_indices:
            key, value = paged_cache.read(request_id, layer_index)
            layers.append(torch.stack((key, value), dim=0))
        stacked = torch.stack(layers, dim=0)
        if mode is TransferMode.INT8_TRANSFER:
            return quantize_int8_torch(stacked)
        if mode is TransferMode.QUANTIZED_TRANSFER:
            return stacked.float().cpu().numpy()
        if stacked.dtype is torch.bfloat16:
            return _PINNED_STAGING.stage(stacked.view(torch.uint16)).numpy()
        return _PINNED_STAGING.stage(stacked).numpy()

    @staticmethod
    def extract_kv_range(
        model: ModelConfig,
        paged_cache,
        request_id: int,
        start: int,
        end: int,
        *,
        mode: TransferMode = TransferMode.FULL_TRANSFER,
    ):
        """Gather one completed logical token range for streaming transfer."""
        if start < 0 or end <= start:
            raise ValueError("invalid KV token range")
        import torch

        if mode is TransferMode.INT8_TRANSFER:
            raw = RuntimeStateCodec._extract_raw_int8_kv(
                model, paged_cache, request_id, start, end
            )
            if raw is not None:
                return raw

        if paged_cache.device.type == "cuda" and paged_cache.kv_quant is None:
            from hydraserve.kernels.staging import fused_gather_paged_kv

            allocation = paged_cache.block_manager.get(request_id)
            block_ids = torch.tensor(
                allocation.block_ids,
                device=paged_cache.device,
                dtype=torch.int32,
            )
            stacked = fused_gather_paged_kv(
                paged_cache.key,
                paged_cache.value,
                block_ids,
                start,
                end,
            )
            if mode is TransferMode.INT8_TRANSFER:
                return quantize_int8_torch(stacked)
            if mode is TransferMode.QUANTIZED_TRANSFER:
                return stacked.float().cpu().numpy()
            if stacked.dtype is torch.bfloat16:
                return _PINNED_STAGING.stage(stacked.view(torch.uint16)).numpy()
            return _PINNED_STAGING.stage(stacked).numpy()

        layers = []
        for layer_index in model.full_attention_layer_indices:
            key, value = paged_cache.read(request_id, layer_index, num_tokens=end)
            layers.append(torch.stack((key[start:end], value[start:end]), dim=0))
        stacked = torch.stack(layers, dim=0).contiguous()
        if mode is TransferMode.INT8_TRANSFER:
            return quantize_int8_torch(stacked)
        if mode is TransferMode.QUANTIZED_TRANSFER:
            return stacked.float().cpu().numpy()
        if stacked.dtype is torch.bfloat16:
            return _PINNED_STAGING.stage(stacked.view(torch.uint16)).numpy()
        return _PINNED_STAGING.stage(stacked).numpy()

    @staticmethod
    def install_kv(model: ModelConfig, paged_cache, request_id: int, payload) -> None:
        RuntimeStateCodec.install_kv_range(
            model, paged_cache, request_id, payload, start=0
        )

    @staticmethod
    def install_kv_range(
        model: ModelConfig, paged_cache, request_id: int, payload, *, start: int
    ) -> None:
        import numpy as np
        import torch

        device_values = None
        if isinstance(payload, Int8Tensor) and paged_cache.device.type == "cuda":
            device_values = dequantize_int8_torch(payload, device=paged_cache.device)
            values = device_values
        elif isinstance(payload, PagedInt8KVTensor):
            if paged_cache.kv_quant == "int8":
                RuntimeStateCodec._install_raw_int8_kv(
                    model, paged_cache, request_id, payload, start=start
                )
                return
            device_values = RuntimeStateCodec._dequantize_raw_int8_kv(
                payload, device=paged_cache.device
            )
            values = device_values
        elif isinstance(payload, Int8Tensor):
            values = dequantize_int8(payload)
        elif isinstance(payload, Int4Tensor):
            values = dequantize_int4(payload)
        else:
            values = np.asarray(payload)
        expected_prefix = (
            model.num_full_attention_layers,
            2,
        )
        if values.shape[:2] != expected_prefix or values.shape[3:] != (
            model.num_kv_heads,
            model.head_dim,
        ):
            raise ValueError(f"invalid transferred KV shape {values.shape}")
        if start < 0:
            raise ValueError("KV range start must be non-negative")
        positions = torch.arange(
            start, start + values.shape[2], device=paged_cache.device
        )
        # FULL transfer ships BF16 as uint16 raw bits; reinterpret, don't convert.
        bf16_bits = isinstance(values, np.ndarray) and values.dtype == np.uint16
        if (
            paged_cache.device.type == "cuda"
            and paged_cache.kv_quant is None
            and (bf16_bits or device_values is not None)
        ):
            from hydraserve.kernels.staging import fused_scatter_paged_kv

            allocation = paged_cache.block_manager.get(request_id)
            block_ids = torch.tensor(
                allocation.block_ids,
                device=paged_cache.device,
                dtype=torch.int32,
            )
            staging = (
                device_values.to(dtype=paged_cache.dtype)
                if device_values is not None
                else torch.from_numpy(values).view(torch.bfloat16).to(
                    device=paged_cache.device, non_blocking=True
                )
            )
            fused_scatter_paged_kv(
                staging,
                paged_cache.key,
                paged_cache.value,
                block_ids,
                start,
            )
            return
        for slot, layer_index in enumerate(model.full_attention_layer_indices):
            if isinstance(values, torch.Tensor):
                key = values[slot, 0]
                value = values[slot, 1]
            else:
                key = torch.from_numpy(values[slot, 0])
                value = torch.from_numpy(values[slot, 1])
            if bf16_bits:
                key = key.view(torch.bfloat16)
                value = value.view(torch.bfloat16)
            key = key.to(device=paged_cache.device, dtype=paged_cache.dtype)
            value = value.to(device=paged_cache.device, dtype=paged_cache.dtype)
            paged_cache.write(request_id, layer_index, positions, key, value)

    @staticmethod
    def _extract_raw_int8_kv(
        model: ModelConfig,
        paged_cache,
        request_id: int,
        start: int,
        end: int,
    ) -> PagedInt8KVTensor | None:
        """Gather the native INT8 cache layout without BF16 dequantization."""

        if getattr(paged_cache, "kv_quant", None) != "int8":
            return None
        import torch

        allocation = paged_cache.block_manager.get(request_id)
        length = end - start
        if start < 0 or length <= 0 or end > allocation.num_tokens:
            raise ValueError("invalid raw INT8 KV range")
        physical = torch.tensor(
            allocation.block_ids,
            device=paged_cache.device,
            dtype=torch.long,
        )
        keys = []
        values = []
        key_scales = []
        value_scales = []
        for layer_index in model.full_attention_layer_indices:
            slot = paged_cache.layer_to_slot[layer_index]
            key = paged_cache.key[slot, physical].reshape(
                -1, model.num_kv_heads, model.head_dim
            )[start:end]
            value = paged_cache.value[slot, physical].reshape(
                -1, model.num_kv_heads, model.head_dim
            )[start:end]
            k_scale = paged_cache.key_scales[slot, physical].reshape(
                -1, model.num_kv_heads
            )[start:end]
            v_scale = paged_cache.value_scales[slot, physical].reshape(
                -1, model.num_kv_heads
            )[start:end]
            keys.append(key.contiguous())
            values.append(value.contiguous())
            key_scales.append(k_scale.contiguous())
            value_scales.append(v_scale.contiguous())
        key_tensor = torch.stack(keys, dim=0)
        value_tensor = torch.stack(values, dim=0)
        key_scale_tensor = torch.stack(key_scales, dim=0)
        value_scale_tensor = torch.stack(value_scales, dim=0)
        return PagedInt8KVTensor(
            _PINNED_STAGING.stage(key_tensor).numpy(),
            _PINNED_STAGING.stage(value_tensor, slot=1).numpy(),
            _PINNED_STAGING.stage(key_scale_tensor, slot=2).numpy(),
            _PINNED_STAGING.stage(value_scale_tensor, slot=3).numpy(),
            (
                model.num_full_attention_layers,
                2,
                length,
                model.num_kv_heads,
                model.head_dim,
            ),
            str(paged_cache.dtype).removeprefix("torch."),
        )

    @staticmethod
    def _install_raw_int8_kv(
        model: ModelConfig,
        paged_cache,
        request_id: int,
        payload: PagedInt8KVTensor,
        *,
        start: int,
    ) -> None:
        """Install raw INT8 KV/scales directly into a quantized PagedKVCache."""

        import torch

        expected = (
            model.num_full_attention_layers,
            2,
            payload.key.shape[1],
            model.num_kv_heads,
            model.head_dim,
        )
        if payload.shape != expected:
            raise ValueError(f"invalid raw INT8 KV shape {payload.shape}")
        if payload.key.shape != payload.value.shape or payload.key.shape != (
            model.num_full_attention_layers,
            payload.shape[2],
            model.num_kv_heads,
            model.head_dim,
        ):
            raise ValueError("invalid raw INT8 KV tensor layout")
        if payload.key_scales.shape != payload.value_scales.shape or payload.key_scales.shape != (
            model.num_full_attention_layers,
            payload.shape[2],
            model.num_kv_heads,
        ):
            raise ValueError("invalid raw INT8 KV scale layout")
        allocation = paged_cache.block_manager.get(request_id)
        end = start + payload.shape[2]
        if start < 0 or end > allocation.num_tokens:
            raise ValueError("raw INT8 KV range exceeds allocation")
        positions = torch.arange(start, end, device=paged_cache.device)
        block_ids = torch.tensor(
            allocation.block_ids,
            device=paged_cache.device,
            dtype=torch.long,
        )
        logical = torch.div(
            positions,
            paged_cache.block_manager.block_size,
            rounding_mode="floor",
        ).long()
        offsets = positions.remainder(paged_cache.block_manager.block_size).long()
        physical = block_ids[logical]
        for slot, layer_index in enumerate(model.full_attention_layer_indices):
            cache_slot = paged_cache.layer_to_slot[layer_index]
            key = torch.from_numpy(payload.key[slot]).to(
                device=paged_cache.device, dtype=torch.int8, non_blocking=True
            )
            value = torch.from_numpy(payload.value[slot]).to(
                device=paged_cache.device, dtype=torch.int8, non_blocking=True
            )
            key_scale = torch.from_numpy(payload.key_scales[slot]).to(
                device=paged_cache.device, dtype=torch.float32, non_blocking=True
            )
            value_scale = torch.from_numpy(payload.value_scales[slot]).to(
                device=paged_cache.device, dtype=torch.float32, non_blocking=True
            )
            paged_cache.key[cache_slot, physical, offsets] = key
            paged_cache.value[cache_slot, physical, offsets] = value
            paged_cache.key_scales[cache_slot, physical, offsets] = key_scale
            paged_cache.value_scales[cache_slot, physical, offsets] = value_scale

    @staticmethod
    def _dequantize_raw_int8_kv(payload: PagedInt8KVTensor, *, device):
        import torch

        key = torch.from_numpy(payload.key).to(device, non_blocking=True).float()
        value = torch.from_numpy(payload.value).to(device, non_blocking=True).float()
        key_scales = torch.from_numpy(payload.key_scales).to(
            device, non_blocking=True
        )
        value_scales = torch.from_numpy(payload.value_scales).to(
            device, non_blocking=True
        )
        return torch.stack(
            (
                key * key_scales.unsqueeze(-1),
                value * value_scales.unsqueeze(-1),
            ),
            dim=1,
        )

    @staticmethod
    def install(
        model: ModelConfig,
        descriptor: StateTransferDescriptor,
        bundle: HybridStateBundle,
        *,
        device,
    ) -> RuntimeState:
        """Restore transferred CPU regions into per-layer GPU FP32 state."""
        import torch

        recurrent = bundle.recurrent
        if recurrent.ssm_state.shape != model.ssm_state_shape:
            raise ValueError("transferred recurrent state shape does not match the model")
        if recurrent.conv_state.shape != model.conv_state_shape:
            raise ValueError("transferred conv state shape does not match the model")
        state = RuntimeState(sequence_length=descriptor.state_token_count)
        recurrent_tensor, convolution_tensor = RuntimeStateCodec._stage_to_device(
            recurrent.ssm_state,
            recurrent.conv_state,
            device=device,
        )
        for slot, layer_index in enumerate(model.linear_layer_indices):
            state.recurrent[layer_index] = recurrent_tensor[slot : slot + 1].contiguous()
            state.convolution[layer_index] = convolution_tensor[slot : slot + 1].contiguous()
        return state

    @staticmethod
    def _stage_to_host(*tensors):
        import torch

        if not tensors or not tensors[0].is_cuda:
            return tuple(tensor.cpu() for tensor in tensors)
        return tuple(
            _PINNED_STAGING.stage(tensor, slot=slot)
            for slot, tensor in enumerate(tensors)
        )

    @staticmethod
    def _stage_to_device(*arrays, device):
        import torch

        target = torch.device(device)
        sources = tuple(torch.from_numpy(array) for array in arrays)
        if target.type != "cuda":
            return tuple(source.to(device=target, dtype=torch.float32) for source in sources)
        pinned = tuple(
            torch.empty(
                source.shape,
                device="cpu",
                dtype=torch.float32,
                pin_memory=True,
            )
            for source in sources
        )
        for destination, source in zip(pinned, sources, strict=True):
            destination.copy_(source)
        return tuple(
            source.to(device=target, dtype=torch.float32, non_blocking=True)
            for source in pinned
        )
