"""Bridge between model-runtime state dictionaries and transfer regions."""

from __future__ import annotations

from hydraserve.cache.state_pool import LinearState
from hydraserve.config import ModelConfig
from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer.descriptor import StateTransferDescriptor, TransferMode
from hydraserve.transfer.pipeline import HybridStateBundle
from hydraserve.cache.kv_quantizer import Int4Tensor, dequantize_int4


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

        layers = []
        for layer_index in model.full_attention_layer_indices:
            key, value = paged_cache.read(request_id, layer_index)
            layers.append(torch.stack((key, value), dim=0))
        stacked = torch.stack(layers, dim=0)
        if mode is TransferMode.QUANTIZED_TRANSFER:
            return stacked.float().cpu().numpy()
        return stacked.view(torch.uint16).cpu().numpy()

    @staticmethod
    def install_kv(model: ModelConfig, paged_cache, request_id: int, payload) -> None:
        import numpy as np
        import torch

        values = dequantize_int4(payload) if isinstance(payload, Int4Tensor) else np.asarray(payload)
        expected_prefix = (
            model.num_full_attention_layers,
            2,
        )
        if values.shape[:2] != expected_prefix or values.shape[3:] != (
            model.num_kv_heads,
            model.head_dim,
        ):
            raise ValueError(f"invalid transferred KV shape {values.shape}")
        positions = torch.arange(values.shape[2], device=paged_cache.device)
        # FULL transfer ships BF16 as uint16 raw bits; reinterpret, don't convert.
        bf16_bits = values.dtype == np.uint16
        for slot, layer_index in enumerate(model.full_attention_layer_indices):
            key = torch.from_numpy(values[slot, 0])
            value = torch.from_numpy(values[slot, 1])
            if bf16_bits:
                key = key.view(torch.bfloat16)
                value = value.view(torch.bfloat16)
            key = key.to(device=paged_cache.device, dtype=paged_cache.dtype)
            value = value.to(device=paged_cache.device, dtype=paged_cache.dtype)
            paged_cache.write(request_id, layer_index, positions, key, value)

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
        staged = tuple(
            torch.empty(
                tensor.shape,
                device="cpu",
                dtype=tensor.dtype,
                pin_memory=True,
            )
            for tensor in tensors
        )
        for destination, source in zip(staged, tensors, strict=True):
            destination.copy_(source, non_blocking=True)
        torch.cuda.current_stream(tensors[0].device).synchronize()
        return staged

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
