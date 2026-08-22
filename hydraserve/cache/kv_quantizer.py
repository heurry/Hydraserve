"""Portable grouped symmetric INT4/INT8 codecs for KV transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Int4Tensor:
    packed: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    group_size: int
    original_dtype: str

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes


@dataclass(frozen=True, slots=True)
class Int8Tensor:
    quantized: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    group_size: int
    original_dtype: str

    @property
    def nbytes(self) -> int:
        return self.quantized.nbytes + self.scales.nbytes


def quantize_int4(tensor: np.ndarray, group_size: int = 64) -> Int4Tensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    source = np.asarray(tensor)
    if not np.issubdtype(source.dtype, np.floating):
        raise TypeError("INT4 quantization requires a floating-point tensor")
    flat = source.astype(np.float32, copy=False).reshape(-1)
    padded_size = ((flat.size + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat.size] = flat
    groups = padded.reshape(-1, group_size)
    maxima = np.max(np.abs(groups), axis=1)
    scales = np.where(maxima == 0, 1.0, maxima / 7.0).astype(np.float32)
    quantized = np.clip(np.rint(groups / scales[:, None]), -7, 7).astype(np.int8)
    unsigned = (quantized.reshape(-1).astype(np.int16) + 8).astype(np.uint8)
    if unsigned.size % 2:
        unsigned = np.pad(unsigned, (0, 1), constant_values=8)
    packed = (unsigned[0::2] | (unsigned[1::2] << 4)).astype(np.uint8)
    return Int4Tensor(packed, scales, source.shape, group_size, str(source.dtype))


def dequantize_int4(tensor: Int4Tensor) -> np.ndarray:
    low = (tensor.packed & 0x0F).astype(np.int16) - 8
    high = ((tensor.packed >> 4) & 0x0F).astype(np.int16) - 8
    values = np.empty(tensor.packed.size * 2, dtype=np.float32)
    values[0::2] = low
    values[1::2] = high
    padded_size = tensor.scales.size * tensor.group_size
    restored = values[:padded_size].reshape(-1, tensor.group_size) * tensor.scales[:, None]
    logical_size = int(np.prod(tensor.shape, dtype=np.int64))
    return restored.reshape(-1)[:logical_size].reshape(tensor.shape)


def quantize_int8(tensor: np.ndarray, group_size: int = 64) -> Int8Tensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    source = np.asarray(tensor)
    if not np.issubdtype(source.dtype, np.floating):
        raise TypeError("INT8 quantization requires a floating-point tensor")
    flat = source.astype(np.float32, copy=False).reshape(-1)
    padded_size = ((flat.size + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat.size] = flat
    groups = padded.reshape(-1, group_size)
    maxima = np.max(np.abs(groups), axis=1)
    scales = np.where(maxima == 0, 1.0, maxima / 127.0).astype(np.float32)
    quantized = np.clip(np.rint(groups / scales[:, None]), -127, 127).astype(np.int8)
    return Int8Tensor(quantized.reshape(-1), scales, source.shape, group_size, str(source.dtype))


def dequantize_int8(tensor: Int8Tensor) -> np.ndarray:
    restored = (
        tensor.quantized.astype(np.float32).reshape(-1, tensor.group_size)
        * tensor.scales[:, None]
    )
    logical_size = int(np.prod(tensor.shape, dtype=np.int64))
    return restored.reshape(-1)[:logical_size].reshape(tensor.shape)


def quantize_int8_torch(tensor, group_size: int = 64) -> Int8Tensor:
    """Quantize on the source GPU and stage only INT8 values plus scales."""
    import torch
    import torch.nn.functional as functional

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    source = tensor.detach()
    flat = source.float().reshape(-1)
    padded_size = ((flat.numel() + group_size - 1) // group_size) * group_size
    if padded_size != flat.numel():
        flat = functional.pad(flat, (0, padded_size - flat.numel()))
    groups = flat.reshape(-1, group_size)
    maxima = groups.abs().amax(dim=1)
    scales = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 127.0)
    quantized = torch.round(groups / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    if source.is_cuda:
        q_host = torch.empty(quantized.shape, dtype=torch.int8, pin_memory=True)
        s_host = torch.empty(scales.shape, dtype=torch.float32, pin_memory=True)
        q_host.copy_(quantized, non_blocking=True)
        s_host.copy_(scales, non_blocking=True)
        torch.cuda.current_stream(source.device).synchronize()
    else:
        q_host = quantized.cpu()
        s_host = scales.cpu()
    return Int8Tensor(
        q_host.numpy().reshape(-1),
        s_host.numpy(),
        tuple(source.shape),
        group_size,
        str(source.dtype).removeprefix("torch."),
    )


def dequantize_int8_torch(tensor: Int8Tensor, *, device):
    """Move compressed payload first, then dequantize on the destination GPU."""
    import torch

    target = torch.device(device)
    quantized = torch.from_numpy(tensor.quantized).to(target, non_blocking=True)
    scales = torch.from_numpy(tensor.scales).to(target, non_blocking=True)
    restored = quantized.float().reshape(-1, tensor.group_size) * scales[:, None]
    logical_size = int(np.prod(tensor.shape, dtype=np.int64))
    return restored.reshape(-1)[:logical_size].reshape(tensor.shape)
