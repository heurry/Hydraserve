"""Portable grouped symmetric INT4 codec for KV transfer."""

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
