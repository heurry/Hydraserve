from __future__ import annotations

from hydraserve.model.runtime import QwenTextRuntime


class QwenHybridAdapter(QwenTextRuntime):
    """Backward-compatible public name for the production Qwen runtime.

    New code may use :class:`QwenTextRuntime` directly. This subclass deliberately
    contains no second execution implementation, so both names use the same
    clean-room CUDA/Triton runtime and checkpoint loader.
    """
