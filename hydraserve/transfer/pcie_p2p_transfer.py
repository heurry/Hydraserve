"""
PCIe P2P Transfer Module.

Re-exports PCIeP2PBackend from the unified backend module.
Enables PD separation without NVLink via INT4 KV quantization.
"""

from hydraserve.transfer.backend import PCIeP2PBackend

__all__ = ["PCIeP2PBackend"]
