"""
Intra-GPU (MPS) Transfer Module.

Re-exports IntraGPUBackend from the unified backend module.
Provides BulletServe-inspired same-GPU PD separation with zero-copy.
"""

from hydraserve.transfer.backend import IntraGPUBackend

__all__ = ["IntraGPUBackend"]
