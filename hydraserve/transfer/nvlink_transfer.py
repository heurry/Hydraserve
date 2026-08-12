"""
NVLink Transfer Module.

Thin wrapper re-exporting NVLinkBackend from the unified backend module.
The full implementation is in hydraserve.transfer.backend.
"""

from hydraserve.transfer.backend import NVLinkBackend

__all__ = ["NVLinkBackend"]
