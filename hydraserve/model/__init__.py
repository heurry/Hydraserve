"""HydraServe model adapters."""
from hydraserve.model.adapter import ModelAdapter
from hydraserve.model.qwen3_5 import Qwen3_5Adapter
from hydraserve.model.qwen3_6 import Qwen3_6Adapter

__all__ = ["ModelAdapter", "Qwen3_5Adapter", "Qwen3_6Adapter"]


def create_adapter(model_name: str, model_path: str, device, precision: str = "int4") -> ModelAdapter:
    """Factory function to create the appropriate adapter for a model."""
    if "Qwen3.6" in model_name or "27B" in model_name:
        return Qwen3_6Adapter(model_path, device, precision)
    elif "Qwen3.5" in model_name:
        return Qwen3_5Adapter(model_path, device, precision)
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Supported: Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-27B")
