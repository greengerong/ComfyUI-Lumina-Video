from .model_loader import LuminaVideoModelLoader
from .sampler import LuminaVideoSampler

NODE_CLASS_MAPPINGS = {
    "LuminaVideoModelLoader": LuminaVideoModelLoader,
    "LuminaVideoSampler": LuminaVideoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaVideoModelLoader": "Lumina Video Model Loader",
    "LuminaVideoSampler": "Lumina Video Sampler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
