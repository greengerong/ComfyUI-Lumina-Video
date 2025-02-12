from .model_loader import LuminaVideoModelLoader
from .sampler import LuminaVideoSampler
from .vae_decode import LuminaVideoVAEDecode

NODE_CLASS_MAPPINGS = {
    "LuminaVideoModelLoader": LuminaVideoModelLoader,
    "LuminaVideoSampler": LuminaVideoSampler,
    "LuminaVideoVAEDecode": LuminaVideoVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaVideoModelLoader": "Lumina Video Model Loader",
    "LuminaVideoSampler": "Lumina Video Sampler",
    "LuminaVideoVAEDecode": "Lumina Video VAE Decode",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
