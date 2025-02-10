from .nodes import LuminaVideoGenerator

NODE_CLASS_MAPPINGS = {
    "LuminaVideoGenerator": LuminaVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminaVideoGenerator": "Lumina Video Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
