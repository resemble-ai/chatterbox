try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-mlx")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .models import (
    DEBUG_LOGGING,
    is_debug,
    set_mlx_cache_limit,
    set_mlx_memory_limit,
)

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "ChatterboxMultilingualTTS",
    "SUPPORTED_LANGUAGES",
    "DEBUG_LOGGING",
    "is_debug",
    "set_mlx_cache_limit",
    "set_mlx_memory_limit",
]
