# Copyright (c) 2026 Wonderful AI
# MIT License
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .streaming import ChatterboxStreamer, AudioChunk, StreamingMetrics