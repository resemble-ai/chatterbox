"""Chatterbox inference library and model preservation tools."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chatterbox-tts")
except PackageNotFoundError:
    __version__ = "0+unknown"

_EXPORTS = {
    "ChatterboxTTS": ".tts",
    "ChatterboxVC": ".vc",
    "ChatterboxMultilingualTTS": ".mtl_tts",
    "ChatterboxTurboTTS": ".tts_turbo",
    "SUPPORTED_LANGUAGES": ".mtl_tts",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # Archiving and inspecting package metadata must not initialize the ML stack.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
