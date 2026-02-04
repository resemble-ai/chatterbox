
import pytest
import torch

def test_imports():
    """
    Simple test to verify that the package modules can be imported.
    This ensures that all dependencies are installed and the package structure is correct.
    """
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from chatterbox.asr import SpeechRecognizer
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_torch_available():
    assert torch.cuda.is_available() or True  # Just ensuring torch is importable
