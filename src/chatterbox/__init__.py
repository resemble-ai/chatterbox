try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Enhanced Toolkit Features
from .streaming import StreamingTTS, SentenceSplitter, stream_to_file
from .voice_bank import VoiceBank, VoiceProfile
from .text_normalizer import get_normalizer
from .lang_detect import LanguageDetector, detect_language
from .emotion_control import get_emotion_params, apply_speaking_rate
from .srt_narrator import SRTNarrator, parse_srt
from .batch_processor import BatchProcessor, TTSJob