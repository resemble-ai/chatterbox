"""
Emotion presets for Chatterbox TTS.
Provides mappings from human-readable emotion names to generation parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import torch
import torchaudio

@dataclass
class EmotionPreset:
    """Dataclass representing a set of generation parameters for a specific emotion."""
    name: str
    exaggeration: float
    temperature: float
    cfg_weight: float
    speaking_rate: float
    description: str

EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    'neutral': EmotionPreset('neutral', 0.3, 0.7, 0.5, 1.0, 'Default neutral speaking style.'),
    'happy': EmotionPreset('happy', 0.7, 0.9, 0.4, 1.05, 'Upbeat and positive tone.'),
    'sad': EmotionPreset('sad', 0.6, 0.6, 0.6, 0.9, 'Slower, subdued, and melancholic.'),
    'angry': EmotionPreset('angry', 0.9, 0.95, 0.3, 1.1, 'Loud, fast, and aggressive.'),
    'whisper': EmotionPreset('whisper', 0.2, 0.5, 0.7, 0.85, 'Quiet, close, and secretive.'),
    'excited': EmotionPreset('excited', 0.85, 0.95, 0.3, 1.15, 'Very fast and enthusiastic.'),
    'calm': EmotionPreset('calm', 0.15, 0.6, 0.6, 0.9, 'Relaxed, slow, and soothing.'),
    'narrator': EmotionPreset('narrator', 0.4, 0.7, 0.5, 0.95, 'Clear, steady, and engaging for storytelling.')
}

def get_emotion_params(emotion: str) -> Dict[str, Any]:
    """
    Returns generation parameters for a given emotion.
    
    Args:
        emotion: The name of the emotion preset.
        
    Returns:
        A dictionary containing kwargs for model.generate().
    """
    preset = EMOTION_PRESETS.get(emotion.lower())
    if not preset:
        raise ValueError(f"Unknown emotion preset: {emotion}")
        
    return {
        'exaggeration': preset.exaggeration,
        'temperature': preset.temperature,
        'cfg_weight': preset.cfg_weight
    }

def list_emotions() -> List[str]:
    """
    Returns a list of all available emotion presets.
    
    Returns:
        List of emotion preset names.
    """
    return list(EMOTION_PRESETS.keys())

def apply_speaking_rate(wav: torch.Tensor, rate: float, sample_rate: int) -> torch.Tensor:
    """
    Applies a speaking rate change to the given audio tensor.
    
    Args:
        wav: The input audio tensor of shape (channels, length) or (length,).
        rate: The speaking rate multiplier (1.0 = normal, > 1.0 = faster, < 1.0 = slower).
        sample_rate: The sample rate of the audio.
        
    Returns:
        The modified audio tensor.
    """
    if rate == 1.0:
        return wav
        
    # Using torchaudio's resample as a simple rate changer (this also changes pitch)
    # To speed up (rate > 1.0), we pretend original sample rate was lower, so we resample from
    # orig_sr * rate to orig_sr.
    new_freq = int(sample_rate * rate)
    out_wav = torchaudio.functional.resample(wav, orig_freq=new_freq, new_freq=sample_rate)
    return out_wav
