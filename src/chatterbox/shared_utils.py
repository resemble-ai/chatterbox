"""
Shared utilities for ChatterboxTTS and ChatterboxMultilingualTTS.

This module contains common functions that are used by both the English-only
and multilingual TTS implementations to reduce code duplication.
"""

import warnings
import re
from pathlib import Path
from typing import Union, Optional, List

import torch
import torchaudio
import torchaudio.functional as taF

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR


def check_mps_availability(device: str) -> str:
    """
    Check if MPS is available on macOS and fallback to CPU if not.
    
    Args:
        device: Requested device ("mps", "cuda", "cpu")
        
    Returns:
        Validated device string that can actually be used
    """
    if device == "mps" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
        return "cpu"
    return device


def get_map_location(device: str) -> Optional[torch.device]:
    """
    Get appropriate map_location for model loading based on device.
    
    Args:
        device: Target device string
        
    Returns:
        torch.device for CPU loading, or None for GPU loading
    """
    if device in ["cpu", "mps"]:
        return torch.device('cpu')
    return None


def validate_audio_file(wav_fpath: Union[str, Path]) -> Path:
    """
    Validate that an audio file exists and is accessible.
    
    Args:
        wav_fpath: Path to the audio file
        
    Returns:
        Validated Path object
        
    Raises:
        TypeError: If wav_fpath is not a string or Path
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the path is not a file
    """
    if not isinstance(wav_fpath, (str, Path)):
        raise TypeError("wav_fpath must be a string or Path object")
    
    wav_path = Path(wav_fpath)
    if not wav_path.exists():
        raise FileNotFoundError(f"Reference audio file not found: {wav_fpath}")
    
    if not wav_path.is_file():
        raise ValueError(f"Reference audio path is not a file: {wav_fpath}")
    
    return wav_path


def load_and_preprocess_audio(
    wav_fpath: Union[str, Path], 
    device: str,
    min_duration: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load, validate, and preprocess audio for TTS conditioning.
    
    Args:
        wav_fpath: Path to the audio file
        device: Target device for tensors
        min_duration: Minimum required audio duration in seconds
        
    Returns:
        Tuple of (s3gen_ref_wav, ref_16k_wav_tensor) - both preprocessed and on device
        
    Raises:
        RuntimeError: If audio loading or resampling fails
        ValueError: If audio is invalid or too short
    """
    # Load reference wav with enhanced error handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            s3gen_ref_wav, _sr = torchaudio.load(wav_fpath)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file '{wav_fpath}': {e}")
    
    # Validate audio data
    if s3gen_ref_wav.numel() == 0:
        raise ValueError("Audio file is empty or contains no valid audio data")
    
    if _sr <= 0:
        raise ValueError(f"Invalid sample rate: {_sr}")
    
    # Check audio duration
    duration = s3gen_ref_wav.shape[-1] / _sr
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum {min_duration}s required)")
    
    # Resample to S3GEN_SR if necessary
    if _sr != S3GEN_SR:
        try:
            s3gen_ref_wav = taF.resample(s3gen_ref_wav, _sr, S3GEN_SR)
        except Exception as e:
            raise RuntimeError(f"Failed to resample audio from {_sr}Hz to {S3GEN_SR}Hz: {e}")
    
    # Ensure we have mono audio (1D tensor)
    if s3gen_ref_wav.dim() > 1:
        s3gen_ref_wav = s3gen_ref_wav.mean(dim=0)  # Convert to mono by averaging channels
    
    # Move to device early to avoid unnecessary transfers
    s3gen_ref_wav = s3gen_ref_wav.to(device)
    
    # Resample to 16kHz directly from tensor
    try:
        ref_16k_wav_tensor = taF.resample(s3gen_ref_wav.unsqueeze(0), S3GEN_SR, S3_SR).squeeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to resample to 16kHz: {e}")
    
    return s3gen_ref_wav, ref_16k_wav_tensor


def drop_bad_tokens(tokens: torch.Tensor, threshold: int = 6561) -> torch.Tensor:
    """
    Efficiently filter out invalid tokens on GPU without CPU sync.
    
    Args:
        tokens: Input token tensor
        threshold: Tokens >= threshold are considered invalid
        
    Returns:
        Filtered token tensor with only valid tokens
    """
    mask = tokens < threshold
    result = torch.masked_select(tokens, mask)
    return result


def prepare_text_tokens(
    text_tokens: torch.Tensor, 
    sot: int, 
    eot: int, 
    cfg_weight: float
) -> torch.Tensor:
    """
    Add start/end tokens and duplicate for CFG if needed.
    
    Args:
        text_tokens: Input text tokens
        sot: Start of text token ID
        eot: End of text token ID
        cfg_weight: CFG weight (if > 0, tokens will be duplicated)
        
    Returns:
        Prepared text tokens with SOT/EOT and CFG duplication
    """
    import torch.nn.functional as F
    
    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    
    return text_tokens


def punc_norm(text: str, multilingual: bool = False) -> str:
    """
    Unified text normalization with optional multilingual support.
    
    Args:
        text: Input text to normalize
        multilingual: Whether to include multilingual sentence enders
        
    Returns:
        Normalized text with consistent punctuation
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    if multilingual:
        sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    else:
        sentence_enders = {".", "!", "?", "-", ","}
    
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def validate_exaggeration(exaggeration) -> float:
    """
    Validate and convert exaggeration parameter.
    
    Args:
        exaggeration: Exaggeration value to validate
        
    Returns:
        Validated float exaggeration value
        
    Raises:
        ValueError: If exaggeration is invalid or out of range
    """
    try:
        exaggeration = float(exaggeration)
        if not (0.0 <= exaggeration <= 2.0):
            raise ValueError("exaggeration must be between 0.0 and 2.0")
        return exaggeration
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid exaggeration value: {e}")


def check_exaggeration_update_needed(
    current_emotion_adv: torch.Tensor,
    new_exaggeration: float,
    device: str,
    atol: float = 1e-6
) -> tuple[bool, torch.Tensor]:
    """
    Check if exaggeration needs to be updated without CPU sync.
    
    Args:
        current_emotion_adv: Current emotion advancement tensor
        new_exaggeration: New exaggeration value
        device: Device for tensor operations
        atol: Absolute tolerance for comparison
        
    Returns:
        Tuple of (needs_update, new_emotion_tensor)
    """
    new_emotion_tensor = new_exaggeration * torch.ones(
        1, 1, 1, 
        device=device, 
        dtype=current_emotion_adv.dtype
    )
    
    needs_update = not torch.allclose(current_emotion_adv, new_emotion_tensor, atol=atol)
    return needs_update, new_emotion_tensor


def validate_text_input(text: str) -> str:
    """
    Validate text input for TTS generation.
    
    Args:
        text: Input text to validate
        
    Returns:
        Validated text string
        
    Raises:
        ValueError: If text is invalid
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    if text.strip() == "":
        raise ValueError("Text cannot be empty or whitespace only")
    
    return text


def validate_language_id(language_id: str, supported_languages: dict) -> str:
    """
    Validate language_id for multilingual TTS.
    
    Args:
        language_id: Language identifier to validate
        supported_languages: Dictionary of supported language codes
        
    Returns:
        Validated language_id string (lowercased)
        
    Raises:
        ValueError: If language_id is invalid
        TypeError: If language_id is not a string
    """
    if language_id is None:
        raise ValueError("language_id is required for multilingual TTS. Use one of: " + ", ".join(supported_languages.keys()))
    
    if not isinstance(language_id, str):
        raise TypeError(f"language_id must be a string, got {type(language_id)}")
    
    if language_id.lower() not in supported_languages:
        supported_langs = ", ".join(supported_languages.keys())
        raise ValueError(
            f"Unsupported language_id '{language_id}'. "
            f"Supported languages: {supported_langs}"
        )
    
    return language_id.lower()


def validate_float_parameter(value, name: str, min_val: float = None, max_val: float = None, allow_zero: bool = True) -> float:
    """
    Validate a float parameter with optional range checking.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_zero: Whether zero is allowed for positive checks
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If value is invalid or out of range
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {name} value: {e}")
    
    # Check minimum value
    if min_val is not None:
        if not allow_zero and value <= min_val:
            raise ValueError(f"{name} of {value} must be greater than {min_val}")
        elif allow_zero and value < min_val:
            raise ValueError(f"{name} of {value} must be >= {min_val}")

    # Check maximum value
    if max_val is not None:
        if value > max_val:
            raise ValueError(f"{name} of {value} must be <= {max_val}")
    
    # Special case for positive-only values
    if min_val is None and not allow_zero and value <= 0.0:
        raise ValueError(f"{name} of {value} must be positive")

    return value


def validate_audio_prompt_path(audio_prompt_path: Union[str, Path]) -> Path:
    """
    Validate audio prompt path and check file format.
    
    Args:
        audio_prompt_path: Path to audio file
        
    Returns:
        Validated Path object
        
    Raises:
        TypeError: If path is not string or Path
        FileNotFoundError: If file doesn't exist
        ValueError: If path is not a file or has unsupported format
    """
    if not isinstance(audio_prompt_path, (str, Path)):
        raise TypeError("audio_prompt_path must be a string or Path object")
    
    audio_path = Path(audio_prompt_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt_path}")
    
    if not audio_path.is_file():
        raise ValueError(f"Audio prompt path is not a file: {audio_prompt_path}")
    
    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    if audio_path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Supported: {', '.join(valid_extensions)}")
    
    return audio_path


def estimate_token_count(text: str, tokenizer, language_id: str = None) -> int:
    """
    Estimate the number of tokens for a given text.
    
    Args:
        text: Input text to estimate
        tokenizer: Tokenizer instance (EnTokenizer or MTLTokenizer)
        language_id: Language ID for multilingual tokenizer (optional)
        
    Returns:
        Estimated number of tokens
    """
    try:
        if hasattr(tokenizer, 'encode'):
            # Use encode method for more efficient token counting
            if language_id:
                tokens = tokenizer.encode(text, language_id=language_id)
            else:
                tokens = tokenizer.encode(text)
            return len(tokens)
        else:
            # Fallback to text_to_tokens
            if language_id:
                text_tokens = tokenizer.text_to_tokens(text, language_id=language_id)
            else:
                text_tokens = tokenizer.text_to_tokens(text)
            return text_tokens.shape[-1]  # Last dimension is the token count
    except Exception:
        # Fallback estimation: roughly 0.75 tokens per word
        words = len(text.split())
        return int(words * 0.75)


def smart_text_splitter(text: str, max_tokens: int = 220, tokenizer=None, language_id: str = None) -> List[str]:
    """
    Split text into chunks that don't exceed the maximum token count.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk (default 300, cache-aware)
        tokenizer: Tokenizer instance for accurate token counting (optional)
        language_id: Language ID for multilingual tokenizer (optional)
        
    Returns:
        List of text chunks
    """
    # First check if text is already under the limit
    if tokenizer:
        total_tokens = estimate_token_count(text, tokenizer, language_id)
        if total_tokens <= max_tokens:
            return [text]
    
    # Define the sentence splitting pattern
    # This pattern splits on sentence endings (.!?) but avoids splitting on:
    # - Abbreviations with periods followed by lowercase
    # - Numbers with periods
    # - Quoted text
    # - Exclamation marks in contractions
    sentence_pattern = r'(?<=[.?!])\s*(?![.\w"\'\d]|[,!]|\*)'
    
    # Split into sentences
    sentences = re.split(sentence_pattern, text.strip())
    if not sentences:
        return [text]
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Try adding this sentence to the current chunk
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        # Check if the test chunk would exceed token limit
        if tokenizer:
            token_count = estimate_token_count(test_chunk, tokenizer, language_id)
            would_exceed = token_count > max_tokens
        else:
            # Fallback: estimate roughly 0.75 tokens per word
            word_count = len(test_chunk.split())
            would_exceed = word_count * 0.75 > max_tokens
        
        if would_exceed and current_chunk:
            # Save current chunk and start new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk = test_chunk
    
    # Add the final chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle edge case where a single sentence is too long
    final_chunks = []
    for chunk in chunks:
        if tokenizer:
            token_count = estimate_token_count(chunk, tokenizer, language_id)
            if token_count > max_tokens:
                # Split long sentence by words as fallback
                words = chunk.split()
                temp_chunk = ""
                for word in words:
                    test_temp = temp_chunk + (" " if temp_chunk else "") + word
                    if tokenizer:
                        test_tokens = estimate_token_count(test_temp, tokenizer, language_id)
                        if test_tokens > max_tokens and temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk = test_temp
                    else:
                        if len(test_temp.split()) * 0.75 > max_tokens and temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk = test_temp
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [text]


_silence_cache = {}

def _get_cached_silence_tensor(silence_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Get or create a cached silence tensor.
    
    Args:
        silence_samples: Number of silence samples
        device: Target device
        dtype: Target dtype
        
    Returns:
        Silence tensor on the specified device and dtype
    """
    cache_key = (silence_samples, str(device), str(dtype))
    
    if cache_key not in _silence_cache:
        _silence_cache[cache_key] = torch.zeros(silence_samples, device=device, dtype=dtype)
    
    return _silence_cache[cache_key]


def concatenate_audio_tensors(audio_tensors: List[torch.Tensor], silence_duration: float = 0.1, sample_rate: int = S3GEN_SR) -> torch.Tensor:
    """
    Concatenate multiple audio tensors with optional silence padding.
    
    Args:
        audio_tensors: List of audio tensors to concatenate
        silence_duration: Duration of silence to add between chunks (in seconds)
        sample_rate: Sample rate of the audio tensors
        
    Returns:
        Concatenated audio tensor
    """
    if not audio_tensors:
        raise ValueError("No audio tensors provided")
    
    if len(audio_tensors) == 1:
        return audio_tensors[0]
    
    # Process tensors and build result list with silence in single loop
    result_tensors = []
    silence = None
    
    for i, audio in enumerate(audio_tensors):
        if audio.dim() == 0:
            # Scalar tensor - convert to 1D
            audio = audio.unsqueeze(0)
        elif audio.dim() > 1:
            # Multi-dimensional - squeeze out batch dimensions to get 1D audio
            audio = audio.squeeze()
            # If still multi-dimensional after squeeze, flatten
            if audio.dim() > 1:
                audio = audio.flatten()
        
        # Create cached silence tensor on first iteration using first audio tensor's properties
        if silence is None:
            silence_samples = int(silence_duration * sample_rate)
            silence = _get_cached_silence_tensor(silence_samples, audio.device, audio.dtype)
        
        # Add silence before audio (except for first audio)
        if i > 0:
            result_tensors.append(silence)
        result_tensors.append(audio)
    
    concatenated = torch.cat(result_tensors, dim=0)
    
    # Check dimensionality before cleanup
    should_unsqueeze = audio_tensors[0].dim() > 1
    
    # Clean up intermediate tensors to free memory
    del audio_tensors
    del result_tensors
    
    # Return with the same dimensionality as the original audio tensors
    if should_unsqueeze:
        # If original was 2D, return as 2D with batch dimension
        return concatenated.unsqueeze(0)
    else:
        return concatenated
