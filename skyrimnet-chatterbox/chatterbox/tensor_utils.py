"""
Tensor operation utilities for Chatterbox TTS models.

This module provides utility functions to reduce code duplication
for common tensor operations across the codebase.
"""
import torch
from pathlib import Path
from typing import Optional, Union
from safetensors.torch import load_file


def safe_model_to_dtype(model_component, dtype: torch.dtype) -> None:
    """
    Safely move a model component to the specified dtype with null checks.
    
    Args:
        model_component: The model component to move (can be None)
        dtype: Target torch.dtype
    """
    if model_component is not None:
        model_component.to(dtype=dtype)


def safe_conditional_to_dtype(model, dtype: torch.dtype) -> None:
    """
    Safely move model conditionals to the specified dtype with comprehensive null checks.
    
    Args:
        model: The model containing conditionals
        dtype: Target torch.dtype
    """
    if hasattr(model, 'conds') and model.conds is not None:
        if hasattr(model.conds, 't3') and model.conds.t3 is not None:
            model.conds.t3.to(dtype=dtype)


def initialize_model_dtype(model, dtype: torch.dtype) -> None:
    """
    Initialize a Chatterbox model with proper dtype handling for all components.
    
    Args:
        model: ChatterboxTTS or ChatterboxMultilingualTTS instance
        dtype: Target torch.dtype
    """
    # Move main t3 model
    if hasattr(model, 't3') and model.t3 is not None:
        model.t3.to(dtype=dtype)
    
    # Move conditionals if they exist
    safe_conditional_to_dtype(model, dtype)


def safe_tensor_to_device_dtype(
    tensor: Optional[torch.Tensor], 
    device: torch.device, 
    dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    """
    Safely move a tensor to device and optionally change dtype.
    
    Args:
        tensor: Input tensor (can be None)
        device: Target device
        dtype: Optional target dtype
        
    Returns:
        Moved tensor or None if input was None
    """
    if tensor is None:
        return None
    
    if dtype is not None:
        return tensor.to(device=device, dtype=dtype)
    else:
        return tensor.to(device=device)


def setup_s3gen_dtypes(model, dtype: torch.dtype) -> None:
    """
    Setup S3Gen model component dtypes with proper float32 constraints.
    
    Args:
        model: Model containing s3gen component
        dtype: Target dtype for main components
    """
    if not hasattr(model, 's3gen') or model.s3gen is None:
        return
    
    # Handle flow fp16 flag
    if dtype == torch.float16:
        model.s3gen.flow.fp16 = True
    elif dtype == torch.float32:
        model.s3gen.flow.fp16 = False
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    
    # Main s3gen to specified dtype
    model.s3gen.to(dtype=dtype)
    
    # These components must stay float32 due to CUDA limitations
    # due to "Error: cuFFT doesn't support tensor of type: BFloat16" from torch.stft
    # and other errors and general instability
    model.s3gen.mel2wav.to(dtype=torch.float32)
    model.s3gen.tokenizer.to(dtype=torch.float32) 
    model.s3gen.speaker_encoder.to(dtype=torch.float32)


def create_t3_conditional_safe(
    model, 
    speaker_emb: torch.Tensor, 
    cond_prompt_speech_tokens: torch.Tensor, 
    emotion_adv: torch.Tensor
) -> None:
    """
    Safely create T3 conditional with proper device/dtype handling.
    
    Args:
        model: Model containing the conditionals
        speaker_emb: Speaker embedding tensor
        cond_prompt_speech_tokens: Conditional prompt speech tokens
        emotion_adv: Emotion advancement tensor
    """
    if not hasattr(model, 'conds') or model.conds is None:
        return
        
    # Import here to avoid circular imports
    from .models.t3.modules.cond_enc import T3Cond
    
    # Get reference dtype from existing speaker embedding
    ref_dtype = model.conds.t3.speaker_emb.dtype if model.conds.t3.speaker_emb is not None else torch.float32
    
    model.conds.t3 = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_prompt_speech_tokens,
        emotion_adv=emotion_adv,
    ).to(device=model.device, dtype=ref_dtype)


def load_t3_state_dict_safe(model, state_dict_path: Path, device: torch.device) -> None:
    """
    Safely load T3 model state dict with backward compatibility support.
    
    Args:
        model: T3 model instance to load state into
        state_dict_path: Path to the state dict file
        device: Target device for the model
    """
    t3_state = load_file(state_dict_path)
    
    # Handle backward compatibility for old format
    if "model" in t3_state.keys():
        model_val = t3_state["model"]
        # Support both old (list) and new (direct) formats
        if isinstance(model_val, (list, tuple)):
            t3_state = model_val[0]
        else:
            t3_state = model_val
    
    model.load_state_dict(t3_state)
    model.to(device).eval()


def load_s3gen_safe(ckpt_dir: Path, device: torch.device, is_multilingual: bool = False):
    """
    Safely load S3Gen model with proper file format handling.
    
    Args:
        ckpt_dir: Checkpoint directory path
        device: Target device
        is_multilingual: Whether to load multilingual (.pt) or standard (.safetensors) format
        
    Returns:
        Loaded and configured S3Gen model
    """
    # Import here to avoid circular imports
    from .models.s3gen import S3Gen
    
    s3gen = S3Gen()
    
    if is_multilingual:
        # Multilingual uses .pt format
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
    else:
        # Standard uses .safetensors format
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
    
    s3gen.to(device).eval()
    return s3gen


def load_voice_encoder_safe(ckpt_dir: Path, device: torch.device, is_multilingual: bool = False):
    """
    Safely load voice encoder with proper file format handling.
    
    Args:
        ckpt_dir: Checkpoint directory path  
        device: Target device
        is_multilingual: Whether to load multilingual (.pt) or standard (.safetensors) format
        
    Returns:
        Loaded and configured VoiceEncoder
    """
    # Import here to avoid circular imports
    from .models.voice_encoder import VoiceEncoder
    
    ve = VoiceEncoder()
    
    if is_multilingual:
        # Multilingual uses .pt format
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
    else:
        # Standard uses .safetensors format  
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    
    ve.to(device).eval()
    return ve


def load_conditionals_safe(ckpt_dir: Path, device: torch.device, is_multilingual: bool = False):
    """
    Safely load conditionals if they exist.
    
    Args:
        ckpt_dir: Checkpoint directory path
        device: Target device
        is_multilingual: Whether this is multilingual (affects .to(device) call)
        
    Returns:
        Loaded conditionals or None if not found
    """
    # Import locally to avoid circular imports
    from .shared_utils import get_map_location
    
    map_location = get_map_location(device)
    builtin_voice = ckpt_dir / "conds.pt"
    
    if builtin_voice.exists():
        # Import the appropriate Conditionals class
        if is_multilingual:
            from .mtl_tts import Conditionals
        else:
            from .tts import Conditionals
            
        conds = Conditionals.load(builtin_voice, map_location=map_location)
        # Multilingual version calls .to(device) explicitly
        if is_multilingual:
            conds = conds.to(device)
        return conds
    
    return None