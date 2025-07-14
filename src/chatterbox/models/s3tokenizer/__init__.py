import torch
from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens_cuda_sync(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0

    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None

    x = x[s: e]
    return x

# Pre-compute constants to avoid syncs
_ZERO_TENSOR = torch.tensor(0)
_LENGTH_TENSOR = torch.tensor(1000 + 1)  # max_new_tokens

def drop_invalid_tokens(x):
    """Returns only tokens between SOS and EOS using a mask. No syncs."""
    assert x.dim() == 1 or (x.dim() == 2 and x.size(0) == 1)
    x = x.squeeze(0)

    length = x.size(0)
    idx = torch.arange(length, device=x.device)
    
    # Create length tensor on the same device
    length_tensor = torch.full((), length, device=x.device)

    # SOS and EOS locations with fallback values
    sos_idx = torch.where(x == SOS, idx, torch.full_like(idx, length))
    eos_idx = torch.where(x == EOS, idx, torch.full_like(idx, length))

    # First match positions
    sos_pos = sos_idx.min()
    eos_pos = eos_idx.min()

    # Handle fallback cases to match original logic:
    # - No SOS: start from beginning (index 0, not after sos_pos)
    # - No EOS: go to end (no upper bound)
    
    # Create device-appropriate tensors
    zero_tensor = torch.zeros((), device=x.device, dtype=sos_pos.dtype)
    
    # Start position: if no SOS found, start from 0, otherwise start after SOS
    start_pos = torch.where(sos_pos == length, zero_tensor, sos_pos + 1)
    
    # End position: if no EOS found, use length, otherwise use EOS position
    end_pos = torch.where(eos_pos == length, length_tensor, eos_pos)

    # Use tensor indexing with arange to avoid both boolean masking and .item()
    # Clamp start_pos and end_pos to valid ranges
    start_pos = torch.clamp(start_pos, 0, length)
    end_pos = torch.clamp(end_pos, 0, length)
    
    # Create indices tensor for the valid range
    valid_length = torch.clamp(end_pos - start_pos, 0, length)
    indices = torch.arange(valid_length, device=x.device) + start_pos
    
    # Use advanced indexing to select elements
    return x[indices]