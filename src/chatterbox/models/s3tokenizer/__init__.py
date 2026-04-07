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


def _drop_invalid_tokens_cuda_sync(x):
    """Drop SoS and EoS (original implementation — causes CUDA sync via .nonzero()). Internal use only."""
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


def drop_invalid_tokens(x):
    """Drop SoS and EoS using only tensor ops — no CUDA syncs."""
    assert x.dim() == 1 or (x.dim() == 2 and x.size(0) == 1), "only batch size of one allowed for now"
    x = x.squeeze(0)

    length = x.size(0)
    idx = torch.arange(length, device=x.device)
    length_tensor = torch.full((), length, device=x.device)

    # Find SOS/EOS positions without .nonzero() (would cause CUDA sync)
    sos_idx = torch.where(x == SOS, idx, torch.full_like(idx, length))
    eos_idx = torch.where(x == EOS, idx, torch.full_like(idx, length))
    sos_pos = sos_idx.min()
    eos_pos = eos_idx.min()

    zero_tensor = torch.zeros((), device=x.device, dtype=sos_pos.dtype)

    # No SOS -> start from 0; otherwise start after SOS
    start_pos = torch.where(sos_pos == length, zero_tensor, sos_pos + 1)
    # No EOS -> go to end; otherwise stop at EOS
    end_pos = torch.where(eos_pos == length, length_tensor, eos_pos)

    start_pos = torch.clamp(start_pos, 0, length)
    end_pos = torch.clamp(end_pos, 0, length)

    valid_length = torch.clamp(end_pos - start_pos, 0, length)
    indices = torch.arange(valid_length, device=x.device) + start_pos
    return x[indices]
