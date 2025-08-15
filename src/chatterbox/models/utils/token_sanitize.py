# src/chatterbox/models/utils/token_sanitize.py
from typing import Optional
import torch

@torch.inference_mode()
def sanitize_to_s3gen_range(
    x: torch.Tensor,
    *,
    start_id: int,          # T3Config.start_speech_token
    stop_id: int,           # T3Config.stop_speech_token
    pad_id: Optional[int],  # T3Config.pad_speech_token
    base_size: int          # S3 tokenizer content vocab size (no specials)
) -> torch.Tensor:
    """
    Prepare a speech-token sequence for S3Gen.
    - Drop one leading BOS if present
    - Truncate at first EOS (exclusive)
    - Remove PAD anywhere
    - Keep only ids in [0, base_size)

    Accepts (T,) or (1, T); returns (T,) on the same device.
    """
    # Flatten optional batch
    if x.dim() == 2:
        x = x[0]
    x = x.view(-1)

    # 1) Drop a single leading BOS if present
    if x.numel() and int(x[0].item()) == int(start_id):
        x = x[1:]

    # 2) Truncate at first EOS (exclusive)
    eos_pos = (x == stop_id).nonzero(as_tuple=True)[0]
    if eos_pos.numel():
        x = x[: int(eos_pos[0].item())]

    # 3) Remove PAD anywhere; defensive
    if pad_id is not None:
        x = x[x != pad_id]

    # 4) Final guard: keep only content ids
    in_range = (x >= 0) & (x < base_size)
    return x[in_range]

