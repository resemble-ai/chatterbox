from typing import Any
import torch
import numpy as np
from tts_webui.utils.outputs.path import get_relative_output_path_ext
from tts_webui.utils.pack_metadata import pack_metadata


def save_npz_musicgen(filename: str, tokens: torch.Tensor, metadata: dict[str, Any]):
    np.savez(
        filename,
        **{
            "tokens": tokens.cpu().numpy(),
            "metadata": pack_metadata(metadata),
        },
    )


def decorator_save_musicgen_npz(fn):
    def wrapper(*args, **kwargs):
        result_dict = fn(*args, **kwargs)
        tokens = result_dict["tokens"]

        if tokens is not None:
            path = get_relative_output_path_ext(result_dict, ".npz")

            save_npz_musicgen(path, tokens, result_dict["metadata"])

        return result_dict

    return wrapper
