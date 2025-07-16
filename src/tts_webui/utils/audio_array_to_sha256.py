import numpy as np
import hashlib


def audio_array_to_sha256(audio_array: np.ndarray) -> str:
    return hashlib.sha256(audio_array.tobytes()).hexdigest()
