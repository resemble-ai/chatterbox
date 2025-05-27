import numpy as np


def trim_silence(wav, speech_timestamps, sr):
    """TODO: fading"""
    if len(speech_timestamps) == 0:
        return wav  # WARNING: no speech detected, returning original wav
    segs = []
    for segment in speech_timestamps:
        start_s, end_s = segment['start'], segment['end']
        start = int(start_s * sr)
        end = int(end_s * sr)
        seg = wav[start: end]
        segs.append(seg)
    return np.concatenate(segs)
