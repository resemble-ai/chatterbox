import numpy as np
from pydub import AudioSegment


SHORT = 2 ** 15


def np2audseg(audio_np, sr):
    """Convert a NumPy array to a Pydub AudioSegment."""
    if audio_np.ndim == 2:  # Stereo
        samples = np.int16(audio_np.T * SHORT)
        raw_audio = samples.T.tobytes()
        channels = 2
    else:  # Mono
        samples = np.int16(audio_np * SHORT)
        raw_audio = samples.tobytes()
        channels = 1
    audio_segment = AudioSegment(
        data=raw_audio,
        sample_width=2,
        frame_rate=sr,
        channels=channels
    )
    return audio_segment


def audseg2np(audio_segment):
    """Convert a Pydub AudioSegment back to NumPy array."""
    samples = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32) / SHORT


def change_audio_pace(audio_seg, speed=1.0):
    """Change pace using Pydub speedup (alters pitch as well)."""
    return audio_seg.speedup(speed)


def resample_audio(audio_seg, new_sr):
    """Resample audio to a new sample rate."""
    return audio_seg.set_frame_rate(new_sr)


def adjust_pace(wav_np, orig_sr, target_sr=None, target_speed=1):
    audio = np2audseg(wav_np, orig_sr)

    paced_audio = change_audio_pace(audio, speed=target_speed)

    if target_sr is None or target_sr == orig_sr:
        resampled_audio = paced_audio
    else:
        resampled_audio = resample_audio(paced_audio, new_sr=target_sr)

    final_audio_np = audseg2np(resampled_audio)
    return final_audio_np
