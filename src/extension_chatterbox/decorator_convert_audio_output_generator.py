import functools
import io
import wave
import numpy as np
import functools


def numpy_to_wav_bytes(audio_data, sample_rate):
    """Convert numpy array to WAV format bytes"""
    # Ensure audio_data is in the right format
    if audio_data.dtype != np.int16:
        # Convert from float [-1, 1] to int16
        audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    # Get the bytes
    buffer.seek(0)
    return buffer.getvalue()


def decorator_convert_audio_output_generator(func):
    """Final decorator to convert audio_out from tuple to bytes before returning to caller"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for chunk in func(*args, **kwargs):
            if "audio_out" in chunk:
                # Convert the audio_out from (sample_rate, numpy_array) to bytes
                sample_rate, audio_data = chunk["audio_out"]
                audio_bytes = numpy_to_wav_bytes(audio_data, sample_rate)
                # chunk = {**chunk, "audio_out": audio_bytes}
                chunk["audio_out"] = audio_bytes
            yield chunk

    return wrapper
