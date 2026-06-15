from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


@dataclass
class StreamingAudioChunk:
    """A chunk of streamed mono audio.

    Audio is always float32 on CPU with shape ``[1, samples]``. Chatterbox
    streaming chunks are not Perth-watermarked; use ``generate`` when the final
    full waveform must include a watermark.
    """

    audio: torch.Tensor
    sample_rate: int
    index: int
    is_final: bool
    start_sample: int
    end_sample: int
    generated_tokens: int
    watermarked: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


def audio_to_pcm_s16le(audio: torch.Tensor) -> bytes:
    """Convert a mono float audio tensor to raw little-endian signed 16-bit PCM."""
    audio_np = audio.detach().cpu().reshape(-1).numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype("<i2", copy=False).tobytes()


def chunks_to_pcm_s16le(chunks: Iterable[StreamingAudioChunk]) -> Iterator[bytes]:
    """Yield raw little-endian signed 16-bit PCM bytes for streamed chunks."""
    for chunk in chunks:
        yield audio_to_pcm_s16le(chunk.audio)


def write_chunks_to_wav(path: str | Path, chunks: Iterable[StreamingAudioChunk]) -> Path:
    """Write streamed chunks to a mono 16-bit PCM WAV file."""
    path = Path(path)
    iterator = iter(chunks)

    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot write WAV from an empty chunk stream") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(first.sample_rate)
        wav_file.writeframes(audio_to_pcm_s16le(first.audio))

        for chunk in iterator:
            if chunk.sample_rate != first.sample_rate:
                raise ValueError(
                    f"stream sample rate changed from {first.sample_rate} to {chunk.sample_rate}"
                )
            wav_file.writeframes(audio_to_pcm_s16le(chunk.audio))

    return path
