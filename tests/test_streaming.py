import tempfile
import unittest
import wave
from pathlib import Path

import torch

from chatterbox.streaming import (
    StreamingAudioChunk,
    audio_to_pcm_s16le,
    chunks_to_pcm_s16le,
    write_chunks_to_wav,
)


class StreamingUtilsTest(unittest.TestCase):
    def test_audio_to_pcm_s16le_clamps_and_converts(self):
        audio = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], dtype=torch.float32)

        pcm = audio_to_pcm_s16le(audio)

        self.assertEqual(len(pcm), 10)
        self.assertEqual(pcm.hex(), "0000ff7f0180ff7f0180")

    def test_chunks_to_pcm_s16le(self):
        chunks = [
            StreamingAudioChunk(
                audio=torch.tensor([[0.0, 0.5]], dtype=torch.float32),
                sample_rate=24000,
                index=0,
                is_final=False,
                start_sample=0,
                end_sample=2,
                generated_tokens=24,
            ),
            StreamingAudioChunk(
                audio=torch.tensor([[-0.5]], dtype=torch.float32),
                sample_rate=24000,
                index=1,
                is_final=True,
                start_sample=2,
                end_sample=3,
                generated_tokens=30,
            ),
        ]

        pcm = b"".join(chunks_to_pcm_s16le(chunks))

        self.assertEqual(len(pcm), 6)

    def test_write_chunks_to_wav(self):
        chunks = [
            StreamingAudioChunk(
                audio=torch.zeros(1, 8),
                sample_rate=24000,
                index=0,
                is_final=False,
                start_sample=0,
                end_sample=8,
                generated_tokens=24,
            ),
            StreamingAudioChunk(
                audio=torch.ones(1, 4) * 0.25,
                sample_rate=24000,
                index=1,
                is_final=True,
                start_sample=8,
                end_sample=12,
                generated_tokens=30,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_chunks_to_wav(Path(tmpdir) / "stream.wav", chunks)
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 24000)
                self.assertEqual(wav_file.getnframes(), 12)


if __name__ == "__main__":
    unittest.main()
