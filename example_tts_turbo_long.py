"""Generate long narration from a UTF-8 text file into one local WAV file."""
import argparse
import logging
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text_file', type=Path)
    parser.add_argument('--reference', type=Path, help='Reference WAV longer than 5 seconds')
    parser.add_argument('--output', type=Path, default=Path('narration.wav'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-chunk-chars', type=int, default=250)
    parser.add_argument('--max-gen-len', type=int, default=2000)
    parser.add_argument('--pause-ms', type=float, default=150)
    args = parser.parse_args()
    text = args.text_file.read_text(encoding='utf-8')
    logging.basicConfig(level=logging.INFO)
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)
    wav = model.generate_long(
        text,
        audio_prompt_path=str(args.reference) if args.reference else None,
        max_chunk_chars=args.max_chunk_chars,
        max_gen_len=args.max_gen_len,
        chunk_pause_ms=args.pause_ms,
    )
    ta.save(str(args.output), wav, model.sr)


if __name__ == '__main__':
    main()
