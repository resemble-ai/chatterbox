"""Inspect Turbo's own alignment signal without running a transcription model.

python example_turbo_alignment.py --text 'First a boom, then another boom.'
The output is diagnostic attention, NOT calibrated word timestamps.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chatterbox.models.t3.inference.turbo_alignment import TurboAlignmentCapture
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--audio-prompt', default=None)
    parser.add_argument('--output', type=Path, default=Path('turbo-alignment'))
    parser.add_argument('--heads', default='4:6,5:14,8:4', help='Exploratory layer:head pairs, not calibrated boundaries')
    args = parser.parse_args()
    heads = [tuple(map(int, pair.split(':'))) for pair in args.heads.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    if args.audio_prompt:
        model.prepare_conditionals(args.audio_prompt)
    normalized = punc_norm(args.text)
    tokens = model.tokenizer(normalized, return_tensors='pt', padding=True, truncation=True).input_ids[0]
    with TurboAlignmentCapture(model.t3.tfmr, len(tokens), heads) as capture:
        wav = model.generate(args.text)
    args.output.mkdir(parents=True, exist_ok=False)
    sf.write(args.output / 'audio.wav', wav.squeeze().numpy(), model.sr)
    np.save(args.output / 'text-attention.npy', capture.text_attention().numpy())
    (args.output / 'metadata.json').write_text(json.dumps({
        'text': normalized,
        'text_token_ids': tokens.tolist(),
        'text_token_pieces': model.tokenizer.convert_ids_to_tokens(tokens.tolist()),
        'heads': heads,
        'sample_rate': model.sr,
        'samples': wav.numel(),
        'attention_axes': ['head', 'decoding_step', 'text_token'],
        'warning': 'Text-normalized attention, not word timestamps. Decoding rows may include EOS; generation filters invalid tokens and adds trailing silence before waveform synthesis.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
