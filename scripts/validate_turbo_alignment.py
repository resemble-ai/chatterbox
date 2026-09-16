"""Real-model diagnostic: no transcription model, no changed generation inputs."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chatterbox.models.t3.inference.turbo_alignment import TurboAlignmentCapture
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)
    model.prepare_conditionals(str(args.audio_prompt))
    heads = [(4, 6), (5, 14), (8, 4)]
    texts = [
        "First came a boom, then another boom, and finally silence.",
        "After the tension built, the crowd went boom boom—and the night erupted in laughter.",
        "The red boat passed the blue boat before sunset.",
    ]
    original_inference = model.s3gen.inference
    received = []

    def observe_tokens(*args, **kwargs):
        received.append(kwargs["speech_tokens"].detach().cpu().clone())
        return original_inference(*args, **kwargs)

    model.s3gen.inference = observe_tokens

    def generate(text, seed, enabled):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tokens = model.tokenizer(
            punc_norm(text),
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        if enabled:
            with TurboAlignmentCapture(
                model.t3.tfmr, len(tokens.input_ids[0]), heads
            ) as capture:
                audio = model.generate(text)
            attention = capture.text_attention().numpy()
        else:
            audio = model.generate(text)
            attention = None
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return audio, attention, tokens, time.perf_counter() - start

    with torch.inference_mode():
        model.generate("Ready.")
        results = []
        for index, (text, seed) in enumerate(
            (text, seed) for text in texts for seed in (968, 969)
        ):
            baseline, _, _, baseline_time = generate(text, seed, False)
            audio, attention, tokens, capture_time = generate(text, seed, True)
            assert torch.equal(baseline, audio), "observer changed waveform"
            assert torch.equal(received[-1], received[-2]), (
                "observer changed vocoder tokens"
            )
            assert (
                sum(len(block.attn.c_attn._forward_hooks) for block in model.t3.tfmr.h)
                == 0
            )
            sf.write(
                out / f"{index}.wav",
                audio.squeeze().numpy(),
                model.sr,
                subtype="PCM_16",
            )
            np.save(out / f"{index}.npy", attention)
            speech_tokens = received[-1].tolist()
            result = {
                "index": index,
                "text": punc_norm(text),
                "seed": seed,
                "heads": heads,
                "text_pieces": model.tokenizer.convert_ids_to_tokens(
                    tokens.input_ids[0].tolist()
                ),
                "text_offsets": tokens.offset_mapping[0].tolist(),
                "speech_tokens": speech_tokens,
                "samples": audio.numel(),
                "sample_rate": model.sr,
                "baseline_seconds": baseline_time,
                "capture_seconds": capture_time,
                "audio_equal": True,
                "speech_tokens_equal": True,
                "hooks_remaining": 0,
                "attention_shape": list(attention.shape),
            }
            # ponytail: this fixed corpus uses whole-word GPT-2 tokens. General word
            # timing needs subword alignment and acoustic calibration, not this probe.
            target = "boom" if index < 4 else "boat"
            token = [
                i
                for i, piece in enumerate(result["text_pieces"])
                if piece.removeprefix("Ġ") == target
            ][1]
            steps = np.flatnonzero(attention[0].argmax(-1) == token)
            assert len(steps), "target word was not observed"
            assert audio.numel() == len(speech_tokens) * 960, (
                "unexpected vocoder frame rate"
            )
            assert attention.shape[1] == len(speech_tokens) - 2, (
                "unexpected EOS/filter mapping"
            )
            result["candidate_second_word_end_seconds"] = float((steps[-1] + 1) / 25)
            result["warning"] = (
                "Raw attention-derived candidate; listener acceptance is application-specific"
            )
            results.append(result)
            (out / "generation.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("speech_tokens", "text_offsets", "text_pieces")
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
