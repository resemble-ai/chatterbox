"""Single-shot inference with Chatterbox-Turbo + finetuned T3 checkpoint.

Loads the pretrained Turbo pipeline, optionally expands the vocab to match the
finetune checkpoint (must be done BEFORE load_state_dict so the embedding/head
shapes match), overrides T3 weights with the checkpoint, and runs generate().

Usage
-----
.venv/bin/python run_inference.py \
    --text "[laughs] Oh no, not again. [monotone] Fantastic." \
    --reference_audio ref.wav \
    --output out.wav \
    --checkpoint checkpoints/step_005000.pt \
    [--temperature 0.8] [--top_p 0.95] [--top_k 1000] [--repetition_penalty 1.2] \
    [--norm_loudness 1]

# Run the pretrained Turbo baseline (no finetune, no expansion):
.venv/bin/python run_inference.py \
    --text "[laughs] Oh no, not again." \
    --reference_audio ref.wav \
    --output baseline.wav \
    --checkpoint pretrained
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatterbox.tts_turbo import ChatterboxTurboTTS


def expand_vocab_for_tags(model, splits_dir: str):
    """Same expansion logic as finetune.py — must be applied BEFORE loading a
    finetuned checkpoint so the text_emb / text_head shapes match."""
    wl = json.load(open(Path(splits_dir) / "tag_whitelist.json"))
    trained = (wl["event"] + wl["prosody"] + wl["emotional"] + wl["neutral_baseline"])
    wrapped = [f"[{t}]" for t in trained]

    tokenizer = model.tokenizer
    vocab = set(tokenizer.get_vocab().keys())
    to_add = [t for t in wrapped if t not in vocab]

    if not to_add:
        return []

    tokenizer.add_tokens(to_add)
    t3 = model.t3
    old_vocab = t3.text_emb.num_embeddings
    new_vocab = old_vocab + len(to_add)
    dim       = t3.text_emb.embedding_dim
    device    = t3.text_emb.weight.device
    dtype     = t3.text_emb.weight.dtype

    new_emb = nn.Embedding(new_vocab, dim).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight[:old_vocab] = t3.text_emb.weight
        nn.init.normal_(new_emb.weight[old_vocab:], std=0.02)
    t3.text_emb = new_emb

    old_head = t3.text_head
    has_bias = old_head.bias is not None
    new_head = nn.Linear(old_head.in_features, new_vocab, bias=has_bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_head.weight[:old_vocab] = old_head.weight
        nn.init.normal_(new_head.weight[old_vocab:], std=0.02)
        if has_bias:
            new_head.bias[:old_vocab] = old_head.bias
            nn.init.zeros_(new_head.bias[old_vocab:])
    t3.text_head = new_head

    t3.hp.text_tokens_dict_size = new_vocab
    return to_add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--reference_audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to finetuned ckpt, or 'pretrained' to skip override")
    ap.add_argument("--splits_dir", default="splits/")
    ap.add_argument("--temperature",        type=float, default=0.8)
    ap.add_argument("--top_p",              type=float, default=0.95)
    ap.add_argument("--top_k",              type=int,   default=1000)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--norm_loudness",      type=int,   default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("[inference] loading Chatterbox-Turbo ...", flush=True)
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

    # align SOT/EOT like the finetune
    eos_id = model.tokenizer.eos_token_id
    model.t3.hp.start_text_token = eos_id
    model.t3.hp.stop_text_token  = eos_id

    if args.checkpoint == "pretrained":
        print("[inference] running pretrained Turbo baseline (no finetune override)")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Expand vocab to match the checkpoint BEFORE loading state
        expected_vocab = ckpt.get("text_vocab_size")
        if expected_vocab and expected_vocab > model.t3.text_emb.num_embeddings:
            added = expand_vocab_for_tags(model, args.splits_dir)
            print(f"[inference] expanded vocab for checkpoint: added {len(added)} tags "
                  f"(text_vocab now {model.t3.text_emb.num_embeddings})")
        t3_state = ckpt.get("t3_state", ckpt)
        missing, unexpected = model.t3.load_state_dict(t3_state, strict=False)
        if missing:    print(f"[inference] missing keys: {len(missing)} (expected for deleted tfmr.wte)")
        if unexpected: print(f"[inference] unexpected keys: {unexpected[:5]}...")
        model.t3.to(args.device).eval()
        print(f"[inference] loaded finetuned T3 from step {ckpt.get('step', '?')}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"[inference] generating for text: {args.text[:80]!r} ...", flush=True)
    wav = model.generate(
        args.text,
        audio_prompt_path=args.reference_audio,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        norm_loudness=bool(args.norm_loudness),
    )
    torchaudio.save(args.output, wav, sample_rate=model.sr)
    print(f"[inference] saved {args.output}  ({wav.shape[-1]/model.sr:.2f}s @ {model.sr}Hz)")


if __name__ == "__main__":
    main()
