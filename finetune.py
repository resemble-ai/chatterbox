"""Finetune Chatterbox-Turbo T3 for inline paralinguistic tag control.

Turbo uses a GPT-2 tokenizer (vocab 50,276) that already contains 19 paralinguistic
tags as dedicated token IDs (50257-50275), including [cough] [sigh] [laugh]
[chuckle] [whispering] [sarcastic] and others.  Our finetune dataset uses ~33
additional tags not in Turbo's base vocab; those are added on-the-fly at startup
and the T3.text_emb / T3.text_head layers are expanded accordingly.

Only T3 is trained.  S3Gen, HiFiGAN, VoiceEncoder, and the S3Tokenizer stay
frozen on the GPU and run inside the training step (no grad).

Usage
-----
.venv/bin/python finetune.py \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --splits_dir splits/ \
    --batch_size 4 --lr 1e-4 --max_steps 5000 \
    [--warmup_steps 500] [--grad_clip 1.0] [--weight_decay 0.01] \
    [--ref_mix 0.30,0.40,0.20,0.10] \
    [--ckpt_every 500] [--log_every 50] \
    [--resume checkpoints/step_000500.pt] \
    [--num_workers 4] [--seed 0] [--dry_run]

Notes on Turbo vs standard Chatterbox
-------------------------------------
- Backbone: GPT-2 medium  (not Llama_520M)
- Text tokenizer: AutoTokenizer (GPT-2 BPE + 19 added tags)
- text_tokens_dict_size: 50,276 pretrained  +  N new tags  =  ~50,309
- speech_cond_prompt_len: 375 (was 150 in standard)
- Turbo does NOT support CFG — no CFG dropout in training (would be inert)
- Turbo does NOT use emotion_adv — that branch in T3CondEnc is disabled
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR


MAX_DURATION = 30.0
MIN_DURATION = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ParalinguisticDataset(Dataset):
    """Returns (script_content, target_wav_16k, ref_wav_16k) per item."""

    def __init__(
        self,
        dataset_path: str,
        row_index: list[dict],
        train_indices: list[int],
        speaker_to_rows: dict[str, list[int]],
        neutral_refs: dict[str, list[int]],
        ref_mix: tuple[float, float, float, float] = (0.30, 0.40, 0.20, 0.10),
        seed: int = 42,
    ):
        self.shards = sorted(glob.glob(f"{dataset_path}/data/train-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(f"No parquet shards found under {dataset_path}/data/")
        self.row_index    = row_index
        self.indices      = train_indices
        self.spk2rows     = speaker_to_rows
        self.neutral_refs = neutral_refs
        self.ref_mix      = ref_mix
        self._rng         = random.Random(seed)
        self._all_neutral = [idx for idx, r in enumerate(row_index)
                              if not r["inline_tags"] and not r.get("tags_col")]
        self._all_speakers = list(speaker_to_rows.keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_idx = self.indices[i]
        row     = self.row_index[row_idx]
        vid     = row["voice_id"]

        ref_idx = self._pick_ref(row_idx, vid, row["inline_tags"])

        script, target_wav = self._load_row(row_idx)
        _,      ref_wav    = self._load_row(ref_idx)

        return {
            "script_content": script,
            "target_wav":     target_wav,
            "ref_wav":        ref_wav,
        }

    def _pick_ref(self, target_idx: int, vid: str, target_tags: list[str]) -> int:
        same_spk = self.spk2rows.get(vid, [])
        neutral  = self.neutral_refs.get(vid, [])
        mode = self._rng.choices(
            ["matched", "neutral_same", "cross", "self"],
            weights=list(self.ref_mix),
        )[0]

        if mode == "matched":
            tagged = [i for i in same_spk
                      if i != target_idx
                      and any(t in (self.row_index[i].get("inline_tags") or [])
                              for t in target_tags)]
            if tagged:
                return self._rng.choice(tagged)

        if mode in ("matched", "neutral_same"):
            others = [i for i in (neutral if neutral else same_spk) if i != target_idx]
            if others:
                return self._rng.choice(others)

        if mode == "cross":
            other_spks = [s for s in self._all_speakers if s != vid]
            if other_spks:
                other_vid = self._rng.choice(other_spks)
                other_neut = self.neutral_refs.get(other_vid) or self.spk2rows.get(other_vid, [])
                if other_neut:
                    return self._rng.choice(other_neut)
            if self._all_neutral:
                return self._rng.choice(self._all_neutral)

        return target_idx

    def _load_row(self, row_idx: int) -> tuple[str, torch.Tensor]:
        r = self.row_index[row_idx]
        shard_path = self.shards[r["shard"]]
        pf = pq.ParquetFile(shard_path)
        rows_seen = 0
        for rg in range(pf.num_row_groups):
            rg_size = pf.metadata.row_group(rg).num_rows
            if rows_seen + rg_size > r["row"]:
                local = r["row"] - rows_seen
                tbl = pf.read_row_group(
                    rg, columns=["preprocessed_audio", "script_content"]
                ).slice(local, 1).to_pydict()
                audio_struct = tbl["preprocessed_audio"][0]
                script = tbl["script_content"][0] or ""
                raw = audio_struct["bytes"] if audio_struct else None
                break
            rows_seen += rg_size
        else:
            return "", torch.zeros(16000)

        if not raw:
            return script, torch.zeros(16000)

        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)
        if sr != S3_SR:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=S3_SR)
        return script, wav


# ─────────────────────────────────────────────────────────────────────────────
# Collate + GPU batch preparation
# ─────────────────────────────────────────────────────────────────────────────

def collate_raw(batch):
    return {
        "script_content": [b["script_content"] for b in batch],
        "target_wavs":    [b["target_wav"]     for b in batch],
        "ref_wavs":       [b["ref_wav"]        for b in batch],
    }


def prepare_batch(raw_batch, model, device: str, speech_cond_prompt_len: int):
    """Run frozen GPU models on a raw batch dict; return T3-ready tensors."""
    scripts     = raw_batch["script_content"]
    target_wavs = raw_batch["target_wavs"]
    ref_wavs    = raw_batch["ref_wavs"]

    tokenizer = model.tokenizer
    s3_tok    = model.s3gen.tokenizer
    ve        = model.ve
    B         = len(scripts)

    with torch.no_grad():
        speech_tokens_raw, speech_lens_raw = s3_tok(target_wavs)
    with torch.no_grad():
        cond_prompt_raw, _ = s3_tok(ref_wavs, max_len=speech_cond_prompt_len)

    ref_np = [w.cpu().numpy() for w in ref_wavs]
    with torch.no_grad():
        spk_emb_np = ve.embeds_from_wavs(ref_np, sample_rate=S3_SR)
    if isinstance(spk_emb_np, np.ndarray):
        speaker_emb = torch.from_numpy(spk_emb_np.astype(np.float32))
    else:
        speaker_emb = spk_emb_np.cpu()
    if speaker_emb.ndim == 1:
        speaker_emb = speaker_emb.unsqueeze(0)
    speaker_emb = speaker_emb.to(device)
    cond_prompt = cond_prompt_raw.to(device)

    # Turbo uses GPT-2 AutoTokenizer with <|endoftext|> (50256) as BOS/EOS/PAD.
    # T3.forward's _ensure_BOT_EOT assertion requires start_text_token and
    # stop_text_token to appear in every sequence.  We set both to the GPT-2 eos
    # token and prepend+append it to every text sequence.
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    text_ids_list = []
    for sc in scripts:
        enc = tokenizer(punc_norm(sc), return_tensors="pt",
                        padding=False, truncation=True, max_length=1022)
        ids = enc.input_ids[0].long()
        ids = F.pad(ids, (1, 0), value=sot)   # prepend SOT
        ids = F.pad(ids, (0, 1), value=eot)   # append  EOT
        text_ids_list.append(ids)

    text_lens = torch.tensor([t.size(0) for t in text_ids_list], dtype=torch.long)
    max_text  = int(text_lens.max().item())
    text_pad  = torch.full((B, max_text), pad_id, dtype=torch.long)
    for i, t in enumerate(text_ids_list):
        text_pad[i, :t.size(0)] = t

    speech_lens = speech_lens_raw.cpu()
    max_speech  = int(speech_lens.max().item())
    speech_pad  = torch.zeros(B, max_speech, dtype=torch.long)
    for i in range(B):
        L = int(speech_lens[i].item())
        if isinstance(speech_tokens_raw, torch.Tensor):
            speech_pad[i, :L] = speech_tokens_raw[i, :L].cpu()
        else:
            tok = torch.tensor(speech_tokens_raw[i], dtype=torch.long)
            speech_pad[i, :tok.size(0)] = tok[:L]

    t3_cond = T3Cond(
        speaker_emb=speaker_emb.to(dtype=torch.float32),
        cond_prompt_speech_tokens=cond_prompt.to(dtype=torch.long),
        # emotion_adv is ignored by Turbo (hp.emotion_adv=False) but we pass a placeholder
        # so the dataclass shape stays consistent with the standard model code path.
        emotion_adv=0.5 * torch.ones(B, 1, 1, device=device),
    ).to(device=device)

    return {
        "t3_cond":           t3_cond,
        "text_tokens":       text_pad.to(device),
        "text_token_lens":   text_lens.to(device),
        "speech_tokens":     speech_pad.to(device),
        "speech_token_lens": speech_lens.to(device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary expansion — add new tags, grow T3.text_emb / text_head
# ─────────────────────────────────────────────────────────────────────────────

def expand_vocab_for_tags(model, splits_dir: str, init_std: float = 0.02):
    """Add every trained tag (as bracketed string) to the tokenizer and expand
    T3.text_emb + T3.text_head to match. Returns the list of tags that were newly
    added.  Idempotent: tags already in the tokenizer are left alone.
    """
    wl = json.load(open(Path(splits_dir) / "tag_whitelist.json"))
    trained = (
        wl["event"] + wl["prosody"] + wl["emotional"] + wl["neutral_baseline"]
    )
    wrapped = [f"[{t}]" for t in trained]

    tokenizer = model.tokenizer
    vocab = set(tokenizer.get_vocab().keys())
    to_add = [t for t in wrapped if t not in vocab]

    if not to_add:
        print(f"[expand_vocab] all {len(wrapped)} tags already in tokenizer — no expansion needed")
        return []

    added = tokenizer.add_tokens(to_add)
    assert added == len(to_add), f"expected to add {len(to_add)} tokens, got {added}"

    t3 = model.t3
    old_vocab = t3.text_emb.num_embeddings
    new_vocab = old_vocab + added
    dim       = t3.text_emb.embedding_dim
    device    = t3.text_emb.weight.device
    dtype     = t3.text_emb.weight.dtype

    # Expand text_emb (input embedding table)
    new_emb = nn.Embedding(new_vocab, dim).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight[:old_vocab] = t3.text_emb.weight
        nn.init.normal_(new_emb.weight[old_vocab:], std=init_std)
    t3.text_emb = new_emb

    # Expand text_head (output projection)  (no bias in standard chatterbox T3)
    old_head = t3.text_head
    has_bias = old_head.bias is not None
    new_head = nn.Linear(old_head.in_features, new_vocab, bias=has_bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_head.weight[:old_vocab] = old_head.weight
        nn.init.normal_(new_head.weight[old_vocab:], std=init_std)
        if has_bias:
            new_head.bias[:old_vocab] = old_head.bias
            nn.init.zeros_(new_head.bias[old_vocab:])
    t3.text_head = new_head

    t3.hp.text_tokens_dict_size = new_vocab

    print(f"[expand_vocab] added {added} tokens; text vocab: {old_vocab} -> {new_vocab}")
    print(f"[expand_vocab] new tags: {to_add}")
    return to_add


# ─────────────────────────────────────────────────────────────────────────────
# Correct next-token loss (replaces the degenerate upstream T3.loss)
# ─────────────────────────────────────────────────────────────────────────────

def t3_next_token_loss(model, batch, text_loss_weight: float = 0.1):
    """Proper causal-LM training loss for T3.

    The upstream T3.loss() feeds the same `speech_tokens` tensor as both input
    AND target.  In a causal transformer that's a degenerate identity-function
    objective: logits at position i can attend to (and trivially copy) the
    input at position i.  The optimizer happily collapses speech_head into
    a near-identity projection — loss goes down, generation breaks.

    The correct objective:
      input_speech  = [BOS, s_0, s_1, ..., s_{L-1}]            (length L+1)
      target_speech = [s_0, s_1, ..., s_{L-1}, EOS]            (length L+1)
      logits[i]  ←— supervised to predict —→  target[i]
                                              (= input[i+1] except for EOS at end)

    This way the model learns to (1) produce the first token from BOS, (2) emit
    EOS when finished — both behaviors needed by inference_turbo.

    Text is auxiliary (not autoregressively generated at inference) but still
    benefits from being shifted — we apply the same trick to text_tokens.
    """
    BOS_speech = model.t3.hp.start_speech_token   # 6561
    EOS_speech = model.t3.hp.stop_speech_token    # 6562
    SOT_text   = model.t3.hp.start_text_token     # 50256 for Turbo
    EOT_text   = model.t3.hp.stop_text_token      # 50256 for Turbo

    text_tokens   = batch["text_tokens"]
    text_lens     = batch["text_token_lens"]
    speech_tokens = batch["speech_tokens"]
    speech_lens   = batch["speech_token_lens"]

    B, T_s = speech_tokens.shape
    device = speech_tokens.device

    # ── shift speech: input = [BOS, s_0..s_{T_s-1}],  target = [s_0..s_{T_s-1}, EOS]
    in_speech = torch.full((B, T_s + 1), BOS_speech, dtype=torch.long, device=device)
    in_speech[:, 1:] = speech_tokens                                 # prepend BOS

    tg_speech = torch.full((B, T_s + 1), -100, dtype=torch.long, device=device)
    pos = torch.arange(T_s + 1, device=device)[None].expand(B, -1)   # (B, T_s+1)
    valid_target = pos < (speech_lens[:, None] + 1)                  # include the EOS position
    valid_input  = pos < speech_lens[:, None]                        # source token positions
    tg_speech[valid_input] = speech_tokens[valid_input[:, :T_s]]      # copy s_0..s_{L-1}
    eos_pos = speech_lens                                            # position of EOS for each row
    tg_speech[torch.arange(B, device=device), eos_pos] = EOS_speech  # set EOS at the right position

    in_speech_lens = speech_lens + 1                                 # the BOS counts; EOS supervised via target

    # ── text: same shift, but text is auxiliary
    # text_tokens already has SOT prepended and EOT appended in the collate, so
    # we just compute next-token loss directly on it: logits[i] predicts text[i+1]
    # using the existing forward (we won't add an extra BOS to text).
    out = model.t3.forward(
        t3_cond=batch["t3_cond"],
        text_tokens=text_tokens,
        text_token_lens=text_lens,
        speech_tokens=in_speech,
        speech_token_lens=in_speech_lens,
        training=True,
    )

    # ── speech loss with correct shift
    speech_logits = out.speech_logits.permute(0, 2, 1)   # (B, vocab, T_s+1)
    loss_speech = F.cross_entropy(speech_logits, tg_speech, ignore_index=-100)

    # ── text loss with shift: logits[i] → text[i+1]
    text_logits = out.text_logits  # (B, T_text, vocab_text)
    text_logits = text_logits[:, :-1, :].permute(0, 2, 1).contiguous()  # (B, vocab, T_text-1)
    text_target = text_tokens[:, 1:].clone()                            # (B, T_text-1)
    text_pos = torch.arange(text_target.size(1), device=device)[None].expand(B, -1)
    text_pad_mask = text_pos >= (text_lens[:, None] - 1)
    text_target[text_pad_mask] = -100
    loss_text = F.cross_entropy(text_logits, text_target, ignore_index=-100)

    return loss_text, loss_speech, text_loss_weight * loss_text + loss_speech


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(step, model, optimizer, scheduler, args, added_tags):
    out = Path(args.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step_{step:06d}.pt"
    torch.save({
        "step":            step,
        "t3_state":        model.t3.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "added_tags":      added_tags,
        "text_vocab_size": model.t3.hp.text_tokens_dict_size,
        "args":            vars(args),
    }, path)
    print(f"  [ckpt] saved {path}", flush=True)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    # ckpt may include expanded vocab; ensure model matches before load
    ckpt_vocab = ckpt.get("text_vocab_size")
    cur_vocab  = model.t3.text_emb.num_embeddings
    if ckpt_vocab and ckpt_vocab != cur_vocab:
        raise RuntimeError(
            f"Checkpoint expects text_vocab={ckpt_vocab} but model has {cur_vocab}. "
            f"Run with the same tag_whitelist.json that produced this checkpoint."
        )
    model.t3.load_state_dict(ckpt["t3_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--splits_dir",   default="splits/")
    ap.add_argument("--ckpt_dir",     default="checkpoints/")
    ap.add_argument("--batch_size",   type=int,   default=4)
    ap.add_argument("--lr",           type=float, default=2e-5,
                    help="LR for finetuning a 427M pretrained LM. 2e-5 is standard; "
                         "1e-4 is too aggressive and contributed to the previous diverged run.")
    ap.add_argument("--text_loss_weight", type=float, default=0.1,
                    help="Weight on the auxiliary text-reconstruction loss. The main objective "
                         "is the speech-token next-token CE (weight=1.0). 0.1 keeps text as a "
                         "mild regularizer without letting it dominate the gradient.")
    ap.add_argument("--max_steps",    type=int,   default=5000)
    ap.add_argument("--warmup_steps", type=int,   default=500)
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--ref_mix",      type=str,   default="0.30,0.40,0.20,0.10",
                    help="Ref mode probs: matched_tagged,neutral_same,cross,self")
    ap.add_argument("--ckpt_every",   type=int,   default=500)
    ap.add_argument("--log_every",    type=int,   default=50)
    ap.add_argument("--resume",       type=str,   default=None)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",       type=str,   default="cuda")
    ap.add_argument("--dry_run",      action="store_true")
    ap.add_argument("--log_file",     type=str, default=None,
                    help="Optional path to mirror stdout to. Defaults to "
                         "<ckpt_dir>/training_<timestamp>.log if not set.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ── tee stdout to log file ────────────────────────────────────────────
    import datetime
    log_path = args.log_file
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path(args.ckpt_dir) / f"training_{ts}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)  # line-buffered

    class _Tee:
        def __init__(self, *streams): self._streams = streams
        def write(self, s):
            for x in self._streams:
                x.write(s); x.flush()
        def flush(self):
            for x in self._streams:
                x.flush()
        def isatty(self): return False
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    print(f"[finetune] logging to {log_path}", flush=True)

    ref_mix = tuple(float(x) for x in args.ref_mix.split(","))
    assert len(ref_mix) == 4 and abs(sum(ref_mix) - 1.0) < 1e-4, \
        "--ref_mix must be 4 comma-separated probs summing to 1"

    # ── splits ─────────────────────────────────────────────────────────────
    spl = Path(args.splits_dir)
    ri_path = Path("analysis/row_index.jsonl")
    if (spl / "row_index.jsonl").exists():
        ri_path = spl / "row_index.jsonl"
    with open(ri_path) as f:
        row_index = [json.loads(line) for line in f]

    with open(spl / "train.json") as f:
        train_indices = json.load(f)
    with open(spl / "speaker_to_rows.json") as f:
        speaker_to_rows = json.load(f)
    with open(spl / "neutral_refs_by_speaker.json") as f:
        neutral_refs = json.load(f)

    print(f"[finetune] rows={len(train_indices)}  speakers={len(speaker_to_rows)}  "
          f"neutral_refs={sum(len(v) for v in neutral_refs.values())}", flush=True)

    # ── Turbo model ───────────────────────────────────────────────────────
    print("[finetune] loading ChatterboxTurboTTS ...", flush=True)
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print(f"[finetune] turbo loaded. text_vocab={model.t3.text_emb.num_embeddings} "
          f"speech_vocab={model.t3.hp.speech_tokens_dict_size} "
          f"cond_prompt_len={model.t3.hp.speech_cond_prompt_len}", flush=True)

    # ── align text SOT/EOT with the GPT-2 tokenizer's <|endoftext|> ──────
    # Default T3Config has start_text_token=255, stop_text_token=0 — those are
    # legacy values from the Llama_520M path.  For Turbo we use the GPT-2
    # sentinel (50256) so (a) the token embedding exists in the pretrained
    # tokenizer and (b) _ensure_BOT_EOT passes during T3.forward.
    eos_id = model.tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer must have an eos_token_id"
    model.t3.hp.start_text_token = eos_id
    model.t3.hp.stop_text_token  = eos_id
    print(f"[finetune] T3 SOT/EOT set to eos_token_id={eos_id}", flush=True)

    # ── expand vocab for any missing tags ─────────────────────────────────
    added_tags = expand_vocab_for_tags(model, args.splits_dir)

    # ── freeze everything except T3 ───────────────────────────────────────
    for m in (model.ve, model.s3gen):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    model.t3.train()
    for p in model.t3.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.t3.parameters() if p.requires_grad)
    print(f"[finetune] T3 trainable params: {n_trainable/1e6:.1f}M", flush=True)

    speech_cond_prompt_len = int(model.t3.hp.speech_cond_prompt_len)  # 375 for Turbo

    # ── dataset / loader ──────────────────────────────────────────────────
    ds = ParalinguisticDataset(
        dataset_path=args.dataset_path,
        row_index=row_index,
        train_indices=train_indices,
        speaker_to_rows=speaker_to_rows,
        neutral_refs=neutral_refs,
        ref_mix=ref_mix,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_raw,
        pin_memory=False,
        drop_last=True,
    )
    prepare = lambda raw: prepare_batch(raw, model, device, speech_cond_prompt_len)

    # ── optimizer / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.t3.parameters(),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_steps - args.warmup_steps), eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps],
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device) + 1
        print(f"[finetune] resumed from {args.resume}, step={start_step}", flush=True)

    # ── training loop ─────────────────────────────────────────────────────
    loader_iter  = iter(loader)
    running_loss = 0.0

    for step in range(start_step, args.max_steps):
        try:
            raw = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            raw = next(loader_iter)

        batch = prepare(raw)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_text, loss_speech, loss = t3_next_token_loss(
                model, batch, text_loss_weight=args.text_loss_weight
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.t3.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % args.log_every == 0:
            avg = running_loss / args.log_every if step > 0 else loss.item()
            running_loss = 0.0
            lr = scheduler.get_last_lr()[0]
            print(
                f"step={step:6d}  loss={loss.item():.4f} "
                f"(text={loss_text.item():.4f} speech={loss_speech.item():.4f})  "
                f"avg={avg:.4f}  lr={lr:.2e}",
                flush=True,
            )

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(step, model, optimizer, scheduler, args, added_tags)

        if args.dry_run:
            print(f"[dry_run] step={step} loss={loss.item():.4f} — OK")
            return

    save_checkpoint(args.max_steps, model, optimizer, scheduler, args, added_tags)
    print("[finetune] done.", flush=True)


if __name__ == "__main__":
    main()
