# Finetune Chatterbox-Turbo T3 for reliable inline-tag control — phased plan

> **Last updated:** 2026-04-23 (after Turbo switch + Phase-0 dataset analysis).
> Usage / commands / architecture diagram live in `~/chatterbox/README_finetune.md`. This file is the "why" document: design decisions, critiques, trade-offs.

## Context

The user wants a TTS model that honors **inline bracketed paralinguistic tags** in the text input — e.g. `[laugh]`, `[cough]`, `[whispering]`, `[monotone]`, `[sarcastic]`, `[frustrated]`, `[pause]`. The goal is **reliable tag-faithful control**, not just general domain adaptation.

**Source:** `InternalCan/stage1-processed-with-audio-aligned` (83,344 rows, 104 parquet shards, cached locally).
**Chosen base model:** `ChatterboxTurboTTS` (NOT standard Chatterbox — see §"Turbo vs standard" below).
**Training method:** full T3 finetune (all 427.4M parameters get gradients). Not LoRA — see §"Finetuning method: full vs LoRA".
**Echo-tts and the pre-computed codebook columns are out of scope.**

Tags appear both as a `tags` list field and inline in `script_content` (e.g. `"[frustrated] Look, I don't make the rules! [speeding up] If it doesn't fit..."`); aligned target audio is in `preprocessed_audio.bytes`.

---

## Turbo vs standard Chatterbox — why we switched

After inspecting `gradio_tts_turbo_app.py` we discovered Turbo was **designed for exactly this task**:

- Turbo uses the GPT-2 tokenizer (vocab 50,257) extended with **19 paralinguistic tag tokens as dedicated single-token IDs** (50,257–50,275): `[cough]`, `[sigh]`, `[laugh]`, `[chuckle]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`, `[whispering]`, `[sarcastic]`, `[angry]`, `[fear]`, `[surprised]`, `[happy]`, `[crying]`, `[dramatic]`, `[narration]`, `[advertisement]`.
- **7 of our 40 target tags already exist as pretrained single tokens in Turbo.** Free capability on day zero.
- Turbo's backbone is GPT-2 medium (1024 hidden, 24 layers) — richer linguistic prior than the standard model's Llama_520M + 704-BPE.
- `del t3.tfmr.wte` after load: Turbo deletes the GPT-2 backbone's word-token-embedding table (≈200 MB) because T3 always passes `inputs_embeds=` via its own `text_emb`. Same trick applies to our finetune.

**Costs of Turbo:**

- No CFG at inference (`tts_turbo.py` line 233 explicitly logs that `cfg_weight` / `min_p` / `exaggeration` are ignored). That removes one lever for amplifying tag signal at inference. We compensate via reference-mixing strategy during training.
- `speech_cond_prompt_len = 375` (vs 150 standard). Longer acoustic reference → more speaker signal in the conditioning, which makes the "reference can bury the tag" problem slightly worse. The reference-mix tuning already addresses this.
- Vocab expansion needs `T3.text_emb` and `T3.text_head` to grow by `N_new_tags` rows/columns — implemented inline in `finetune.py`.

---

## Finetuning method: full vs LoRA

**We run a full finetune — all 427.4M T3 parameters get gradients.** Not LoRA.

Reasoning:

1. **Vocab expansion forces at least some layers to be fully trainable.** We add 33 new rows to `text_emb` (50,276 → 50,309) and 33 new columns to `text_head`. Those rows/columns start from random init and need full gradient access; LoRA can't adapt rows that don't exist in the pretrained matrix. A LoRA setup would end up hybrid: LoRA on the GPT-2 backbone + full finetune on the expanded `text_emb`/`text_head`/`cond_enc` — more plumbing for a small saving.
2. **Scale is modest.** 427M params at bf16 + AdamW optimizer state ≈ 5 GB for weights and optimizer; activations at batch 4 fit comfortably on a single 24 GB GPU.
3. **5 k steps is short.** LoRA's main win (cheap per-experiment iteration) matters more for hyperparameter sweeps than a single training run.

**When to add LoRA later (opt-in path):** many tag-subset experiments, many shippable variants, VRAM-constrained hardware, or stackable tag capabilities. The integration point would be `peft.LoraConfig` on `model.t3.tfmr` with `target_modules=["c_attn", "c_proj"]`, while keeping `text_emb`, `text_head`, `cond_enc`, `speech_emb`, `speech_head` fully trainable. Not implemented in Phase 1.

---

## Critiques this plan is designed to address

1. Next-token CE over (text, audio) does **not** directly supervise "tag → acoustic change." Without measurement and counterfactuals we cannot claim the model learned tags.
2. T3 is an AR LM over semantic tokens. It can only express tag effects through sounds S3Gen + HiFiGAN can render. Events (`[laugh]`) and prosody (`[whispering]`) are in S3Gen's training distribution; **accent and identity tags are not** — S3Gen's identity comes from the speaker xvector, not the text.
3. Reference-audio conditioning (speaker embedding + `cond_prompt_speech_tokens`) is a strong signal and can drown tag effects. If the reference is calm and the text says `[shouts]`, the plan must specify what should win and train accordingly.
4. "Different clip from same speaker" can still share session/style clusters. Counterfactual reference construction must be explicit.
5. Tokenization of bracketed tags matters. Under the 704-BPE of standard Chatterbox, `[laugh]` was 6 subword pieces; under Turbo's GPT-2 BPE it's 1 token (IDs 50257–50275 for the pretrained ones). For our newly-added tags it's also 1 token after expansion.
6. Tag categories are heterogeneous (events vs prosody vs accent/identity). Lumping them produces muddy results.
7. Tag imbalance is severe (26,047 unique bracketed strings; 15,859 singletons; only 77 with ≥500 rows). Without scoping, frequent tags dominate learning and rare tags never converge.

## Response to critiques (design decisions)

- **Phase 0 (dataset analysis) runs before training.** Outputs drive tag scoping — not vibes.
- **First experiment is scoped to 40 tags** (≥600 rows, ≥150/217 speakers, event/prosody/emotional only). Accent/identity deferred to Phase 2.
- **Reference-audio strategy mixes four modes** (matched-tagged, neutralized-same-speaker, cross-speaker, self-fallback). The 40% neutralized-ref weight prevents the model from using reference acoustics as a shortcut for tag content.
- **Vocabulary expansion is frequency-gated.** Tags with ≥50 training rows get dedicated token IDs; rarer ones keep BPE fragmentation (a random embedding with <10 gradient updates is strictly worse than pretrained BPE context).
- **Evaluation is first-class.** Held-out eval set with 4 structured slices and counterfactual conditions (with/without tag, matched/neutral ref, shuffled tags).
- **T3-only full finetune is a first experiment, not a final claim.** If Phase-1c eval shows it's insufficient for a category, we plan Phase 2 separately.

---

## Phases

- **Phase 0** — Dataset analysis. No training code. Produces tag stats, BPE tokenization report, row index, split files, curated tag whitelist.
- **Phase 1a** — Baseline capture. Run `evaluate.py --checkpoint pretrained` BEFORE any training.
- **Phase 1b** — T3 full finetune on 40-tag whitelist with reference-mixing and vocab expansion.
- **Phase 1c** — Eval on held-out slices with counterfactual conditions. Go/no-go.
- **Phase 2** (deferred) — Accent/identity tag support if Phase 1c shows T3-only is insufficient. Separate plan.

Each phase produces artifacts the next consumes. Do not skip.

---

## Phase 0 — Dataset analysis (done, see outputs)

**Scripts:** `analysis/analyze_tags.py`, `analysis/verify_speakers.py`, `splits/build_splits.py`.

**Findings (from actual run, not assumptions):**

- **83,344 rows, 217 distinct speakers** (voice_id = 13-char hex prefix in filename, not the collection UUID in the GCS path — we initially misread the path and thought it was one speaker).
- **27 English locales** (`english_north-american`, `english_british`, `english_canadian`, `english_indian-india`, etc.).
- **~400 rows / speaker, ~20 neutral rows / speaker** — remarkably balanced.
- **26,047 unique bracketed tag strings** with a brutal long tail: 15,859 singletons, 77 tags with ≥500 rows (covering 23.7% of all tag mentions), 429 tags with ≥100 rows (52.3% cumulatively).
- **Tag tokenization** under Turbo's GPT-2 BPE: 7 of 40 target tags are already single tokens (Resemble's 19 pretrained tag tokens); the remaining 33 are multi-token under BPE but become single tokens after our vocab expansion.

**Curated whitelist (40 tags, `splits/tag_whitelist.json`):**

| Category | Tags |
|---|---|
| event (9) | `cough`, `clearing throat`, `sigh`, `laugh`, `chuckle`, `bitter laugh`, `belly laugh`, `pause`, `long pause` |
| prosody (13) | `quickly`, `slowly`, `very slowly`, `very quietly`, `whispering`, `shouting`, `monotone`, `trailing off`, `mumbling`, `deep pitch`, `speeding up`, `more firmly`, `firmly` |
| emotional (17) | `warmly`, `warm`, `confidently`, `passionately`, `friendly`, `delighted`, `exhausted`, `animated`, `curious`, `playful`, `hesitantly`, `stern`, `mocking`, `excited`, `joking`, `sarcastic`, `frustrated` |
| neutral baseline (1) | `normal voice` |

**Splits (`splits/`):**

- `train.json` — 74,618 rows (195 speakers)
- `held_out_speakers.json` — 22 speakers withheld entirely for eval
- `eval_slices.json` — 2,020 structured entries across 4 slices:
  - `S_tag_presence` (760) — same speaker with/without tag pairs
  - `S_cross_speaker` (760) — same tag across held-out speakers
  - `S_neutral` (300) — baseline-drift control
  - `S_multi_tag` (200) — ≥2 trained tags per row
- `speaker_to_rows.json`, `neutral_refs_by_speaker.json`, `locale_to_speakers.json`

**Staged whitelist expansion (Phase 1.5+, optional):**

| Frequency band | # tags | Add to vocab? |
|---|---|---|
| ≥ 500 rows | 77 | Yes |
| 100–500 rows | 352 | Yes (Phase 2) |
| 20–100 rows | 1,390 | Maybe — cluster first |
| < 20 rows | 23,000+ | No — BPE fragments carry pretrained context; random embeddings with <10 gradient updates are worse |

---

## Phase 1a — Baseline capture

```
.venv/bin/python evaluate.py \
    --checkpoint pretrained \
    --splits_dir splits/ \
    --dataset_path <snap> \
    --out_dir eval_runs/baseline/
```

Runs the pretrained Turbo model (no finetune, no vocab expansion) on the full eval set. Every Phase-1c metric compares against this.

**Already observed** on smoke-test (2 entries, 2 conditions): `[frustrated]` in text vs stripped gives F0 mean 704 → 410 Hz on one row — the baseline already expresses *some* tag behavior thanks to Turbo's 19 pretrained tag tokens. Our finetuned model needs to **beat** this baseline, not just express tags.

---

## Phase 1b — T3 full finetune (script: `finetune.py`)

### T3 inputs (same structure for training and inference)

| Input | Shape | Source | Role |
|---|---|---|---|
| `t3_cond.speaker_emb` | (B, 256) | `VoiceEncoder(ref_wav_16k)` | Speaker identity; projected to hidden and prepended |
| `t3_cond.cond_prompt_speech_tokens` | (B, 375) | `S3Tokenizer(ref_wav_16k, max_len=375)` | Acoustic reference prefix; embedded via `speech_emb` |
| `t3_cond.emotion_adv` | (B, 1, 1) | scalar 0.5 | Ignored by Turbo (`hp.emotion_adv=False`) |
| `text_tokens` | (B, T_text) | `AutoTokenizer(script)` + SOT/EOT | Text conditioning |
| `text_token_lens` | (B,) | per-sequence lengths | Masks padding in loss |
| `speech_tokens` | (B, T_speech) | `S3Tokenizer(target_wav_16k)` | **Training target** (teacher-forced); generated at inference |
| `speech_token_lens` | (B,) | per-sequence lengths | Masks padding in loss |

### T3 outputs (training)

`T3.loss(...)` → `(loss_text, loss_speech)`:

- `text_logits`:   (B, T_text,   `50276 + N_new_tags`) — auxiliary reconstruction loss
- `speech_logits`: (B, T_speech, 6563)                 — **main loss**; `6563 = 6561 codec + SOS + EOS`

### T3 outputs (inference)

`T3.inference_turbo(...)` → `speech_tokens` of shape (1, T_gen). Filtered to `< 6561` before being handed to S3Gen — T3's output vocab is a superset of S3Gen's codec vocab (6,561 valid codec classes at indices 0–6560); IDs ≥ 6561 are SOS/EOS sentinels that the codec cannot decode.

### Vocabulary expansion at startup

```python
# finetune.py :: expand_vocab_for_tags
tokenizer.add_tokens(["[pause]", "[long pause]", "[bitter laugh]", ..., "[normal voice]"])   # 33 missing tags
# Expand T3.text_emb from (50276, 1024) to (50310, 1024)
#   rows 0..50275 copied from pretrained weights
#   rows 50276..50309 initialized N(0, 0.02)
# Same expansion applied to T3.text_head
# T3.hp.text_tokens_dict_size updated to new size
```

Verified on dry-run: `added 34 tokens; text vocab: 50276 -> 50310`.

### Reference-audio mixing (per batch item)

| Mode | Prob | Role |
|---|---|---|
| `matched_same_speaker_tagged` | 0.30 | Reference acoustically aligns with the task |
| `neutral_same_speaker` | **0.40** | **Forces model to read the tag, not copy the ref** |
| `cross_speaker` (tag-free) | 0.20 | Cross-speaker generalization |
| `self_fallback` | 0.10 | Only for singleton speakers |

The 0.40 weight on neutralized reference is the main mechanism for preventing "reference buries tag." Tunable via `--ref_mix a,b,c,d`.

### No CFG dropout

Unlike the original standard-Chatterbox plan, **we do not do CFG conditioning dropout** because Turbo's `generate()` ignores `cfg_weight` entirely. The reference-mix is the only mechanism forcing tag-conditional behavior.

### Training loop

`T3.loss(t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens) → (loss_text, loss_speech)` → sum → backward.

- **Full finetune** (all 427.4M T3 parameters trainable). Not LoRA.
- AdamW (lr 1e-4, betas (0.9, 0.95), wd 0.01)
- LinearLR warmup (500 steps) → CosineAnnealingLR (T_max = 4500)
- bf16 autocast
- grad clip 1.0
- SOT = EOT = 50256 (`<|endoftext|>`) — set explicitly at startup because `_ensure_BOT_EOT` asserts both occur ≥ B times per sequence. We prepend SOT and append EOT in the collate.
- `speech_cond_prompt_len = 375` (Turbo default).

### Checkpoint format

`{step, t3_state, optimizer_state, scheduler_state, added_tags, text_vocab_size, args}`. `run_inference.py` and `evaluate.py` both detect the expanded `text_vocab_size` and re-expand the model before `load_state_dict`.

### Sanity checks (dry run, verified)

`loss_text = 10.49`, `loss_speech = 5.88` on batch_size=2. Both below random baselines (`ln(50310) ≈ 10.83`, `ln(6563) ≈ 8.79`), confirming pretrained priors survived vocab expansion.

### Memory & throughput

T3 trainable params: **427.4M** (GPT-2 medium; smaller than standard's Llama_520M because no perceiver resampler). Frozen submodules (VE, S3Gen, S3Tokenizer) live on GPU in `eval()` with `requires_grad=False`. Batch size 4 is the recommended start; bump up if memory allows.

---

## Phase 1c — Held-out evaluation (script: `evaluate.py`)

Each eval entry runs up to 5 counterfactual conditions:

```
with_tag_matched_ref        → feed script as-is, ref is a tagged clip
with_tag_neutral_ref        → feed script as-is, ref is a tag-free clip  ← isolates text effect
without_tag_matched_ref     → strip bracketed tags, matched ref
without_tag_neutral_ref     → strip bracketed tags, neutral ref           ← baseline
shuffled_tags               → replace each trained tag with a random different one
```

Per-wav metrics:

- Prosody proxies (F0 mean/std, RMS, duration) — always computed via torchaudio
- WER via Whisper-small — optional, skipped gracefully if `whisper` is not installed
- Audio-event probs via MIT/ast — optional, skipped gracefully

### Go / no-go criteria (to claim success over baseline)

1. **Tag-presence sensitivity** — (with_tag_neutral_ref minus without_tag_neutral_ref) delta on prosody proxies / event probs must move in the tag-expected direction **and exceed the baseline's delta**.
2. **Cross-speaker generalization** — the same tag produces the expected acoustic signature across ≥3 held-out speakers.
3. **Neutrality preservation** — on `S_neutral`, WER must not degrade > 1 pp vs the baseline.

If (1) or (2) fail for a category, the conclusion is "T3-only insufficient for category X." That's Phase-2 scope, not Phase-1b.

---

## Phase 2 (deferred — sketch only)

If Phase 1c says accent/identity tags need more than T3:

- **Option A**: Train a small adapter that maps text tags → an additive offset to the speaker xvector S3Gen consumes. S3Gen weights stay frozen; only the adapter trains.
- **Option B**: Full S3Gen CFM finetune with text-tag conditioning injected into the flow-matching decoder. Larger change.

Neither is in this plan file.

---

## File inventory

```
chatterbox/
├── analysis/
│   ├── analyze_tags.py           # Phase 0 scanner
│   ├── verify_speakers.py        # CLI: extract sample wavs per speaker
│   ├── row_index.jsonl           # Flat index of all 83k rows
│   ├── tag_counts_v2.csv         # with correct per-voice-id speaker counts
│   ├── tag_per_speaker_v2.csv
│   ├── neutral_rows_per_speaker_v2.csv
│   ├── tag_tokenization.json     # 704-BPE report (kept for reference)
│   └── voice_id_stats.json
├── splits/
│   ├── tag_whitelist.json        # 40 tags in 4 categories + turbo_pretrained flags
│   ├── build_splits.py
│   ├── train.json                # 74,618 rows
│   ├── held_out_speakers.json    # 22 speakers
│   ├── eval_slices.json          # 4 slices, 2,020 entries
│   ├── speaker_to_rows.json
│   ├── neutral_refs_by_speaker.json
│   └── locale_to_speakers.json
├── checkpoints/                  # Produced by finetune.py
├── eval_runs/                    # Produced by evaluate.py
├── plans/
│   └── 2026-04-23-finetune-chatterbox-t3-for-paralinguistic-tags.md  ← this file
├── finetune.py                   # Turbo full finetune + vocab expansion
├── evaluate.py                   # Held-out counterfactual eval
├── run_inference.py              # Single-shot Turbo inference
└── README_finetune.md            # Usage / commands / full architecture diagram
```

---

## Modifications to the upstream repo

One surgical fix was required in `src/chatterbox/models/t3/t3.py`:

```python
# Before (buggy: cross_entropy with (B,T,vocab) is misinterpreted):
loss_text   = F.cross_entropy(out.text_logits,   masked_text,   ignore_index=IGNORE_ID)
loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

# After:
loss_text   = F.cross_entropy(out.text_logits.permute(0, 2, 1),   masked_text,   ignore_index=IGNORE_ID)
loss_speech = F.cross_entropy(out.speech_logits.permute(0, 2, 1), masked_speech, ignore_index=IGNORE_ID)
```

The `permute(0, 2, 1)` converts `(B, T, vocab)` → `(B, vocab, T)`, which is what `F.cross_entropy` expects for N-D inputs. The original code never triggered this path because upstream inference uses `T3.inference()` / `T3.inference_turbo()` instead of `T3.loss()`.

---

## Verification (end-to-end)

1. **Phase 0 smoke:** `analyze_tags.py` completes (observed: ~3 min). Inspect `tag_counts_v2.csv` top-40 and `voice_id_stats.json`.
2. **Phase 0 splits:** `build_splits.py` produces 74,618 train / 22 held-out speakers / 2,020 eval entries.
3. **Speaker verification:** `verify_speakers.py --list` shows 217 voice_ids; extracting 5 wavs per speaker and listening confirms identities are distinct.
4. **Phase 1a baseline:** `evaluate.py --checkpoint pretrained --slices S_tag_presence --conditions with_tag_neutral_ref without_tag_neutral_ref --max_per_slice 50`. Spot-check 5 wavs per slice.
5. **Phase 1b dry-run:** `finetune.py --dry_run --batch_size 2`. Expect `loss_speech` below `ln(6563) = 8.79` (observed 5.88).
6. **Phase 1b smoke run:** `--max_steps 100 --ckpt_every 50 --log_every 10`. Loss decreases; ckpts land in `checkpoints/`.
7. **Phase 1b full run:** `--max_steps 5000 --batch_size 4 --lr 1e-4 --warmup_steps 500 --num_workers 4`.
8. **Phase 1c eval:** `evaluate.py --checkpoint checkpoints/step_005000.pt --out_dir eval_runs/run_5k/`. Apply go/no-go criteria above.
9. **A/B listening:** `run_inference.py --checkpoint pretrained` vs `--checkpoint step_005000.pt` on the same text + ref.
10. **Pretrained vs finetuned metrics table:** compare `eval_runs/baseline/metrics.csv` and `eval_runs/run_5k/metrics.csv` for each (slice, tag, condition) triple.

---

## Explicitly out of scope

- echo-tts finetune and inference scripts.
- Using the pre-computed `semantic_codes` / `cb_*` columns (produced by a different codec).
- Finetuning S3Gen or HiFiGAN in Phase 1 (moved to Phase 2 sketch).
- Accent / identity tag experiments as **training targets** in Phase 1 (they are in the held-out eval only, for measurement).
- Multi-GPU / Accelerate / FSDP / W&B / MLflow — stdout logging only; wrap later if needed.
- CFG dropout training — irrelevant for Turbo (no CFG at inference).
- Vocab expansion for rare tags (< 50 rows) — BPE fragmentation is strictly better in that regime.
- **LoRA** — out of Phase 1. Add-on integration point documented under "Finetuning method" if ever wanted.
