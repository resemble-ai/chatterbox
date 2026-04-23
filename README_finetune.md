# Chatterbox T3 Finetune — Paralinguistic Tag Control

Finetunes the **T3** (text → semantic token AR LM) stage of Chatterbox on `InternalCan/stage1-processed-with-audio-aligned`, teaching the model to honor inline bracketed acting directions like `[laughs]`, `[whispering]`, `[monotone]`, `[shouting]`, `[sarcastic]`, etc.

S3Gen, HiFiGAN, VoiceEncoder, and S3Tokenizer stay **frozen**. Only T3 (~532M parameters, Llama-based) is trained.

---

## Architecture

### Full inference pipeline (with tensor shapes)

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                     CHATTERBOX TTS — INFERENCE PIPELINE (with shapes)               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

 INPUTS
 ──────
 Text string "[laughs] Hello world"          Reference WAV "ref.wav"
 (any length, inline tags allowed)           (mono, any SR → librosa-resampled)
        │                                                 │
        ▼                                        ─────────┴──────────
 ┌─────────────────────────────────────────┐    │                   │
 │ class EnTokenizer                       │    ▼                   ▼
 │ models/tokenizers/tokenizer.py         │  ┌────────────────────┐  ┌──────────────────────┐
 │ BPE vocab=704, spaces→[SPACE]          │  │ class VoiceEncoder │  │ class S3Tokenizer     │
 │ tokenizer.encode() → ids               │  │ voice_encoder/     │  │ s3tokenizer/          │
 │                                        │  │ voice_encoder.py   │  │ s3tokenizer.py        │
 │ Output: (1, T_text) int32              │  │ 3-layer LSTM       │  │ (subclass of          │
 │ T_text = len(BPE(text)) + 2 (SOT+EOT) │  │ 40-mel input       │  │  S3TokenizerV2)       │
 └──────────────────────────┬─────────────┘  │ 16kHz audio        │  │ 128-mel, 16kHz        │
                            │                │ FROZEN ❄️           │  │ 25 tokens/sec         │
                            │                │ (1,T,40)→LSTM      │  │ FROZEN ❄️              │
                            │                │ →proj→L2-norm      │  │ wav→mel→quantize      │
                            │                └────────┬───────────┘  └──────────┬────────────┘
                            │                         │                          │
                            │                   (1, 256) float            (1, 150) long
                            │                   speaker_emb            cond_prompt_tokens
                            │                   [L2-normed]             [max_len=150]
                            │                         │                          │
                            │                         └──────────┬───────────────┘
                            │                           class T3Cond  (dataclass)
                            │                           models/t3/modules/cond_enc.py
                            │                              .speaker_emb    (1, 256)
                            │                              .cond_prompt     (1, 150)
                            │                              .emotion_adv     (1, 1, 1) = 0.5
                            ▼                              │
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │   class T3  —  models/t3/t3.py                                                  │
 │   Llama_520M backbone (24-layer, hidden=520)                                    │
 │                     ╔══════════════════════════════════╗                        │
 │                     ║        FINETUNED HERE ←          ║                        │
 │                     ╚══════════════════════════════════╝                        │
 │                                                                                 │
 │  1. class T3CondEnc  —  models/t3/modules/cond_enc.py                           │
 │     speaker_emb   (1, 256) → nn.Linear(256, 520) → (1, 1, 520)                 │
 │     cond_prompt   (1, 150) → speech_emb lookup  → (1, 150, 520)                │
 │     emotion_adv   (1, 1, 1)→ nn.Linear(1, 520)  → (1, 1, 520)                 │
 │     concat → T3Cond_embeds  (1, L_cond, 520)   L_cond ≈ 152                    │
 │                                                                                 │
 │  2. Build combined sequence:                                                    │
 │     [ T3Cond_embeds | text_embeds | speech_embeds ]                             │
 │     shapes: (B, L_cond, 520) | (B, T_text, 520) | (B, T_speech, 520)           │
 │     total seq len = L_cond + T_text + T_speech                                  │
 │                                                                                 │
 │  3. LlamaModel  —  transformers.LlamaModel  (causal, 24 layers, hidden=520)     │
 │     input: embeddings  (B, L_total, 520)                                        │
 │     output: hidden_states[-1]  (B, L_total, 520)                                │
 │                                                                                 │
 │  4. Splice hidden states back:                                                  │
 │     text_latents   = hidden[L_cond : L_cond+T_text]   → (B, T_text, 520)       │
 │     speech_latents = hidden[L_cond+T_text : ...]      → (B, T_speech, 520)     │
 │                                                                                 │
 │  5. Projection heads  —  defined in models/t3/t3.py  lines 85-86:              │
 │     text_head   = nn.Linear(520, 704)  → text_logits   (B, T_text,   704)      │
 │     speech_head = nn.Linear(520, 8194) → speech_logits (B, T_speech, 8194)     │
 │                                     (8194 = 6561 codec + SOS + EOS + padding)  │
 │                                                                                 │
 │  TRAINING LOSS  —  T3.loss()  in models/t3/t3.py  lines 190-224:               │
 │    mask = arange(T) >= token_lens   →  (B, T) bool                             │
 │    masked_text   = text_tokens.masked_fill(mask, -100)   (B, T_text)           │
 │    masked_speech = speech_tokens.masked_fill(mask, -100) (B, T_speech)         │
 │                                                                                 │
 │    text_logits  (B, T_text,   704) → .permute(0,2,1) → (B, 704,   T_text)     │
 │    speech_logits(B, T_speech, 8194)→ .permute(0,2,1) → (B, 8194, T_speech)    │
 │        ↑ permute needed because F.cross_entropy expects (B, C, T) not (B,T,C) │
 │                                                                                 │
 │    loss_text   = F.cross_entropy(text_logits.permute(0,2,1),   masked_text)   │
 │    loss_speech = F.cross_entropy(speech_logits.permute(0,2,1), masked_speech) │
 │    loss = loss_text + loss_speech                                               │
 │                                                                                 │
 │  INFERENCE: T3.inference() AR-generates speech_tokens one by one               │
 │     using KV-cache; CFG doubles the batch (cond + uncond pass)                 │
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    │
                               speech_tokens  (1, T_speech)   long
                               values in [0, 6560] after drop_invalid_tokens()
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                    S3Gen  —  CausalMaskedDiffWithXvec         FROZEN ❄️           │
 │                    file: s3gen.safetensors                                      │
 │                                                                                  │
 │  Reference prep  S3Gen.embed_ref(ref_wav, S3GEN_SR=24000):                      │
 │    S3GEN_SR → prompt_token via S3Tokenizer (1, ≤150) long                       │
 │    mel(ref_24k) → prompt_feat              (1, 80, T_mel_ref) float              │
 │    CAMPPlus(ref_16k) → xvector / embedding (1, 80) float                        │
 │                                                                                  │
 │  UpsampleConformerEncoder:                                                       │
 │    speech_tokens (1, T_s) → embedding → upsample → (1, T_s*upsample, 512)      │
 │                                                                                  │
 │  ConditionalDecoder  (CFM ODE, 2–4 steps):                                      │
 │    noise (1, 80, T_mel) + speech_enc + xvec + prompt_feat                       │
 │    → mel_hat  (1, 80, T_mel)  @ 24 kHz, 80 mel channels                        │
 └──────────────────────────────────────────────────────────────────────────────────┘
                                    │
                               mel_hat  (1, 80, T_mel)
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │             HiFTGenerator (HiFiGAN + F0 predictor)              FROZEN ❄️         │
 │             file: inside s3gen.safetensors (mel2wav submodule)                  │
 │                                                                                  │
 │  F0 branch: ConvRNNF0Predictor(mel) → f0  (1, 1, T_mel)                         │
 │  upsample rates [8, 5, 3] → waveform @ 24000 Hz                                 │
 │  mel (1, 80, T_mel) + f0 → wav (1, T_wav)                                       │
 └────────────────────────────────────┬─────────────────────────────────────────────┘
                                      │
                            wav (1, T_wav) @ 24kHz
                                      │
                             Perth watermark (implicit)    FROZEN ❄️
                                      │
                                      ▼
                             OUTPUT WAV  (1, T_wav)  @ 24 kHz
                             torchaudio.save(path, wav, 24000)
```

### How tag control actually works

```
                    TAG CONTROL FLOW
                    ────────────────

 Text "[laughs] Hello"
          │
          │  EnTokenizer BPE:
          │  [ '[' 'l' 'a' 'u' 'gh' ']' 'H' 'el' 'lo' ]
          │   ←── tag ──→                ←── word ──→
          ▼
         T3  ← FINETUNED
          │
          │  Learns: "[laughs]" tokens in input
          │       → predict semantic tokens encoding
          │         laughter (pitch modulation, rhythmic
          │         breaks, energy bursts)
          │
          ▼
    speech_tokens with laugh-pattern semantics
          │
          ▼
    S3Gen (FROZEN) — renders those tokens in whatever
          voice the xvector specifies
          │
          ▼
    "[laughs] Hello" in the reference speaker's voice
```

**Why T3-only is enough for event/prosody tags:**
S3Gen uses an `xvector` from the reference audio for speaker identity — it is reference-driven, not text-driven. T3 only needs to learn "which semantic tokens correspond to `[laughs]` in the text". S3Gen renders those tokens in any target voice.

**Why T3-only is NOT enough for accent tags:**
`[filipino accent]` would require changing the xvector (speaker identity signal) or injecting accent conditioning at the S3Gen level. The text in T3 cannot override the reference-audio xvector that S3Gen uses for rendering.

### Training setup

```
  Each batch item: (script_content_with_tags, target_wav, ref_wav)
                                                    │
                               S3Tokenizer(target_wav)  ← FROZEN on GPU
                                                    │
                               speech_tokens = [t₁…tₙ]  ← "laugh pattern"
                                                    │
  T3.loss: CE(T3_predicted_speech_tokens, speech_tokens)
                     ↑
  T3 is trained to predict the "laugh pattern" semantic tokens
  whenever "[laughs]" appears in its text input, regardless of speaker.

  Reference audio is sampled from 4 modes per batch item:
    0.30 — matched-tag same speaker  (reference aligns with the task)
    0.40 — neutral same speaker      (forces model to read the TAG, not the ref)
    0.20 — neutral cross speaker     (tests cross-speaker generalization)
    0.10 — self (fallback)

  CFG dropout (p=0.10): 10% of the time speaker conditioning is zeroed.
  Amplifies the text/tag signal over reference acoustics at inference
  when cfg_weight > 0 (default 0.7 post-finetune).
```

---

## Directory layout

```
chatterbox/
├── analysis/
│   ├── analyze_tags.py         # Phase 0: scan dataset, output tag stats
│   ├── verify_speakers.py      # Extract sample wavs per speaker to confirm identity
│   ├── row_index.jsonl         # Flat index of all 83k rows (voice_id, locale, tags, …)
│   ├── tag_counts_v2.csv       # Tag → n_rows, n_speakers, mean_duration
│   ├── tag_per_speaker_v2.csv  # (voice_id, tag) → n_rows
│   ├── tag_tokenization.json   # EnTokenizer output for each tag
│   └── voice_id_stats.json     # Speaker/locale/gender distribution
├── splits/
│   ├── tag_whitelist.json      # 40 curated tags in 4 categories
│   ├── build_splits.py         # Builds train/eval splits from row_index
│   ├── train.json              # 74,618 training row indices (195/217 speakers)
│   ├── held_out_speakers.json  # 22 speakers withheld for eval
│   ├── eval_slices.json        # 2,020 structured eval entries
│   ├── speaker_to_rows.json
│   ├── neutral_refs_by_speaker.json
│   └── locale_to_speakers.json
├── checkpoints/                # Saved during training
├── eval_runs/                  # Generated wavs + metrics per eval run
├── plans/
│   └── 2026-04-23-finetune-chatterbox-t3-for-paralinguistic-tags.md
├── finetune.py                 # Main training script
├── evaluate.py                 # Held-out eval with counterfactual conditions
└── run_inference.py            # Single-shot inference CLI
```

---

## Dataset

**Source:** `InternalCan/stage1-processed-with-audio-aligned`  
**Snapshot:** `~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549`

**Key facts:**
- 83,344 rows, 217 distinct speakers (`voice_id` = 13-char hex prefix in filename), 27 English locales
- ~400 rows per speaker; ~20 neutral (tag-free) rows per speaker
- Tags are inline in `script_content`: `[tag] text [tag] more text`
- 26k unique bracketed tag strings in the wild; ~429 with ≥101 occurrences

**Trained tag categories (40 tags total):**

| Category | Examples |
|---|---|
| event | `cough`, `clearing throat`, `sigh`, `laugh`, `chuckle`, `bitter laugh`, `belly laugh`, `pause`, `long pause` |
| prosody | `quickly`, `slowly`, `very slowly`, `very quietly`, `whispering`, `shouting`, `monotone`, `trailing off`, `mumbling`, `deep pitch`, `speeding up`, `more firmly`, `firmly` |
| emotional | `warmly`, `confidently`, `passionately`, `friendly`, `delighted`, `exhausted`, `animated`, `curious`, `playful`, `hesitantly`, `stern`, `mocking`, `excited`, `joking`, `sarcastic`, `frustrated`, `warm` |
| neutral baseline | `normal voice` |

Accent/identity tags (`[filipino accent]`, `[elderly voice]`, etc.) are **deferred** — T3-only training is insufficient for those.

---

## Step 0 — Explore speakers (optional but recommended)

```bash
# List all 217 speakers with locale and row count
.venv/bin/python analysis/verify_speakers.py --list

# Extract 5 sample wavs from a specific speaker (to verify identity)
.venv/bin/python analysis/verify_speakers.py --voice_id 6931c586ddeae --out_dir /tmp/check/

# By locale
.venv/bin/python analysis/verify_speakers.py --locale english_british --out_dir /tmp/check/
.venv/bin/python analysis/verify_speakers.py --locale english_australian --out_dir /tmp/check/
.venv/bin/python analysis/verify_speakers.py --locale english_indian-india --out_dir /tmp/check/

# By locale + gender
.venv/bin/python analysis/verify_speakers.py --locale english_australian --gender female --out_dir /tmp/check/
```

---

## Step 1 — Build splits (already done; re-run if needed)

```bash
.venv/bin/python splits/build_splits.py \
    --analysis_dir analysis/ \
    --splits_dir splits/ \
    --held_out_frac 0.10 \
    --seed 42
```

Produces `train.json` (74,618 rows), `held_out_speakers.json` (22 speakers), `eval_slices.json`.

---

## Step 2 — Sanity check (dry run)

```bash
.venv/bin/python finetune.py \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --splits_dir splits/ \
    --batch_size 4 --max_steps 1 \
    --num_workers 0 --dry_run
```

Expected output: `step=0  loss=<finite>  (text=... speech=...)  — OK`  
`loss_speech` near `ln(6561) ≈ 8.8` means random weights; well below that means pretrained T3 already has priors.

---

## Step 3 — Full training

```bash
.venv/bin/python finetune.py \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --splits_dir splits/ \
    --batch_size 8 \
    --lr 1e-4 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --ref_mix 0.30,0.40,0.20,0.10 \
    --p_uncond 0.10 \
    --ckpt_every 500 \
    --log_every 50 \
    --num_workers 4
```

**Key flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--ref_mix a,b,c,d` | `0.30,0.40,0.20,0.10` | Ref selection: matched-tagged / neutral-same-spk / cross-speaker / self |
| `--p_uncond` | `0.10` | CFG dropout probability (zeroes speaker conditioning) |
| `--warmup_steps` | `500` | Linear warmup before cosine decay |

**Resume from checkpoint:**

```bash
.venv/bin/python finetune.py ... --resume checkpoints/step_002500.pt
```

**Checkpoint format:** `{step, t3_state, optimizer_state, scheduler_state, args}` — only `t3_state` is needed at inference.

---

## Step 4 — Baseline capture (run BEFORE evaluating finetuned model)

```bash
.venv/bin/python evaluate.py \
    --checkpoint pretrained \
    --splits_dir splits/ \
    --dataset_path <snap> \
    --out_dir eval_runs/baseline/
```

---

## Step 5 — Evaluate finetuned model

```bash
.venv/bin/python evaluate.py \
    --checkpoint checkpoints/step_005000.pt \
    --splits_dir splits/ \
    --dataset_path <snap> \
    --out_dir eval_runs/run_5k/
```

Each row in `eval_slices.json` is generated under 5 conditions:  
`with_tag+matched_ref` / `with_tag+neutral_ref` / `without_tag+matched_ref` / `without_tag+neutral_ref` / `shuffled_tags`

Metrics: WER (Whisper-small if available), per-tag audio event probs (MIT/ast model if available), F0/energy proxies.

---

## Step 6 — Inference

```bash
.venv/bin/python run_inference.py \
    --text "[laughs] Oh no, not again. [deadpan] Wonderful." \
    --reference_audio /path/to/ref.wav \
    --output out.wav \
    --checkpoint checkpoints/step_005000.pt

# Pretrained baseline (no finetuning):
.venv/bin/python run_inference.py \
    --text "[laughs] Oh no, not again." \
    --reference_audio /path/to/ref.wav \
    --output baseline.wav \
    --checkpoint pretrained
```

Output is a 24 kHz mono WAV. `cfg_weight` default raised to 0.7 post-finetune (CFG dropout training makes the conditional/unconditional contrast more informative; higher `cfg_weight` amplifies the text/tag signal over the reference acoustics).

---

## Design notes

### Why T3 only?
T3 is the text → semantic token AR LM. Semantic tokens carry phonetic/prosodic content. S3Gen (flow matching, semantic → mel) uses speaker xvector from the reference audio for identity — it stays frozen and can render T3's tokens faithfully for any voice. T3-only works for event and prosody tags. Accent/identity tags require S3Gen-level intervention (Phase 2, not in this plan).

### Reference mixing strategy
The training DataLoader picks the reference audio in 4 modes:

| Mode | Prob | What it does |
|---|---|---|
| `matched_tagged` | 0.30 | Different clip, same speaker, same tag(s). Acoustic reference aligns with the task. |
| `neutral_same_spk` | 0.40 | Tag-free clip from same speaker. Forces model to extract tag behavior from text. |
| `cross_speaker` | 0.20 | Tag-free clip from a different speaker. Tests cross-speaker generalization. |
| `self` | 0.10 | Same clip as target (fallback only). |

The 0.40 neutral weight is critical: it prevents the model from learning "reference acoustics → same output," which would let it ignore the text tags entirely.

### CFG dropout
10% of the time during training, speaker conditioning is zeroed (speaker_emb → 0, cond_prompt → EOS). This aligns with how `ChatterboxTTS.generate()` dispatches a CFG path at inference (`cfg_weight > 0`). After finetuning, the default `cfg_weight=0.7` (higher than the pretrained default of 0.5) will more strongly amplify the text — and therefore the inline tags — over the reference acoustic signal.

### Evaluation slices
22 held-out speakers never seen during training. Four slices:
- **S_tag_presence** (760 entries): same speaker, same text with and without the tag → measures "did the model learn the tag?"
- **S_cross_speaker** (760 entries): same tag across multiple held-out speakers → measures generalization
- **S_neutral** (300 entries): tag-free rows → measures baseline speech quality drift
- **S_multi_tag** (200 entries): rows with ≥2 trained tags → measures compositional behavior
