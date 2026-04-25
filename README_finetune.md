# Chatterbox-Turbo T3 Finetune — Paralinguistic Tag Control

Finetunes the **T3** (text → semantic token AR LM) stage of **Chatterbox-Turbo** on `InternalCan/stage1-processed-with-audio-aligned`, teaching the model to honor inline bracketed acting directions like `[laughs]`, `[whispering]`, `[monotone]`, `[shouting]`, `[sarcastic]`, etc.

**Why Turbo (not standard Chatterbox)?** Turbo's tokenizer is GPT-2's BPE (50,257 tokens) extended with 19 paralinguistic tag tokens (IDs 50257–50275) including `[cough] [sigh] [chuckle] [laugh] [gasp] [groan] [sniff] [shush] [clear throat] [whispering] [sarcastic] [angry] [happy] [crying] [fear] [surprised] [dramatic] [narration] [advertisement]`. It was designed for exactly this task.

**7 of our 40 target tags already have dedicated single-token IDs** in Turbo's tokenizer — added by Resemble during Turbo's own pretraining. The remaining 33 target tags (`[monotone]`, `[shouting]`, `[warmly]`, `[frustrated]`, `[pause]`, etc.) are added to the tokenizer at the start of our finetune via `tokenizer.add_tokens(...)`; the text vocab grows to ~50,310 and the `T3.text_emb` / `T3.text_head` layers are expanded accordingly (first 50,276 rows inherit the pretrained weights, last 33 rows start from a small random init).

S3Gen, HiFiGAN, VoiceEncoder, and S3Tokenizer stay **frozen**. Only T3 (~427M parameters, GPT-2 medium backbone) is trained.

---

## Architecture

### Full inference pipeline (with tensor shapes)

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║              CHATTERBOX-TURBO — INFERENCE PIPELINE (with shapes)                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

 INPUTS
 ──────
 Text string "[laughs] Hello world"          Reference WAV "ref.wav"
 (any length, inline tags allowed)           (≥ 5 s, mono, any SR)
        │                                                 │
        ▼                                        ─────────┴──────────
 ┌─────────────────────────────────────────┐    │                   │
 │ transformers.AutoTokenizer              │    ▼                   ▼
 │ class GPT2Tokenizer (Byte-BPE)          │  ┌────────────────────┐  ┌──────────────────────┐
 │ base vocab = 50 257                     │  │ class VoiceEncoder │  │ class S3Tokenizer     │
 │ + 19 pretrained tag tokens  (50257-75)  │  │ voice_encoder/     │  │ s3tokenizer/          │
 │ + N new tags added in-run (50276..)     │  │ voice_encoder.py   │  │ s3tokenizer.py        │
 │ tokenizer(text) → BatchEncoding         │  │ 3-layer LSTM       │  │ (subclass of          │
 │ collate pads SOT=EOT=<|endoftext|>(50256)│  │ 40-mel input       │  │  S3TokenizerV2)       │
 │ Output: (1, T_text) int64              │  │ 16kHz audio        │  │ 128-mel, 16kHz        │
 └──────────────────────────┬─────────────┘  │ FROZEN ❄️           │  │ 25 tokens/sec         │
                            │                │ (1,T,40)→LSTM      │  │ FROZEN ❄️              │
                            │                │ →proj→L2-norm      │  │ wav→mel→quantize      │
                            │                └────────┬───────────┘  └──────────┬────────────┘
                            │                         │                          │
                            │                   (1, 256) float            (1, 375) long
                            │                   speaker_emb            cond_prompt_tokens
                            │                   [L2-normed]     [Turbo max_len=375, std=150]
                            │                         │                          │
                            │                         └──────────┬───────────────┘
                            │                           class T3Cond  (dataclass)
                            │                           models/t3/modules/cond_enc.py
                            │                              .speaker_emb    (1, 256)
                            │                              .cond_prompt     (1, 375)
                            │                              .emotion_adv     (ignored for Turbo)
                            │                              │
                            ▼                              ▼ 
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │   class T3  —  models/t3/t3.py                                                  │
 │   BACKBONE:  GPT-2 medium  (24-layer, hidden=1024, heads=16)                    │
 │              transformers.GPT2Model, loaded with config "GPT2_medium"           │
 │              AFTER LOAD: `del t3.tfmr.wte`                                      │
 │                ↑ the GPT-2 backbone has its own 50276×1024 token-emb; T3 passes │
 │                  inputs_embeds= directly, bypassing wte. Deleting it saves      │
 │                  ~200 MB of VRAM.                                               │
 │                                                                                 │
 │                     ╔══════════════════════════════════╗                        │
 │                     ║        FINETUNED HERE ←          ║                        │
 │                     ╚══════════════════════════════════╝                        │
 │                                                                                 │
 │  1. class T3CondEnc  —  models/t3/modules/cond_enc.py                           │
 │     speaker_emb   (1, 256)  → nn.Linear(256, 1024) → (1, 1, 1024)               │
 │     cond_prompt   (1, 375)  → speech_emb lookup   → (1, 375, 1024)              │
 │     (emotion_adv branch disabled — hp.emotion_adv=False)                        │
 │     concat → T3Cond_embeds  (1, L_cond, 1024)   L_cond ≈ 376                    │
 │                                                                                 │
 │  2. Text token lookup:                                                          │
 │     self.text_emb = nn.Embedding(50276+N, 1024)  ← EXPANDED at startup          │
 │       first 50276 rows copied from the pretrained Turbo checkpoint              │
 │       last N rows (new tags) init N(0, 0.02)                                    │
 │     text_emb(text_tokens) → (B, T_text, 1024)                                   │
 │                                                                                 │
 │  3. Build combined sequence:                                                    │
 │     [ T3Cond_embeds | text_embeds | speech_embeds ]                             │
 │     shapes: (B, L_cond, 1024) | (B, T_text, 1024) | (B, T_speech, 1024)         │
 │     total seq len = L_cond + T_text + T_speech                                  │
 │                                                                                 │
 │  4. GPT2Model forward (causal, 24 layers, hidden=1024)                          │
 │     input: embeddings  (B, L_total, 1024)                                       │
 │     output: hidden_states[-1]  (B, L_total, 1024)                               │
 │                                                                                 │
 │  5. Splice hidden states back:                                                  │
 │     text_latents   = hidden[L_cond : L_cond+T_text]   → (B, T_text, 1024)      │
 │     speech_latents = hidden[L_cond+T_text : ...]      → (B, T_speech, 1024)    │
 │                                                                                 │
 │  6. Projection heads  —  defined in models/t3/t3.py  lines 85-86:              │
 │     text_head   = nn.Linear(1024, 50276+N, bias=False)  ← EXPANDED at startup   │
 │        → text_logits    (B, T_text, 50276+N)                                    │
 │     speech_head = nn.Linear(1024, 6563,    bias=True)                           │
 │        → speech_logits  (B, T_speech, 6563)                                     │
 │                     (6563 = 6561 codec + SOS + EOS)                             │
 │                                                                                 │
 │  TRAINING LOSS  —  t3_next_token_loss()  in finetune.py                         │
 │    (NOT the upstream T3.loss() — that's a degenerate identity objective         │
 │    that causes catastrophic divergence; see "Training objective" below.)        │
 │                                                                                 │
 │    Build shifted speech sequences:                                              │
 │      input_speech  = [BOS, s_0, s_1, …, s_{L-1}]   (length L+1)                │
 │      target_speech = [s_0, s_1, …, s_{L-1}, EOS]   (length L+1)                │
 │    Pass input_speech to T3.forward(); compare logits[i] ↔ target[i].            │
 │    With this alignment, position i (which encodes [BOS..s_{i-1}]) is            │
 │    supervised to predict s_i — proper causal LM. Same shift for text.           │
 │                                                                                 │
 │    loss_speech = F.cross_entropy(speech_logits.permute(0,2,1), target_speech)  │
 │    loss_text   = F.cross_entropy(shifted text_logits, text_tokens[:,1:])       │
 │    loss = text_loss_weight · loss_text + loss_speech                            │
 │           (text_loss_weight default 0.1 — text is auxiliary)                    │
 │                                                                                 │
 │  INFERENCE: T3.inference_turbo()  (models/t3/t3.py  lines 416-490)              │
 │     AR-generates speech_tokens one by one with KV-cache.                        │
 │     NOTE: Turbo's inference path does NOT support CFG                           │
 │     (cfg_weight/min_p/exaggeration are ignored per tts_turbo.py:233).           │
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    │
                               speech_tokens  (1, T_speech)   long
                               values in [0, 6560] (after  < 6561 filter)
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │  class S3Gen(meanflow=True)  —  models/s3gen/s3gen.py          FROZEN ❄️         │
 │  (Turbo uses the meanflow variant; ChatterboxTurboTTS loads it via from_local)  │
 │                                                                                  │
 │  Reference prep  S3Gen.embed_ref()  in s3gen.py:                                │
 │    S3Tokenizer(ref_16k)   → prompt_token   (1, ≤375) long                        │
 │    mel(ref_24k)           → prompt_feat    (1, 80, T_mel_ref) float              │
 │    class CAMPPlus  —  models/s3gen/xvector.py                                   │
 │    CAMPPlus(ref_16k)      → xvector        (1, 80) float   ← speaker identity    │
 │                                                                                  │
 │  class UpsampleConformerEncoder  —  models/s3gen/s3gen.py:                      │
 │    speech_tokens (1, T_s) → embed → upsample → (1, T_s×upsample, 512)          │
 │                                                                                  │
 │  class CausalMaskedDiffWithXvec  —  models/s3gen/flow_matching.py:              │
 │    meanflow CFM (2 steps at inference in Turbo; cf. n_cfm_timesteps=2)          │
 │    noise (1, 80, T_mel) + speech_enc + xvec + prompt_feat                       │
 │    → mel_hat  (1, 80, T_mel)  @ 24 kHz, 80 mel channels                        │
 └──────────────────────────────────────────────────────────────────────────────────┘
                                    │
                               mel_hat  (1, 80, T_mel)
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │  class HiFTGenerator  —  models/s3gen/hifigan.py               FROZEN ❄️         │
 │  (accessed as S3Gen.mel2wav in s3gen.py)                                        │
 │                                                                                  │
 │  class ConvRNNF0Predictor  —  models/s3gen/hifigan.py:                          │
 │  F0 branch: ConvRNNF0Predictor(mel) → f0  (1, 1, T_mel)                         │
 │  upsample rates [8, 5, 3] → waveform @ 24000 Hz                                 │
 │  mel (1, 80, T_mel) + f0 → wav (1, T_wav)                                       │
 └────────────────────────────────────┬─────────────────────────────────────────────┘
                                      │
                            wav (1, T_wav) @ 24 kHz
                                      │
                             Perth watermark (implicit)    FROZEN ❄️
                                      │
                                      ▼
                             OUTPUT WAV  (1, T_wav)  @ 24 kHz
                             torchaudio.save(path, wav, 24000)
```

### How tag control actually works

```
                    TAG CONTROL FLOW (Turbo)
                    ────────────────────────

 Text "[laugh] Hello"
          │
          │  GPT-2 AutoTokenizer (50276 + N pretrained + added tags):
          │  [ '[laugh]' , 'ĠHello' ]           ← each tag is 1 TOKEN, not 6!
          │     ↑ single token ID (50275 for [laugh], or 5027X for tags
          │       we added ourselves at startup)
          ▼
         T3  ← FINETUNED
          │
          │  Learns: tag-token appears in input
          │       → predict semantic tokens encoding
          │         that acoustic behavior (laughter, monotone,
          │         whispering, …)
          │
          ▼
    speech_tokens with tag-specific semantics
          │
          ▼
    S3Gen (FROZEN) — renders those tokens in whatever
          voice the xvector specifies
          │
          ▼
    "[laugh] Hello" in the reference speaker's voice
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
                               speech_tokens = [t₁…tₙ]  ← tag-specific pattern
                                                    │
            t3_next_token_loss (in finetune.py — NOT upstream T3.loss):
              input  = [BOS, t₁, …, tₙ₋₁, tₙ]
              target = [t₁, t₂, …, tₙ, EOS]
              loss   = CE(speech_logits, target shifted by 1)
                     ↑
  T3 is trained to predict the target semantic token pattern whenever the
  matching tag token appears in its text input, regardless of speaker.
  The BOS prefix and EOS suffix teach the model how to start from a clean
  state and when to stop — both essential for inference_turbo.

  Reference audio is sampled from 4 modes per batch item:
    0.30 — matched-tag same speaker  (reference aligns with the task)
    0.40 — neutral same speaker      (forces model to read the TAG, not the ref)
    0.20 — neutral cross speaker     (tests cross-speaker generalization)
    0.10 — self (fallback)

  No CFG dropout: Turbo's generate() explicitly ignores cfg_weight / min_p /
  exaggeration (tts_turbo.py line 233). The tag has to carry its own weight
  in the AR sampling distribution — which is why the 40/30/20/10 reference
  mixing (40% neutralized ref) is especially important here.
```

---

## Directory layout

```
chatterbox/
├── analysis/
│   ├── analyze_tags.py         # Phase 0: scan dataset, output tag stats
│   ├── verify_speakers.py      # Extract sample wavs per speaker to confirm identity
│   ├── inspect_checkpoints.py  # Per-module weight-drift diagnostic for ckpts
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

### Why 40 tags for Phase 1?

The dataset contains **26,047 unique bracketed tag strings** but the distribution is extreme:

| Frequency band | # unique tags | # row occurrences | % of all tag rows |
|---|---|---|---|
| ≥ 500 rows | 77 | 57,930 | 23.7% |
| 101–500 rows | 352 | 69,730 | 28.6% |
| 21–100 rows | 1,390 | 63,599 | 26.1% |
| 6–20 rows | 1,873 | 18,737 | 7.7% |
| 2–5 rows | 6,496 | 18,140 | 7.4% |
| 1 (singletons) | 15,859 | 15,859 | 6.5% |

We picked **40 tags** from the top band by applying three filters:

1. **Frequency gate:** ≥600 row occurrences (strong signal, robust to noise).
2. **Speaker coverage:** present in ≥150 of 217 speakers (tag effect isn't a single-speaker artifact).
3. **Category match:** event / prosody / emotional only. Accent/identity tags (`[filipino accent]`, `[elderly voice]`) are deferred because T3 can't override S3Gen's xvector-driven speaker identity — those need a Phase 2 approach (see "Out of scope" below).

**Trained tag categories (40 tags total):**

| Category | Examples |
|---|---|
| event | `cough`, `clearing throat`, `sigh`, `laugh`, `chuckle`, `bitter laugh`, `belly laugh`, `pause`, `long pause` |
| prosody | `quickly`, `slowly`, `very slowly`, `very quietly`, `whispering`, `shouting`, `monotone`, `trailing off`, `mumbling`, `deep pitch`, `speeding up`, `more firmly`, `firmly` |
| emotional | `warmly`, `confidently`, `passionately`, `friendly`, `delighted`, `exhausted`, `animated`, `curious`, `playful`, `hesitantly`, `stern`, `mocking`, `excited`, `joking`, `sarcastic`, `frustrated`, `warm` |
| neutral baseline | `normal voice` |

### Extending beyond 40 — when to grow the vocabulary

You can absolutely train on more tags. The mechanism is the same (`splits/tag_whitelist.json` → `finetune.py` expands the vocab on startup), but **what you add to the vocab** and **what you train on** are separate questions.

**Rule of thumb — add a tag to the text vocab when it has enough training examples to warm up the new embedding:**

| Tag frequency | Add to vocab? | Why |
|---|---|---|
| ≥ 50 rows | ✅ Yes | Enough gradient signal to move the randomly-initialized embedding into a useful region. |
| 10–50 rows | ⚠️ Maybe | Adding helps disambiguate from BPE fragments, but the embedding may undertrain. Consider only if the tag category is important. |
| < 10 rows | ❌ No | A random dedicated embedding with 5 gradient updates is strictly worse than the BPE fragmentation, which inherits pretrained context. Let BPE handle it. |

**Staged expansion plan:**

1. **Phase 1 (this plan):** 40 high-confidence tags. Done.
2. **Phase 2:** extend whitelist to the ~429 tags with ≥100 rows — roughly 10× more coverage. All can be added to the vocab (frequency >> 50). Same pipeline: update `tag_whitelist.json`, rerun `build_splits.py`, rerun `finetune.py` which re-expands. No code changes.
3. **Phase 3:** The 1,390 tags in the 21–100 band. These are borderline for vocab expansion; consider clustering similar tags (e.g. all the `voice rising X` variants → one canonical form) before deciding. Some should stay BPE-only.
4. **Never trained:** the ~15k singletons and near-singletons — mostly one-off stage directions (`[counting on fingers out loud]`, `[words crumbling apart]`). Let them pass through as BPE fragments. They'll produce some domain-adaptation signal in loss_text without the model committing to specific token meanings.

**Note on vocab size scaling:** Expanding from 50,276 → 50,676 (400 new tags) is fine. The new rows in `text_emb` (400 × 1024 ≈ 1.6 MB) and `text_head` (400 × 1024 ≈ 1.6 MB) are negligible. Training time per step barely changes. The bottleneck would only appear if you added tens of thousands of new tokens, at which point the softmax over the text head starts to dominate.

**Key insight:** not every bracketed string needs to be a token. Adding a tag to the vocab is only useful when the model will see enough examples of it. Otherwise, BPE fragmentation is the better default — the fragments carry pretrained English context from GPT-2, which is useful even for stage-direction-like phrases the model will never confidently emit.

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
`loss_speech` near `ln(6563) ≈ 8.79` means random weights; well below that means pretrained T3 already has priors (we observed ~5.9 on a clean dry-run).

---

## Step 3 — Full training

```bash
cd ~/chatterbox && .venv/bin/python finetune.py \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --splits_dir splits/ \
    --batch_size 4 \
    --lr 2e-5 \
    --text_loss_weight 0.1 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --ref_mix 0.30,0.40,0.20,0.10 \
    --ckpt_every 500 \
    --log_every 50 \
    --num_workers 4
```

stdout is auto-tee'd to `checkpoints/training_<timestamp>.log` so the loss curve survives the run.

**Resume from a checkpoint** (e.g. continue past step 1000):

```bash
cd ~/chatterbox && .venv/bin/python finetune.py \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --splits_dir splits/ \
    --batch_size 4 --lr 2e-5 --text_loss_weight 0.1 \
    --max_steps 3000 --warmup_steps 300 --grad_clip 1.0 \
    --ckpt_every 500 --log_every 50 --num_workers 4 \
    --resume checkpoints/step_001000.pt

```

**Key flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--ref_mix a,b,c,d` | `0.30,0.40,0.20,0.10` | Ref selection: matched-tagged / neutral-same-spk / cross-speaker / self |
| `--warmup_steps` | `500` | Linear warmup before cosine decay |
| `--batch_size` | `4` | Turbo's GPT-2 medium backbone uses more VRAM than the standard Llama_520M; start at 4, bump up if memory allows |

**Note on CFG dropout:** Turbo's `generate()` explicitly ignores `cfg_weight`, `min_p`, and `exaggeration` — so there is no CFG dropout flag in this finetune. The reference-mixing ratio (40% neutralized ref) is the main mechanism for making the tag signal dominate the reference acoustics.

**Resume from checkpoint:**

```bash
.venv/bin/python finetune.py ... --resume checkpoints/step_002500.pt
```

**Checkpoint format:** `{step, t3_state, optimizer_state, scheduler_state, args}` — only `t3_state` is needed at inference.

---

```bash

mkdir -p inference/qual_check
TEXTS=(
    "[warmly] It's so good to see you again. [angry] You look wonderful."
)
for i in "test_warm_anger"; do
    .venv/bin/python run_inference.py --text "${TEXTS[$i]}" --reference_audio test-1.wav \
        --output inference/qual_check/${i}_pretrained.wav --checkpoint pretrained
    .venv/bin/python run_inference.py --text "${TEXTS[$i]}" --reference_audio test-1.wav \
        --output inference/qual_check/${i}_run2k.wav --checkpoint checkpoints/step_002000.pt
done


```

## Step 4 — Baseline capture (run BEFORE evaluating finetuned model)

```bash
.venv/bin/python evaluate.py \
    --checkpoint pretrained \
    --splits_dir splits/ \
    --dataset_path <snap> \
    --out_dir eval_runs/baseline/

cd ~/chatterbox && .venv/bin/python evaluate.py \
    --checkpoint pretrained \
    --splits_dir splits/ \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --out_dir eval_runs/baseline/ \
    --slices S_tag_presence S_cross_speaker S_neutral S_multi_tag \
    --conditions with_tag_neutral_ref without_tag_neutral_ref \
    --max_per_slice 25 \
    --seed 0

```

---

## Step 5 — Evaluate finetuned model

```bash
.venv/bin/python evaluate.py \
    --checkpoint checkpoints/step_005000.pt \
    --splits_dir splits/ \
    --dataset_path <snap> \
    --out_dir eval_runs/run_5k/

cd ~/chatterbox && .venv/bin/python evaluate.py \
    --checkpoint checkpoints/step_001000.pt \
    --splits_dir splits/ \
    --dataset_path ~/.cache/huggingface/hub/datasets--InternalCan--stage1-processed-with-audio-aligned/snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549 \
    --out_dir eval_runs/run_2k/ \
    --slices S_tag_presence S_cross_speaker S_neutral S_multi_tag \
    --conditions with_tag_neutral_ref without_tag_neutral_ref \
    --max_per_slice 25 \
    --seed 0

```

Each row in `eval_slices.json` is generated under 5 conditions:  
`with_tag+matched_ref` / `with_tag+neutral_ref` / `without_tag+matched_ref` / `without_tag+neutral_ref` / `shuffled_tags`

Metrics: WER (Whisper-small if available), per-tag audio event probs (MIT/ast model if available), F0/energy proxies.

## Compare baseline vs finetuned
Once both metrics.csv files exist:
```
cd ~/chatterbox && .venv/bin/python -c "
import pandas as pd
b = pd.read_csv('eval_runs/baseline/metrics.csv')
f = pd.read_csv('eval_runs/run_2k/metrics.csv')
b['run'] = 'baseline'; f['run'] = 'run_2k'
df = pd.concat([b, f])

# 1) Mean prosody per (slice, condition, run) — overview
agg = df.groupby(['slice','condition','run'])[['f0_mean','f0_std','rms','duration_s']].mean().round(2)
print('=== Mean prosody by slice × condition × run ===')
print(agg.to_string()); print()

# 2) Tag-presence sensitivity: with_tag_neutral_ref MINUS without_tag_neutral_ref
pres = df[df['condition'].isin(['with_tag_neutral_ref','without_tag_neutral_ref'])]
pvt = pres.pivot_table(index=['slice','tag_truth','run'], columns='condition',
                       values='f0_std', aggfunc='mean')
pvt['delta_f0std'] = pvt['with_tag_neutral_ref'] - pvt['without_tag_neutral_ref']
print('=== Per-tag F0_std delta (with_tag − without_tag), top 30 ===')
print(pvt.sort_values('delta_f0std', ascending=False).head(30).to_string())
"

```
---

## Step 6 — Inference

```bash
.venv/bin/python run_inference.py \
    --text "[laugh] Oh no, not again. [monotone] Wonderful." \
    --reference_audio /path/to/ref.wav \
    --output out.wav \
    --checkpoint checkpoints/step_005000.pt

# Pretrained Turbo baseline (no finetuning):
.venv/bin/python run_inference.py \
    --text "[laugh] Oh no, not again." \
    --reference_audio /path/to/ref.wav \
    --output baseline.wav \
    --checkpoint pretrained

cd ~/chatterbox && .venv/bin/python run_inference.py     --text "[laugh] Oh no, not again. [monotone] Fantastic."     --reference_audio test-1.wav     --output inference/sanity/step_000500.wav     --checkpoint checkpoints/step_000500.pt
```

```bash
mkdir -p inference/sweep
cd ~/chatterbox

for STEP in 000500 001000 001500 002000 002500 003000 003500 004000 004500 005000; do
    .venv/bin/python run_inference.py \
        --text "[laugh] Oh no, not again. [monotone] Fantastic." \
        --reference_audio test-1.wav \
        --output inference/sweep/step_${STEP}.wav \
        --checkpoint checkpoints/step_${STEP}.pt 2>&1 | grep -E "saved|loaded"
done

```

Output is a 24 kHz mono WAV. Turbo sampling knobs: `--temperature`, `--top_p`, `--top_k`, `--repetition_penalty`, `--norm_loudness`. CFG / exaggeration flags are intentionally omitted because Turbo's `generate()` ignores them.

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

### Training objective fix (important)

The upstream `T3.loss()` in `models/t3/t3.py:190-224` is a **degenerate identity-function objective**:
it feeds the same `speech_tokens` tensor as both the model's input AND the cross-entropy
target without any shift. In a causal transformer this is trivially solvable — `logits[i]`
can attend to (and copy) the input at position `i` — so the optimizer collapses
`speech_head` into a near-identity projection. The loss decreases, but the model loses
its ability to (a) generate from BOS, (b) emit EOS to terminate, and (c) produce coherent
autoregressive speech. Symptom: inference runs the full `max_gen_len` and emits noise.

**Our fix:** `finetune.py` defines `t3_next_token_loss()` which builds proper shifted
sequences:

```
input_speech  = [BOS, s_0, s_1, …, s_{L-1}]
target_speech = [s_0, s_1, …, s_{L-1}, EOS]
```

This is standard causal-LM next-token prediction — `logits[i]` (which encodes everything
up to position `i`) supervises `target[i]`. The training task now matches the
inference task (`T3.inference_turbo` starts from BOS and stops at EOS), and the model
actually learns to generate.

Why upstream T3.loss is broken: it appears to be an out-of-date convenience helper that
doesn't match Resemble's actual training pipeline (which is not public). The pretrained
weights work because they were trained with a correct loss internally; only the public
`T3.loss()` stub is wrong.

**Diagnostic:** the symptom is `speech_head` drifting >> the rest of the network
(e.g. 22% relative drift while `tfmr` only moved 3%). Run `analysis/inspect_checkpoints.py`
on a checkpoint dir to see this — healthy training should show drift roughly proportional
across all groups.

### Evaluation slices
22 held-out speakers never seen during training. Four slices:
- **S_tag_presence** (760 entries): same speaker, same text with and without the tag → measures "did the model learn the tag?"
- **S_cross_speaker** (760 entries): same tag across multiple held-out speakers → measures generalization
- **S_neutral** (300 entries): tag-free rows → measures baseline speech quality drift
- **S_multi_tag** (200 entries): rows with ≥2 trained tags → measures compositional behavior

---

## Troubleshooting

### Inference produces ~40 s of noise instead of ~5 s of speech

The model never emits `stop_speech_token` so `inference_turbo` runs the full `max_gen_len=1000` AR steps. This means training was using the broken upstream `T3.loss()` (see "Training objective fix" above). Make sure `finetune.py` is on a version that calls `t3_next_token_loss()`, not `model.t3.loss()`. Re-run training from a pretrained-Turbo init (delete the diverged checkpoints first).

### Eval is taking hours instead of minutes

Same root cause as above — every generation runs 1000 AR steps for diverged models. A healthy finetuned model stops at ~100–200 steps and the same eval finishes in 10–30 minutes for `--max_per_slice 25`.

### CUDA out of memory after a previous run

Suspended training processes hold GPU allocations indefinitely. Check:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -eo pid,state,cmd | grep -E '\.venv/bin/python|train\.py'
```

State `Z` = zombie (kill the parent to reap), state `T` = stopped/suspended (kill with `-9`). One-liner to nuke everything on the GPU:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

Use with caution — it kills *all* GPU processes, including any other user's runs.

### The drift report shows speech_head drifting 5×+ harder than tfmr

Catastrophic divergence — usually means the training loss is misaligned (see "Training objective fix"). A healthy run shows drift growing roughly proportionally across groups, with `text_emb`, `speech_emb`, `tfmr` all under ~3% relative by step 5000 and the heads (`text_head`, `speech_head`) under ~10%.

### Resume training picked up old broken checkpoints

If `~/chatterbox/checkpoints/` has files from multiple training runs, `--resume` will use whichever one you point at — but other broken-run checkpoints in the directory may get loaded by `evaluate.py` or `run_inference.py` if you reference them by step number. Always verify timestamps:

```bash
ls -la ~/chatterbox/checkpoints/
```

If timestamps are inconsistent (e.g. step_001000.pt is from today but step_002000.pt is from a week ago), the older ones are stale and should be moved aside before evaluating.

### Logs

Every training run auto-writes a log file to `checkpoints/training_<YYYYMMDD_HHMMSS>.log` containing every `step= … loss= …` line. After the fact, `analysis/inspect_checkpoints.py` parses these and prints the loss curve alongside the drift report.
