# Plan: A/B Inference Script for Paralinguistic Tag Tokens

## Context

We verified that Chatterbox ships two distinct T3 variants with completely different tokenizer vocabularies for paralinguistic tags:

- **Standard** (`ResembleAI/chatterbox`, 704-token BPE + Llama 520M backbone): 49 "added" tokens at IDs 604+, heavily skewed toward **sound-effect** tokens — `[bark]`, `[howl]`, `[meow]`, `[sneeze]`, `[snore]`, `[chew]`, `[sip]`, `[kiss]`, `[whistle]`, etc. Has `[whisper]` (not `[whispering]`) and `[laughter]` (not `[laugh]`).
- **Turbo** (`ResembleAI/chatterbox-turbo`, GPT-2 50,257 + 19 tags + GPT2 medium backbone): 19 tokens at IDs 50257–50275, all **emotion/style** tokens — `[angry]`, `[fear]`, `[surprised]`, `[whispering]`, `[dramatic]`, `[narration]`, `[crying]`, `[happy]`, `[sarcastic]`, `[sigh]`, `[cough]`, `[groan]`, `[sniff]`, `[gasp]`, `[chuckle]`, `[laugh]`, `[clear throat]`, `[shush]`, `[advertisement]`.

We don't yet know empirically how each model behaves on tags that are (a) native to its vocab, (b) present-but-renamed in the other model, or (c) completely out-of-vocab. A side-by-side listening test is the fastest way to learn before committing a finetune run to either backbone. The existing `finetune.py` currently loads Standard (via `ChatterboxTTS.from_pretrained`), while the `README_finetune.md` describes Turbo's vocabulary as "Resemble's exact approach" — this mismatch is the decision we're trying to resolve.

## Goal

One script, `test_tags_ab.py`, that:

1. Loads both `ChatterboxTTS` (standard) and `ChatterboxTurboTTS` (turbo) in the same process.
2. For each tag in a curated taxonomy, generates audio with **both** models using the same carrier sentence and the models' bundled default voices (no `audio_prompt_path`).
3. Logs tokenization diagnostics per (model, tag) pair to a text report, so we can see at-a-glance whether the tag survived as a single token or got char-split.
4. Saves paired wavs into an organized directory tree for easy A/B listening.

No finetune changes. Read-only use of the current models — inference only.

## Design

### File: `/home/sruthi/chatterbox/test_tags_ab.py`

### Tag taxonomy (3 groups, grouped for the report)

```python
TURBO_NATIVE = [  # IDs 50257–50275 in turbo; NOT in standard
    "angry", "fear", "surprised", "whispering", "dramatic",
    "narration", "crying", "happy", "sarcastic", "sigh",
    "cough", "groan", "sniff", "gasp", "chuckle", "laugh",
    "clear throat", "shush",  # skip [advertisement] — not useful for listening
]

STANDARD_NATIVE = [  # IDs 604+ in standard; NOT in turbo
    "bark", "howl", "meow", "sneeze", "snore",
    "chew", "sip", "kiss", "whistle", "hum",
    "giggle", "guffaw", "cry", "mumble",
]

SHARED_BUT_RENAMED = [  # Same concept, different surface form — test both forms
    # (concept_label, turbo_tag, standard_tag)
    ("whisper",    "whispering", "whisper"),
    ("laugh",      "laugh",      "laughter"),
    ("sigh",       "sigh",       "sigh"),       # exact match in both
    ("cough",      "cough",      "cough"),      # exact match
    ("sniff",      "sniff",      "sniff"),      # exact match
    ("gasp",       "gasp",       "gasp"),       # exact match
    ("groan",      "groan",      "groan"),      # exact match
]
```

For each tag the script will cross-test — i.e. Turbo's native tags also get run through Standard (to hear the OOV breakdown), and vice versa.

### Carrier sentence strategy

Each tag gets a short sentence where it naturally cues behavior, e.g.:
- `"[angry] I cannot believe you did that again."`
- `"She paused and then [sigh] continued reading the letter."`
- `"The dog ran outside and [bark] chased the mailman."`

A `CARRIER_SENTENCES: dict[str, str]` literal at the top of the script — keyed by tag concept, roughly 15–20 words each. Include one **no-tag baseline** sentence per voice so we can calibrate the default speaking style.

### Tokenization diagnostic

Before each `generate()` call, encode the prompt with that model's tokenizer and log:

```
turbo  [angry] → ids=[..., 50257, ...]  (1 token  ✓ single ID)
std    [angry] → ids=[..., 5, 14, 27, 23, 20, 34, 6, ...]  (7 tokens  ✗ char-split)
```

- Standard: `model.tokenizer.encode(text)` returns a list of ints.
- Turbo: `model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()`.

Also log the total token count so we know when a tag got mangled. Write everything to `outputs/tag_ab/tokenization_report.txt`.

### Output layout

```
outputs/tag_ab/
├── tokenization_report.txt        # full per-(model,tag) encoding log
├── README.md                      # generated: what each file tests
├── baseline/
│   ├── turbo_no_tag.wav
│   └── standard_no_tag.wav
├── turbo_native/                  # tags native to turbo
│   ├── angry__turbo.wav
│   ├── angry__standard.wav       # cross-test: does std react?
│   ├── sarcastic__turbo.wav
│   ├── sarcastic__standard.wav
│   └── ...
├── standard_native/               # tags native to standard
│   ├── bark__turbo.wav
│   ├── bark__standard.wav
│   └── ...
└── shared_renamed/                # same concept, different names
    ├── whisper__turbo[whispering].wav
    ├── whisper__standard[whisper].wav
    └── ...
```

Filenames encode (concept)__(model)[actual_tag_fed].wav so listeners can match pairs.

### Skeleton

```python
# /home/sruthi/chatterbox/test_tags_ab.py
import json
from pathlib import Path
import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

OUT = Path("outputs/tag_ab")
device = "cuda" if torch.cuda.is_available() else "cpu"

CARRIER_SENTENCES = { ... }
TURBO_NATIVE = [ ... ]
STANDARD_NATIVE = [ ... ]
SHARED_BUT_RENAMED = [ ... ]

def encode_std(model, text):   return model.tokenizer.encode(text)
def encode_turbo(model, text): return model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

def run_one(model, encode_fn, text, out_path, log_file, label):
    ids = encode_fn(model, text)
    log_file.write(f"{label}  {text!r}\n  ids={ids}  (n_tokens={len(ids)})\n\n")
    with torch.no_grad():
        wav = model.generate(text)  # uses bundled conds.pt — no audio_prompt_path
    ta.save(str(out_path), wav.cpu(), model.sr)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline").mkdir(exist_ok=True)
    (OUT / "turbo_native").mkdir(exist_ok=True)
    (OUT / "standard_native").mkdir(exist_ok=True)
    (OUT / "shared_renamed").mkdir(exist_ok=True)

    print("Loading standard ...")
    std = ChatterboxTTS.from_pretrained(device=device)
    print("Loading turbo ...")
    turbo = ChatterboxTurboTTS.from_pretrained(device=device)

    log = open(OUT / "tokenization_report.txt", "w")
    # baseline
    run_one(std,   encode_std,   CARRIER_SENTENCES["_baseline"], OUT / "baseline/standard_no_tag.wav", log, "std   baseline")
    run_one(turbo, encode_turbo, CARRIER_SENTENCES["_baseline"], OUT / "baseline/turbo_no_tag.wav",    log, "turbo baseline")

    # turbo-native tags × both models
    for tag in TURBO_NATIVE:
        text = CARRIER_SENTENCES[tag]
        run_one(turbo, encode_turbo, text, OUT / f"turbo_native/{slug(tag)}__turbo.wav",    log, f"turbo [{tag}]")
        run_one(std,   encode_std,   text, OUT / f"turbo_native/{slug(tag)}__standard.wav", log, f"std   [{tag}]")

    # standard-native tags × both models
    for tag in STANDARD_NATIVE:
        text = CARRIER_SENTENCES[tag]
        run_one(turbo, encode_turbo, text, OUT / f"standard_native/{slug(tag)}__turbo.wav",    log, f"turbo [{tag}]")
        run_one(std,   encode_std,   text, OUT / f"standard_native/{slug(tag)}__standard.wav", log, f"std   [{tag}]")

    # shared-but-renamed
    for concept, turbo_tag, std_tag in SHARED_BUT_RENAMED:
        text_t = CARRIER_SENTENCES[concept].replace("{TAG}", f"[{turbo_tag}]")
        text_s = CARRIER_SENTENCES[concept].replace("{TAG}", f"[{std_tag}]")
        run_one(turbo, encode_turbo, text_t, OUT / f"shared_renamed/{concept}__turbo[{turbo_tag}].wav", log, f"turbo [{turbo_tag}]")
        run_one(std,   encode_std,   text_s, OUT / f"shared_renamed/{concept}__standard[{std_tag}].wav", log, f"std   [{std_tag}]")

    log.close()
    write_readme(OUT)
    print(f"Done. See {OUT}/")

if __name__ == "__main__":
    main()
```

### Reused code & reference files

- `ChatterboxTTS.from_pretrained` — [src/chatterbox/tts.py:168-180](src/chatterbox/tts.py#L168-L180)
- `ChatterboxTurboTTS.from_pretrained` — [src/chatterbox/tts_turbo.py:186-202](src/chatterbox/tts_turbo.py#L186-L202)
- `ChatterboxTTS.generate` — [src/chatterbox/tts.py:208-218](src/chatterbox/tts.py#L208-L218)
- `ChatterboxTurboTTS.generate` — [src/chatterbox/tts_turbo.py:248-260](src/chatterbox/tts_turbo.py#L248-L260)
- Pattern for bundled-voice inference (no `audio_prompt_path`) — [example_tts_turbo.py:15](example_tts_turbo.py#L15)
- Pattern for saving output — [example_tts.py:42](example_tts.py#L42)
- Both models expose `.sr = 24000` and return `(1, N)` torch tensors → feed straight to `torchaudio.save`.

## Verification

After implementation:

1. `cd /home/sruthi/chatterbox && uv run python test_tags_ab.py` — should complete without errors and produce ~75–85 wav files under `outputs/tag_ab/`.
2. Inspect `outputs/tag_ab/tokenization_report.txt` — confirm:
   - Turbo native tags encode as **1 token** in turbo, **many tokens** (char-split) in standard.
   - Standard native tags encode as **1 token** in standard, **many tokens** in turbo.
   - Shared tokens encode as 1 token in whichever model owns that exact spelling.
3. Spot-listen 3 pairs:
   - `turbo_native/angry__turbo.wav` vs `turbo_native/angry__standard.wav` (expect turbo to sound angry; std to sound neutral-or-weird)
   - `standard_native/bark__standard.wav` vs `standard_native/bark__turbo.wav` (expect std to actually produce a bark-like sound)
   - `shared_renamed/whisper__turbo[whispering].wav` vs `shared_renamed/whisper__standard[whisper].wav`
4. If results show Turbo winning on emotion tags (as hypothesized), that empirically supports switching `finetune.py:380` from `ChatterboxTTS.from_pretrained` to `ChatterboxTurboTTS.from_pretrained` before the real training run.

## Out of scope

- No finetuning changes (`finetune.py` stays as-is).
- No tokenizer editing (no new tokens added to either model).
- No voice-cloning experiments (default voice only).
- No automated perceptual evaluation — this is a human-listening A/B.
