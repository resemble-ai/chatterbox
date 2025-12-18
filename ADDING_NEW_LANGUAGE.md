# Adding a New Language to Chatterbox

This guide explains how to add a new language to the Chatterbox multilingual TTS model.

## Overview

Chatterbox's multilingual support is built on three key components:
1. **MTLTokenizer**: Handles language-specific text preprocessing and tokenization
2. **T3 Model**: The text-to-speech token generator (trained on 23+ languages)
3. **S3Gen Model**: Converts speech tokens to audio

To add a new language, you need to:
1. Prepare your language data
2. Create/update the tokenizer with your language's vocabulary
3. Fine-tune or adapt the models (or use existing multilingual model)
4. Update the supported languages list
5. Implement language-specific preprocessing if needed

---

## Step 1: Data Preparation

### Requirements

You'll need high-quality TTS training data for your language:

- **Format**: `.wav` files (16kHz sample rate recommended for tokenization, 44.1kHz for S3Gen)
- **Text**: Corresponding text transcriptions
- **Quantity**: At least 10-50 hours of audio is recommended for fine-tuning
- **Quality**: Clear audio with minimal background noise
- **Diversity**: Multiple speakers, various prosodies and phonetic contexts

### Data Organization

```
your_language_data/
├── audio/
│   ├── speaker1_001.wav
│   ├── speaker1_002.wav
│   └── ...
└── transcriptions.txt  (one text per line, matching audio files)
```

---

## Step 2: Update the Tokenizer

The tokenizer handles language-specific text preprocessing. Edit [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py):

### 2a. Add Language-Specific Preprocessing

If your language needs special handling (like Japanese kanji→hiragana conversion), add a preprocessing function:

```python
def your_language_normalize(text: str) -> str:
    """
    Language-specific text normalization for [YOUR LANGUAGE].
    
    This function handles:
    - Character normalization
    - Stress marks (if applicable)
    - Diacritics
    - Script conversion if needed
    """
    # Your preprocessing logic here
    return normalized_text
```

**Examples in codebase:**
- **Chinese**: `ChineseCangjieConverter` - Converts Chinese characters to Cangjie codes
- **Japanese**: `hiragana_normalize()` - Converts kanji to hiragana
- **Russian**: `add_russian_stress()` - Adds stress marks to Russian text
- **Korean**: `korean_normalize()` - Handles Korean text normalization
- **Hebrew**: `add_hebrew_diacritics()` - Adds diacritical marks

### 2b. Register Your Language in MTLTokenizer.encode()

In [mtl_tts.py](src/chatterbox/mtl_tts.py), update the `MTLTokenizer.encode()` method to use your preprocessing:

```python
def encode(self, txt: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
    txt = self.preprocess_text(txt, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
    
    # Language-specific text processing
    if language_id == 'zh':
        txt = self.cangjie_converter(txt)
    elif language_id == 'ja':
        txt = hiragana_normalize(txt)
    # ... ADD YOUR LANGUAGE HERE:
    elif language_id == 'YOUR_LANG_CODE':
        txt = your_language_normalize(txt)
    
    # Prepend language token
    if language_id:
        txt = f"[{language_id.lower()}]{txt}"
    
    txt = txt.replace(' ', SPACE)
    return self.tokenizer.encode(txt).ids
```

### 2c. Build/Update Tokenizer Vocabulary

You need a BPE tokenizer vocabulary file (JSON format) that includes:
- All graphemes from your language
- Special tokens: `[START]`, `[STOP]`, `[UNK]`, `[SPACE]`, `[PAD]`, `[SEP]`, `[CLS]`, `[MASK]`
- Language tokens: `[language_code]` for each language
- BPE merges for subword units

**Using Hugging Face tokenizers library:**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create a BPE tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=10000, special_tokens=[
    "[START]", "[STOP]", "[UNK]", "[SPACE]", 
    "[PAD]", "[SEP]", "[CLS]", "[MASK]",
    "[en]", "[fr]", "[YOUR_LANG_CODE]"  # Add language tokens
])

tokenizer.pre_tokenizer = Whitespace()

# Train on your text data
files = ["path/to/your_language_texts.txt"]
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("grapheme_mtl_merged_expanded_v1.json")
```

---

## Step 3: Update Supported Languages

### 3a. Add to SUPPORTED_LANGUAGES Dictionary

Edit [src/chatterbox/mtl_tts.py](src/chatterbox/mtl_tts.py):

```python
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  # ... existing languages ...
  "YOUR_LANG_CODE": "Your Language Name",  # ADD HERE
}
```

Language code conventions:
- Use ISO 639-1 (2-letter) codes: `en` (English), `fr` (French), `de` (German), etc.
- See: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes

---

## Step 4: Model Training/Fine-tuning

### Option A: Fine-tune Existing Multilingual Model (Recommended for Similar Languages)

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config

# Load pre-trained multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Your language-specific fine-tuning code here
# (This requires PyTorch Lightning or custom training loop)
# Dataset preparation, loss computation, optimizer setup, etc.

# Save the fine-tuned model
torch.save(model.t3.state_dict(), "t3_finetuned_YOUR_LANG.pt")
torch.save(model.s3gen.state_dict(), "s3gen_finetuned_YOUR_LANG.pt")
torch.save(model.ve.state_dict(), "ve_finetuned_YOUR_LANG.pt")
```

### Option B: Train from Scratch (Advanced)

For languages very different from existing ones, train the T3 and S3Gen models from scratch using your data. This requires:
- Significant computational resources (multiple GPUs)
- Preprocessing pipeline for your language
- Training loop implementation
- Model architecture understanding

---

## Step 5: Create Language Config (Optional)

If your language needs special model configuration, create a config file in [src/chatterbox/models/s3gen/configs.py](src/chatterbox/models/s3gen/configs.py):

```python
YOUR_LANGUAGE_PARAMS = AttrDict({
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
})
```

---

## Step 6: Update Model Loading

Update [mtl_tts.py](src/chatterbox/mtl_tts.py) `from_local()` and `from_pretrained()` methods to include your new model files:

```python
@classmethod
def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
    ckpt_dir = Path(ckpt_dir)

    ve = VoiceEncoder()
    ve.load_state_dict(
        torch.load(ckpt_dir / "ve.pt", weights_only=True)
    )
    ve.to(device).eval()

    t3 = T3(T3Config.multilingual())
    # Check for language-specific checkpoint
    t3_checkpoint = ckpt_dir / "t3_mtl23ls_v2.safetensors"
    if not t3_checkpoint.exists():
        t3_checkpoint = ckpt_dir / "t3_finetuned_YOUR_LANG.safetensors"
    
    t3_state = load_safetensors(t3_checkpoint)
    # ... rest of loading code
```

---

## Step 7: Testing Your Implementation

### Basic Test

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio

# Load model with your new language support
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Test with reference audio
reference_audio = "path/to/reference_voice.wav"

# Generate speech
text = "Your test text in the new language"
audio = model.generate(
    text=text,
    language_id="YOUR_LANG_CODE",
    audio_prompt_path=reference_audio
)

# Save output
torchaudio.save("output.wav", audio, model.sr)
```

### Validation Checklist

- [ ] Tokenizer correctly preprocesses your language text
- [ ] Language token is properly prepended to encoded text
- [ ] Model generates audio for your language
- [ ] Audio quality is acceptable (clear, natural prosody)
- [ ] Special characters are handled correctly
- [ ] Multiple speakers/voices work (if applicable)

---

## Step 8: Update Documentation

1. Add your language to the README.md supported languages table
2. Update the `SUPPORTED_LANGUAGES` documentation
3. Create example code for your language in [multilingual_app.py](multilingual_app.py)

### Example for Your Language

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

your_language_text = "Your text here"
wav = model.generate(
    your_language_text, 
    language_id="YOUR_LANG_CODE",
    audio_prompt_path="reference_speaker.wav"
)
torchaudio.save("output_YOUR_LANGUAGE.wav", wav, model.sr)
```

---

## Language-Specific Implementation Examples

### English (`en`)
- **Status**: ✅ Already fully supported (original Chatterbox language)
- **Preprocessing**: None required (basic normalization)
- **Data**: Any English speech dataset (audiobooks, podcasts, narration)
- **Sample Rate**: 44.1kHz recommended
- **Phoneme System**: Standard English IPA phonemes
- **Best Practices**:
  - Include diverse accents (American, British, Australian, etc.)
  - Vary speaking styles (casual, formal, narrative)
  - Include various punctuation and prosodies
- **Example Usage**:
```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Hello, this is an example of English text-to-speech synthesis."
wav = model.generate(text)
torchaudio.save("english_output.wav", wav, model.sr)
```

### Hindi (`hi`)
- **Status**: ✅ Already supported in multilingual model
- **Preprocessing**: Devanagari script normalization
- **Data**: Hindi speech data with Devanagari script transcriptions
- **Sample Rate**: 16kHz for tokenization, 44.1kHz for audio generation
- **Writing System**: Devanagari script (U+0900–U+097F Unicode range)
- **Phoneme System**: Hindi has ~40 consonants and vowels
- **Special Considerations**:
  - Devanagari has inherent schwa vowel (/ə/) - important for pronunciation
  - Voiced/unvoiced consonant pairs are phonemic
  - Nasalization and gemination (doubled consonants) affect meaning
  - Hindi uses 11 vowels (मात्रा - matra marks)
- **Text Preprocessing Example**:
```python
def hindi_normalize(text: str) -> str:
    """Normalize Hindi text for synthesis."""
    # Normalize Unicode combining characters
    from unicodedata import normalize
    text = normalize("NFC", text)  # Canonical decomposition
    
    # Replace common abbreviations
    text = text.replace("श्री", "श्री")  # Sri title
    text = text.replace("डॉ", "डॉक्टर")  # Dr. abbreviation
    
    return text
```
- **Example Usage**:
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

hindi_text = "नमस्ते, यह हिंदी टेक्स्ट-टू-स्पीच संश्लेषण का एक उदाहरण है।"
wav = model.generate(
    hindi_text,
    language_id="hi",
    audio_prompt_path="hindi_reference_voice.wav"
)
torchaudio.save("hindi_output.wav", wav, model.sr)
```
- **Data Sources**:
  - Google's Indic TTS datasets
  - Indian news broadcasts (TimesNow, NDTV archives)
  - YouTube Hindi audiobooks and podcasts
- **Tips for Hindi**:
  - Use native speakers from different regions (Hindustani, Modern Standard Hindi)
  - Include both formal and conversational speech
  - Ensure proper Unicode encoding (UTF-8)

### Chinese (Simplified: `zh`)
- **Preprocessing**: Cangjie character encoding
- **File**: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py#L175)
- **Data**: Character-level romanization or Pinyin
- **Tones**: Mandarin uses 4 tones plus neutral tone (mark with numbers 1-5)
- **Example**: "你好" (nǐ hǎo / ni3 hao3) = "hello"
- **Data Preparation**:
  - Pinyin with tone marks: `Nǐ hǎo` or number marks: `Ni3 hao3`
  - Simplified Chinese characters only
  - Multiple speakers for better generalization
- **Example Usage**:
```python
chinese_text = "你好，这是中文文本转语音合成的一个例子。"
wav = model.generate(
    chinese_text,
    language_id="zh",
    audio_prompt_path="chinese_reference.wav"
)
```

### Japanese (`ja`)
- **Preprocessing**: Kanji→Hiragana conversion using Kakasi
- **File**: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py#L60)
- **Dependencies**: `pykakasi` library
- **Writing System**: Mix of Hiragana, Katakana, and Kanji
- **Pitch Accent**: Japanese has lexical pitch accents (important for natural synthesis)
- **Example**:
```python
japanese_text = "こんにちは、これは日本語のテキスト音声合成の例です。"
wav = model.generate(
    japanese_text,
    language_id="ja",
    audio_prompt_path="japanese_reference.wav"
)
```

### Russian (`ru`)
- **Preprocessing**: Stress mark addition for phoneme accuracy
- **File**: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py#L145)
- **Dependencies**: `russian_text_stresser` library
- **Stress Marks**: Essential - Russian uses lexical stress that changes meaning
- **Example**:
  - мУка (múka) = flour
  - мукА (muká) = torment
- **Installation**: `pip install russian-text-stresser`
- **Example Usage**:
```python
russian_text = "Привет, это пример русского синтеза речи из текста."
wav = model.generate(
    russian_text,
    language_id="ru",
    audio_prompt_path="russian_reference.wav"
)
```

### Hebrew (`he`)
- **Preprocessing**: Diacritical marks for vowel accuracy
- **File**: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py)
- **Note**: Right-to-left text handling
- **Script**: Hebrew alphabet (Aleph-Bet)
- **Vowel Marks**: Nikud (diacritical points) - crucial for pronunciation
- **Example**:
```python
hebrew_text = "שלום, זה דוגמה של סינתזת דיבור בעברית."
wav = model.generate(
    hebrew_text,
    language_id="he",
    audio_prompt_path="hebrew_reference.wav"
)
```

### Korean (`ko`)
- **Preprocessing**: Hangul normalization
- **Phoneme conversion**: Jamo decomposition if needed
- **File**: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py)
- **Writing System**: Hangul (한글) - systematic and phonetic
- **Example**:
```python
korean_text = "안녕하세요, 이것은 한국어 텍스트 음성 합성 예제입니다."
wav = model.generate(
    korean_text,
    language_id="ko",
    audio_prompt_path="korean_reference.wav"
)
```

---

## Detailed Implementation: English & Hindi

### English Implementation

#### Data Collection
```
english_data/
├── audio/
│   ├── audiobook_001.wav (44.1kHz, mono)
│   ├── audiobook_002.wav
│   ├── podcast_001.wav
│   └── ...
└── transcriptions.json
    {
        "audiobook_001.wav": "This is the first sentence in the audiobook.",
        "audiobook_002.wav": "Here's another example.",
        ...
    }
```

#### Preprocessing (English needs minimal processing)
```python
def english_preprocess(text: str) -> str:
    """Basic English text preprocessing."""
    import re
    from unicodedata import normalize
    
    # Unicode normalization
    text = normalize("NFKD", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Expand common abbreviations
    abbrevs = {
        "Mr.": "Mister",
        "Mrs.": "Misses",
        "Dr.": "Doctor",
        "St.": "Street",
        "Ave.": "Avenue",
        "etc.": "et cetera",
    }
    for abbrev, expansion in abbrevs.items():
        text = re.sub(r"\b" + re.escape(abbrev) + r"\b", expansion, text, flags=re.IGNORECASE)
    
    return text
```

#### Tokenizer Vocabulary Setup
```python
# Create tokenizer vocabulary for English
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=5000,  # English needs fewer tokens
    special_tokens=[
        "[START]", "[STOP]", "[UNK]", "[SPACE]",
        "[PAD]", "[SEP]", "[CLS]", "[MASK]",
        "[en]"  # Language token for English
    ]
)

# Train on your English texts
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["english_transcriptions.txt"], trainer)
tokenizer.save("grapheme_mtl_english.json")
```

#### Fine-tuning the Model
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load base multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Your training loop (simplified)
optimizer = torch.optim.Adam(model.t3.parameters(), lr=1e-5)

for epoch in range(10):
    for batch_idx, (text_ids, audio_tokens, speaker_emb) in enumerate(train_loader):
        # Forward pass
        predicted_tokens = model.t3(
            text_tokens=text_ids.to("cuda"),
            t3_cond=speaker_emb.to("cuda")
        )
        
        # Compute loss
        loss = nn.functional.cross_entropy(predicted_tokens, audio_tokens.to("cuda"))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# Save fine-tuned model
torch.save(model.t3.state_dict(), "t3_english_finetuned.pt")
```

#### Testing English
```python
import torchaudio
from chatterbox.tts import ChatterboxTTS

# For English-only (use original model)
model = ChatterboxTTS.from_pretrained(device="cuda")

test_texts = [
    "Hello, this is a test of English synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?"
]

for i, text in enumerate(test_texts):
    wav = model.generate(text)
    torchaudio.save(f"english_test_{i}.wav", wav, model.sr)
    print(f"✅ Generated english_test_{i}.wav")
```

---

### Hindi Implementation

#### Data Collection
```
hindi_data/
├── audio/
│   ├── speaker1_001.wav (44.1kHz, mono, Devanagari)
│   ├── speaker1_002.wav
│   ├── speaker2_001.wav  (multiple speakers important)
│   └── ...
└── transcriptions.json
    {
        "speaker1_001.wav": "नमस्ते, यह हिंदी भाषण संश्लेषण का एक उदाहरण है।",
        "speaker1_002.wav": "भारत एक विविध और समृद्ध देश है।",
        ...
    }
```

#### Devanagari Text Preprocessing
```python
def hindi_preprocess(text: str) -> str:
    """Hindi text preprocessing with Devanagari normalization."""
    from unicodedata import normalize
    import re
    
    # NFC Normalization (important for Devanagari combining marks)
    text = normalize("NFC", text)
    
    # Remove extra whitespace but preserve structure
    text = " ".join(text.split())
    
    # Common Hindi abbreviation expansions
    hindi_abbrevs = {
        "डॉ": "डॉक्टर",      # Dr.
        "श्री": "श्री",        # Mr. (respectful)
        "सुश्री": "सुश्री",    # Ms.
        "आदि": "और इसी तरह",  # etc.
        "आज": "आज",          # today
    }
    
    for abbrev, expansion in hindi_abbrevs.items():
        text = text.replace(abbrev, expansion)
    
    # Normalize common punctuation variations
    text = text.replace("…", ",")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    
    # Ensure proper spacing before punctuation
    text = re.sub(r'\s+([।॥?!,।])', r'\1', text)
    
    return text
```

#### Devanagari Character Validation
```python
def is_devanagari(text: str) -> bool:
    """Check if text contains valid Devanagari characters."""
    # Devanagari Unicode range: U+0900–U+097F
    devanagari_pattern = r'[\u0900-\u097F]'
    return bool(re.search(devanagari_pattern, text))

def validate_hindi_data(transcription_file: str) -> None:
    """Validate that all transcriptions are in Devanagari."""
    import json
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    invalid_count = 0
    for filename, text in data.items():
        if not is_devanagari(text):
            print(f"⚠️ Non-Devanagari text in {filename}: {text}")
            invalid_count += 1
    
    print(f"✅ Validation complete: {invalid_count} invalid files")
```

#### Hindi Tokenizer Setup
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Hindi needs more tokens due to Devanagari complexity
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=15000,  # Larger vocab for Devanagari
    special_tokens=[
        "[START]", "[STOP]", "[UNK]", "[SPACE]",
        "[PAD]", "[SEP]", "[CLS]", "[MASK]",
        "[hi]"  # Language token for Hindi
    ]
)

tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["hindi_transcriptions.txt"], trainer)
tokenizer.save("grapheme_mtl_hindi.json")
```

#### Register Hindi in Tokenizer
Add to [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py):

```python
def hindi_normalize(text: str) -> str:
    """Hindi-specific text normalization."""
    from unicodedata import normalize
    
    # NFC normalization for proper Devanagari combining marks
    text = normalize("NFC", text)
    
    # Additional Hindi-specific preprocessing
    text = text.replace("॰", "।")  # Normalize danda variants
    
    return text

# In MTLTokenizer.encode():
elif language_id == 'hi':
    txt = hindi_normalize(txt)
```

#### Fine-tuning Model for Hindi
```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.tokenizers import MTLTokenizer

# Load pre-trained multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load Hindi-specific tokenizer
tokenizer = MTLTokenizer("grapheme_mtl_hindi.json")
model.tokenizer = tokenizer

# Training setup
optimizer = torch.optim.Adam(model.t3.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx, (text_ids, audio_tokens, speaker_emb) in enumerate(hindi_train_loader):
        # Ensure text_ids are tokenized with Hindi language token
        # text_ids should already include [hi] token from preprocessing
        
        predicted_tokens = model.t3(
            text_tokens=text_ids.to("cuda"),
            t3_cond=speaker_emb.to("cuda")
        )
        
        loss = torch.nn.functional.cross_entropy(predicted_tokens, audio_tokens.to("cuda"))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.t3.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / 50
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")
            total_loss = 0
    
    scheduler.step()
    print(f"✅ Epoch {epoch+1} completed")

# Save fine-tuned model
torch.save(model.t3.state_dict(), "t3_hindi_finetuned.safetensors")
torch.save(model.s3gen.state_dict(), "s3gen_hindi_finetuned.pt")
torch.save(model.ve.state_dict(), "ve_hindi_finetuned.pt")
```

#### Testing Hindi
```python
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model with Hindi support
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Test texts in Hindi
test_texts = [
    "नमस्ते, यह एक परीक्षण है।",  # Hello, this is a test.
    "भारत एक विविध देश है।",  # India is a diverse country.
    "संगीत जीवन की भाषा है।",  # Music is the language of life.
]

# Reference voice for cloning
reference_audio = "hindi_speaker_10s_sample.wav"

for i, text in enumerate(test_texts):
    try:
        wav = model.generate(
            text=text,
            language_id="hi",
            audio_prompt_path=reference_audio,
            cfg_weight=0.5,
            temperature=0.8
        )
        torchaudio.save(f"hindi_test_{i}.wav", wav, model.sr)
        print(f"✅ Generated hindi_test_{i}.wav: {text}")
    except Exception as e:
        print(f"❌ Error generating hindi_test_{i}.wav: {e}")

print("\n✅ All Hindi tests completed!")
```

#### Validation Checklist for Hindi
- [ ] All transcriptions use Devanagari script (U+0900–U+097F)
- [ ] Unicode is NFC normalized (not NFD)
- [ ] Audio files are 44.1kHz mono WAV
- [ ] Multiple speakers included (at least 3-5 unique speakers)
- [ ] Tokenizer includes [hi] language token
- [ ] Language-specific preprocessing works on sample text
- [ ] Model generates clear Hindi audio
- [ ] Tone and naturalness are acceptable
- [ ] Special characters (anusvara ँ, visarga ः) handled correctly

#### Common Hindi TTS Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Audio sounds robotic | Poor data quality or low-quality reference voice | Use clear, natural reference audio from native speakers |
| Wrong pronunciation | Missing or incorrect Devanagari diacritics | Validate text encoding; ensure NFC normalization |
| Degraded quality mid-synthesis | Model hasn't seen enough Hindi data | Fine-tune with more diverse Hindi speech (20+ hours) |
| Cannot tokenize text | Text encoding issues | Verify UTF-8 encoding and NFC normalization |
| Language token not recognized | [hi] not in vocabulary | Rebuild tokenizer with [hi] special token |

---

## Comparison: English vs Hindi Implementation

| Aspect | English | Hindi |
|--------|---------|-------|
| **Tokenizer Vocab Size** | ~5,000 tokens | ~15,000 tokens (Devanagari complexity) |
| **Preprocessing Complexity** | Minimal (lowercase, punctuation) | High (Devanagari normalization, matras) |
| **Script Type** | Latin alphabet | Devanagari (abugida) |
| **Diacritics** | None | Critical (vowel marks - मात्रा) |
| **Data Required** | ~10 hours | ~20-30 hours (more speakers needed) |
| **Fine-tuning Time** | ~5-10 hours (1 GPU) | ~20-40 hours (1 GPU) |
| **Speakers Recommended** | 3-5 | 5-10 (regional variations) |
| **Special Handling** | Abbreviation expansion | Unicode normalization, Devanagari marks |

---

## References

- **Hindi Resources**:
  - Devanagari Unicode: https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
  - Hindi Grammar: https://en.wikipedia.org/wiki/Hindi_grammar
  - IIT Bombay Hindi Corpus: https://www.cse.iitb.ac.in/~cvit/corpora/

- **English Resources**:
  - Phoneme Inventory: https://en.wikipedia.org/wiki/English_phonology
  - IPA Chart: https://www.internationalphoneticsassociation.org/

---

## Troubleshooting

### Issue: "Unsupported language_id"
**Solution**: Make sure your language code is added to `SUPPORTED_LANGUAGES` in [mtl_tts.py](src/chatterbox/mtl_tts.py#L21)

### Issue: Tokenizer fails on your language text
**Solution**: 
1. Check if all characters are in the tokenizer vocabulary
2. Implement language-specific preprocessing
3. Rebuild tokenizer with your language data

### Issue: Poor audio quality for your language
**Solution**:
1. Fine-tune the model on more language-specific data
2. Check audio preprocessing (sample rate, normalization)
3. Verify your reference voice is clear and natural

### Issue: Special characters not handled
**Solution**: Add character mapping in language-specific preprocessing function

---

## Performance Tips

1. **Data Quality**: 1 hour of high-quality data > 10 hours of low-quality
2. **Diversity**: Multiple speakers improve generalization
3. **Preprocessing**: Language-specific text normalization significantly improves quality
4. **Fine-tuning**: Start with learning rate ~1e-5 and adjust based on loss
5. **Voice Cloning**: Reference audio should be 10+ seconds of clear speech

---

## References

- **ISO 639-1 Language Codes**: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
- **Tokenizers Library**: https://huggingface.co/docs/tokenizers/
- **Chatterbox Repository**: https://github.com/resemble-ai/chatterbox
- **Model Config**: [src/chatterbox/models/s3gen/configs.py](src/chatterbox/models/s3gen/configs.py)

---

## Summary Checklist

- [ ] Data collected and organized (10-50 hours)
- [ ] Tokenizer vocabulary created with your language
- [ ] Language-specific preprocessing implemented (if needed)
- [ ] Language code added to `SUPPORTED_LANGUAGES`
- [ ] Model fine-tuned or adapted for your language
- [ ] Tests pass and audio quality is acceptable
- [ ] Documentation and examples updated
- [ ] Model checkpoint files are properly saved and loadable

