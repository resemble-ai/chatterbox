import os

import soundfile as sf

from chatterbox import ChatterboxInference


# Example 1: custom or finetuned multilingual model from Hugging Face

inference = ChatterboxInference.from_pretrained(
    repo_id="CoRal-project/roest-v3-chatterbox-500m",
    model_type="multilingual",
    language="da",
)

# Shows sentence splitting and number normalization in Danish.
text_da = "Temperaturen i dag er 12 grader. Der er 3 æbler og 1 liter mælk tilbage i køleskabet."
wav = inference.generate(
    text_da,
    language_id="da",
    normalize_text=True,
    sentence_split=True,
    inter_sentence_silence_ms=100,
)
sf.write("example-1.wav", wav.squeeze().numpy(), inference.sr)

# Example 1b: same model with CUDA graph acceleration (~2x faster on CUDA)
# Falls back to generate() automatically on CPU / MPS.
wav = inference.generate_fast(
    text_da,
    language_id="da",
    normalize_text=True,
    sentence_split=True,
    inter_sentence_silence_ms=100,
)
sf.write("example-1-fast.wav", wav.squeeze().numpy(), inference.sr)


# Example 2: default upstream turbo model (English)
turbo_inference = ChatterboxInference.from_pretrained(
    model_type="turbo",
)

text_en = "Hello! This is Chatterbox, a high quality text to speech model. It handles multiple sentences naturally."
wav = turbo_inference.generate(
    text_en,
    normalize_text=True,
    sentence_split=True,
    inter_sentence_silence_ms=100,
    exaggeration=0.0,
    temperature=0.8,
    top_p=0.95,
)
sf.write("example-2.wav", wav.squeeze().numpy(), turbo_inference.sr)


# Example 3: local checkpoint directory
# local_inference = ChatterboxInference.from_local(
#     "/path/to/model_dir",
#     model_type="multilingual",
#     language="da",
# )
# wav = local_inference.generate(text_da, language_id="da")
# sf.write("example-local.wav", wav.squeeze().numpy(), local_inference.sr)


# Optional: synthesize with a reference voice
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
if os.path.exists(AUDIO_PROMPT_PATH):
    wav = inference.generate(
        text_da,
        language_id="da",
        audio_prompt_path=AUDIO_PROMPT_PATH,
        normalize_text=True,
        sentence_split=True,
        inter_sentence_silence_ms=100,
        exaggeration=0.5,
    )
    sf.write("example-3.wav", wav.squeeze().numpy(), inference.sr)
