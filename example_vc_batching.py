import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


model = ChatterboxTTS.from_pretrained(device=device)

texts_batch = [ "This is the first sentence to be synthesized in a batch.",
                "This is the second one." ]


# If you want to synthesize with a different voice, specify the audio prompt:
AUDIO_PROMPT_PATH = "YOUR_AUDIO_PROMPT.wav"


# Batching - list of strings to synthesize multiple different texts in a single batch.
# This is the most efficient way to process multiple, different prompts at once.
# Careful: 1 text = 1 additional KV Cache (Vram)
wavs_batch = model.generate(texts_batch, audio_prompt_path=AUDIO_PROMPT_PATH)
for i, wav in enumerate(wavs_batch):
    ta.save(f"test-batch-{i+1}.wav", wav, model.sr)

# Batching - Use num_return_sequences to generate multiple variations for each text.
# This is highly efficient for creating diverse samples, as the prompt is only processed once.
# Without making extra KV Caches. 
num_variations = 3

wavs_batch_multi = model.generate(texts_batch, audio_prompt_path=AUDIO_PROMPT_PATH, num_return_sequences=num_variations)
for i, group in enumerate(wavs_batch_multi):
    for j, wav in enumerate(group):
        ta.save(f"test-batch-{i+1}-variant-{j+1}.wav", wav, model.sr)