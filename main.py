import os
import io
import time
import warnings
import logging
import contextlib

import torch
import torchaudio as ta

# --- Quieting noisy libraries & progressbars ---
# Recommended environment flags (best-effort; different libs may still log)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Python warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)

# Redirect stdout/stderr when loading model / running generate to suppress prints
@contextlib.contextmanager
def suppress_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(new_out), contextlib.redirect_stderr(new_err):
        yield

# --- Patch torch.load to map to CPU (your original approach) ---
device = "cpu"
map_location = torch.device(device)
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# --- Import / instantiate model (suppress verbose output) ---
# Replace these imports with the exact module paths you use in your project.
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Loading model (this may take a moment)...")
with suppress_output():
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("Model loaded.")

# --- Inference helper that times and suppresses verbose sampling output ---
def synthesize_and_save(text: str, language_id: str = "it", out_path: str = "out.wav"):
    # Ensure text is shown in output; but suppress other lib messages.
    start_time = time.time()
    with suppress_output():
        wav = model.generate(text, language_id=language_id)
    end_time = time.time()

    # wav is assumed to be a Tensor [channels, samples] or [samples]
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if wav.ndim == 1:
        channels = 1
        samples = wav.shape[0]
        # torchaudio expects (channels, samples)
        wav_to_save = wav.unsqueeze(0)
    else:
        channels = wav.shape[0]
        samples = wav.shape[1]
        wav_to_save = wav

    sr = getattr(model, "sr", None)
    if sr is None:
        # fallback sample rate (adjust if needed)
        sr = 22050

    ta.save(out_path, wav_to_save, sr)

    inference_time_s = float(end_time - start_time)
    audio_duration_s = float(samples / sr) if sr > 0 else 0.0

    # real-time factor: audio duration / inference time
    # >1 = faster-than-real-time (you produced more audio seconds than wall seconds)
    rtf = audio_duration_s / inference_time_s if inference_time_s > 0 else float("inf")

    # Friendly CLI report
    print("\n--- Inference report ---")
    print(f"Sentence: {text}")
    print(f"Inference time: {inference_time_s:.3f} s")
    print(f"Audio duration: {audio_duration_s:.3f} s")
    print(f"Real-time factor (audio_duration / inference_time): {rtf:.3f}x")
    print(f"Saved WAV: {out_path}")
    print("------------------------\n")

    return {
        "text": text,
        "inference_time_s": inference_time_s,
        "audio_duration_s": audio_duration_s,
        "real_time_factor": rtf,
        "wav_path": out_path,
    }

# --- Example usage ---
if __name__ == "__main__":
    text = "Il mio nome è Conan, e faccio il detective!"
    synthesize_and_save(text, language_id="it", out_path="test1.wav")
