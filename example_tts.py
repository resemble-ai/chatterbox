import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

logging.info(f"Using device: {device}")

# Save original torch.load
original_torch_load = torch.load

def cpu_torch_load(*args, **kwargs):
    # Force map_location to CPU
    kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)

try:
    logging.info("Loading pretrained TTS model...")

    # Monkeypatch torch.load temporarily
    torch.load = cpu_torch_load
    model = ChatterboxTTS.from_pretrained(device=device)
    # Restore torch.load
    torch.load = original_torch_load

    logging.info("Model loaded successfully.")

    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    logging.info(f"Generating audio for text: {text}")

    wav = model.generate(text)
    logging.info("Audio generation completed successfully.")

    output_path = "test-1.wav"
    ta.save(output_path, wav, model.sr)
    logging.info(f"Audio saved to {output_path}")

except Exception as e:
    logging.error("An error occurred during TTS processing", exc_info=True)
    # Restore torch.load if exception happens before restoring
    torch.load = original_torch_load
