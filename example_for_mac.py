import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = """
A “Spooky” Science 

In 1935, Albert Einstein and colleagues first pointed out the “spooky” action of quantum entanglement. Quantum entanglement, however, appeared to conflict with Einstein’s theory of special relativity, which postulates that nothing can travel faster than the speed of light and is demonstrated mathematically by the well-known equation E=mc2.  


What is Quantum Entanglement? 

Quantum science explores and helps explain some of the strangest phenomena in the universe, even shedding light on the mystery of dark matter and dark energy. Quantum is the study of atoms and subatomic particles, and how they interact with each other. It examines the very stuff we, and everything around us, are made of.  

One of the most far-out phenomena of quantum theory is quantum entanglement, the idea that particles of the same origin, which were once connected, always stay connected. Even if they separate and move far apart in time and space, they continue to share something beyond a mere bond — they shed their original quantum states and take on a new, united quantum state which they maintain forever. This means if something happens to one particle, it affects all the others with which it’s entangled. 


"""

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "./EQM - Big Hero 6 - Hello, I am Baymax (mp3cut.net).mp3"
wav = model.generate(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.4,
    cfg_weight=0.3
    )
ta.save("test-6.wav", wav, model.sr)
