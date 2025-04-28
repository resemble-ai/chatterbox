# orator
Open source TTS model

# Installation
```
conda create -yn orator python=3.11
conda activate orator
git clone https://github.com/resemble-ai/orator.git
pip install .  # will be `pip install orator`
```

# Usage
```python
import torchaudio as ta
from orator.models.wrapper import Orator

model = Orator.from_pretrained(device="cuda")

text = "What does the fox say?"
wav = model.generate(text)  # using the default voice
ta.save("test.wav", wav, model.sr)
```
See `example_tts.py` for more examples.
