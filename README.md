# Chatterbox TTS
We're excited to introduce Chatterbox, our first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like 11labs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our Hugging Face Gradio app.

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="LINK">link</a>). Starting at $X per hour, it delivers reliable performance with ultra-low latency of Y ms—ideal for production use in agents, applications, or interactive media.

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

# Acknowledgements
- Cosyvoice
- HiFT-GAN
- Llama

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
