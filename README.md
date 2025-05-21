<img width="1200" alt="chatterbox-logo-black" src="https://github.com/user-attachments/assets/172bd283-5d88-4302-aed9-4b0eda94f20f" />

# Chatterbox TTS
We're excited to introduce Chatterbox, our first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our Hugging Face Gradio app.

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- SoTA zeroshot TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Very low WER < 1%
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- Outperforms ElevenLabs - #TODO REPORT LINK

# Installation
```
conda create -yn chatterbox python=3.11
conda activate chatterbox
git clone https://github.com/resemble-ai/chatterbox.git
pip install .  # will be `pip install chatterbox`
```

# Usage
```python
import torchaudio as ta
from chatterbox.models.wrapper import Chatterbox

model = Chatterbox.from_pretrained(device="cuda")

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
