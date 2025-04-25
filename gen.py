import torchaudio as ta
from orator.models.wrapper import Orator

model = Orator.from_local("checkpoints", "cuda")

text = "Oh no! The S&P 500 stock fell by 40% today. I gotta sleep in the park tonight."
wav = model.generate(text, "tests/trimmed_8b7f38b1.wav")
ta.save("test.wav", wav, 24000)
