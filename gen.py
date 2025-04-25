from models.wrapper import Orator
import torchaudio as ta

model = Orator.from_local("checkpoints", "cuda")

text = "Oh no! The S&P 500 stock fell by 40% today. I gotta sleep in the park tonight."
wav = model.generate(text, "tests/trimmed_8b7f38b1.wav")
ta.save("test1.wav", wav, 24000)
