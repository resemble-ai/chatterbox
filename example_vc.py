import torchaudio as ta
from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained("cuda")
wav = model.generate("tests/trimmed_8b7f38b1.wav")
ta.save("testvc.wav", wav, model.sr)
