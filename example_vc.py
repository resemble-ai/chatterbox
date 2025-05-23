from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained("cuda")
wav = model.generate("tests/trimmed_8b7f38b1.wav")
import torchaudio as ta
ta.save("testvc.wav", wav, model.sr)
