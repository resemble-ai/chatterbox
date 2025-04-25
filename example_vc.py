from orator.vc import OratorVC

model = OratorVC.from_local("checkpoints", "cuda")
wav = model.generate("tests/trimmed_8b7f38b1.wav")
import torchaudio as ta
ta.save("testvc.wav", wav, model.sr)
