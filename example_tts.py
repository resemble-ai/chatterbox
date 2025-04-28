import torchaudio as ta
from orator.tts import OratorTTS

model = OratorTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
text = "Pikachu, Bulbasaur, and Eevee were sitting next to Meowth, staring at Vincent van Gogh's Starry Night."
wav = model.generate(text, audio_prompt_path="tests/trimmed_8b7f38b1.wav")
ta.save("test-2.wav", wav, model.sr)
