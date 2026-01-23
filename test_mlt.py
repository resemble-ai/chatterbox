import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Hi this is Paula! How can I assist you today?"
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

norway_text = "Hei, dette er Paula! Hvordan kan jeg hjelpe deg i dag?"
wav_norway = multilingual_model.generate(norway_text, language_id="no")
ta.save("test-norway.wav", wav_norway, model.sr)