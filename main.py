from chatterbox_git.src.chatterbox import mtl_tts
import torchaudio as ta
from safetensors.torch import load_file as load_safetensors

device = "cuda" # or mps or cuda

multilingual_model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)

# ----
# Then download the file from huggingface and place it in the current directory.
# ----



t3_state = load_safetensors("t3_cs.safetensors", device="cuda")
multilingual_model.t3.load_state_dict(t3_state)
multilingual_model.t3.to(device).eval()

czech_text = "Přečtěte si krátký text a odpovězte na několik otázek, které testují porozumění. Můžete se začíst do krátkých úryvků z článků nebo do některého z našich krátkých a vtipných příběhů. Pozor, vybraný text můžete řešit pouze jednou v daný den."
wav_czech = multilingual_model.generate(czech_text, language_id="cs")
ta.save("test-cs.wav", wav_czech, multilingual_model.sr)

