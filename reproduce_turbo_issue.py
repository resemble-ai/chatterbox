
import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Long text (> 350 chars)
LONG_TEXT = """
In the heart of the bustling city, where neon lights flickered like distant stars, lived a detective named Jack. 
Jack wasn't your ordinary investigator; he specialized in the peculiar, the unexplained, and the down-right weird. 
One rainy Tuesday, a woman walked into his office, her coat dripping water onto his already stained rug. 
She claimed her cat had started reciting Shakespeare in perfect iambic pentameter. 
Intrigued, Jack grabbed his fedora and followed her into the storm, unaware that this case would lead him to a secret society of literary felines plotting world domination through sonnets.
"""

def reproduce():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        print("Generating audio for long text (approx {} chars)...".format(len(LONG_TEXT)))
        
        wav = model.generate(LONG_TEXT)
        ta.save("turbo_long_test.wav", wav, model.sr)
        print("Saved 'turbo_long_test.wav'. Check for hallucinations.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reproduce()
