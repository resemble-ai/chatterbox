import torch
import os
import tempfile
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
import queue
import time

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# --- Optional dependencies ---
import nltk
from pydub import AudioSegment
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize


# --- Audio Conversion Utility ---
def convert_to_wav(audio_path):
    """Convert any supported audio format (mp3, m4a, flac, etc.) to wav."""
    if audio_path.lower().endswith(".wav"):
        return audio_path
    print(f"[INFO] Converting {audio_path} → WAV...")
    sound = AudioSegment.from_file(audio_path)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.export(tmp_wav.name, format="wav")
    return tmp_wav.name


# --- WAV Saving Utility ---
def save_wav(path, wav, sr):
    """Save waveform tensor/array to a .wav file safely."""
    if isinstance(wav, torch.Tensor):
        arr = wav.detach().cpu().numpy()
    else:
        arr = np.asarray(wav)

    # Ensure correct shape for saving
    if arr.ndim == 1:
        arr = arr[:, None]  # make mono (samples, 1)
    elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (samples, channels)

    arr = arr.astype("float32")
    sf.write(path, arr, sr)
    print(f"[INFO] Saved file: {os.path.abspath(path)}")


# --- Background Audio Player Thread ---
def audio_player_thread(play_queue: queue.Queue, sr: int):
    """Continuously play audio chunks from a queue."""
    while True:
        wav_chunk = play_queue.get()
        if wav_chunk is None:  # Sentinel to stop
            break

        arr = np.asarray(wav_chunk)
        if arr.ndim == 2 and arr.shape[0] <= 2:
            arr = arr.T
        arr = np.clip(arr.astype("float32"), -1.0, 1.0)

        sd.play(arr, sr)
        sd.wait()
        play_queue.task_done()


# --- Main TTS Streaming Function ---
def stream_tts(
    text: str,
    voice_path: str = None,
    multilingual: bool = False,
    language_id: str = None,
):
    # Auto device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # Model selection
    if multilingual:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        model = ChatterboxTTS.from_pretrained(device=device)

    # Convert voice if provided
    audio_prompt = None
    if voice_path:
        audio_prompt = convert_to_wav(voice_path)
        print(f"[INFO] Using voice clone from: {audio_prompt}")

    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    print(f"[INFO] {len(sentences)} sentences detected.")

    combined_audio = []
    play_q = queue.Queue()

    # Start background player thread
    player = threading.Thread(target=audio_player_thread, args=(play_q, model.sr))
    player.daemon = True
    player.start()

    # Generate and enqueue sentences
    for i, sentence in enumerate(sentences, start=1):
        print(f"\n[GEN {i}/{len(sentences)}] {sentence}")

        if multilingual:
            wav = model.generate(sentence, language_id=language_id, audio_prompt_path=audio_prompt)
        else:
            wav = model.generate(sentence, audio_prompt_path=audio_prompt)

        # Convert to numpy
        wav_np = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)

        # Normalize shape → always (samples,)
        if wav_np.ndim == 2:
            if wav_np.shape[0] <= 2:
                wav_np = wav_np.T
            wav_np = wav_np.mean(axis=1)  # mix stereo → mono

        combined_audio.append(wav_np)
        play_q.put(wav_np)

    # Wait for all playback
    play_q.join()
    play_q.put(None)
    player.join(timeout=1)

    # Combine all chunks (safe concatenation)
    try:
        final_wav = np.concatenate(combined_audio)
        save_wav("final_output.wav", final_wav, model.sr)
        print("\n✅ Final audio saved successfully as 'final_output.wav'")
    except Exception as e:
        print(f"[ERROR] Saving failed: {e}")
        print("Attempting fallback save...")
        try:
            np.save("final_output_backup.npy", np.array(combined_audio, dtype=object))
            print("[INFO] Saved raw numpy backup as 'final_output_backup.npy'")
        except Exception as e2:
            print(f"[FATAL] Backup failed: {e2}")


# --- Example Usage ---
if __name__ == "__main__":
    text_input = """
    Dubai is a city like no other, a shimmering jewel set between the golden sands of the Arabian Desert and the turquoise waves of the Persian Gulf, where the ancient rhythms of Bedouin life meet the pulse of twenty-first-century ambition. Once a modest fishing and pearl-diving village, Dubai has transformed within a single lifetime into a global symbol of innovation, wealth, and vision. Its story is not merely one of oil and opulence, but of determination, leadership, and imagination that dared to defy the limits of geography and expectation. When you first arrive, the skyline announces itself before anything else—towers of glass and steel reflecting the desert sun, each structure a statement of possibility. Rising above them all stands the Burj Khalifa, an engineering miracle that pierces the clouds at over eight hundred meters high, shimmering like a needle stitching the desert to the heavens. But Dubai’s spirit is not confined to its skyscrapers; it extends to the very ground beneath, where roads are lined with palm trees, fountains dance in synchronization with music, and the scent of oud and cardamom lingers in the air like a promise of something timeless.
    """

    voice_clone_path = "Audio/PM Modi's speech at launch of Financing Facility under Agriculture Infrastructure Fund.wav"

    stream_tts(
        text=text_input,
        voice_path=voice_clone_path,
        multilingual=False,
        language_id="en",
    )
