import nltk
# from pydub import AudioSegment
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
# No numpy needed if we are directly using tensors and torchaudio
# import numpy as np
import os

# --- NLTK Punkt Tokenizer Download ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
# --- ----------------------------- ---

# --- Chatterbox TTS Generation Function (returns tensor) ---
def chatterbox_tts_generate_tensor(
    text_chunk: str,
    tts_model: ChatterboxTTS,
    # sample_rate is known from the model, not needed as param here
    audio_prompt_path: str = None,
    # device is already handled by the model being on a device
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
) -> torch.Tensor | None: # Return tensor or None on error
    """
    Generates an audio tensor from a text chunk using ChatterboxTTS.
    Returns a torch.Tensor (on CPU, shape [1, num_samples]) or None on error.
    """
    print(f"Synthesizing with Chatterbox: {text_chunk}")

    wav_tensor = None
    effective_prompt_path = None
    if audio_prompt_path and os.path.exists(audio_prompt_path):
        effective_prompt_path = audio_prompt_path
    elif audio_prompt_path: # Path given but not found
        print(f"Warning: Audio prompt '{audio_prompt_path}' not found. Synthesizing without prompt.")

    try:
        wav_tensor = tts_model.generate(
            text_chunk,
            audio_prompt_path=effective_prompt_path,
            temperature=temperature,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration
        )
    except Exception as e:
        print(f"Error during TTS generation for chunk: '{text_chunk[:60]}...': {e}")
        return None # Indicate error

    # Ensure tensor is on CPU and is float for consistency before concatenation
    wav_tensor_cpu = wav_tensor.cpu().float()

    # Ensure tensor is 2D: [channels, samples]
    # Chatterbox typically returns [1, N] (which is desired for torchaudio.save and cat)
    # or sometimes [N] (which needs unsqueeze)
    if wav_tensor_cpu.ndim == 1: # If shape is [num_samples]
        wav_tensor_cpu = wav_tensor_cpu.unsqueeze(0) # Make it [1, num_samples]
    elif wav_tensor_cpu.ndim > 2:
        print(f"Warning: Unexpected tensor shape {wav_tensor_cpu.shape}. Squeezing.")
        wav_tensor_cpu = wav_tensor_cpu.squeeze() # Try to remove extra dims
        if wav_tensor_cpu.ndim == 1:
             wav_tensor_cpu = wav_tensor_cpu.unsqueeze(0)
        elif wav_tensor_cpu.ndim != 2 or wav_tensor_cpu.shape[0] != 1:
            print(f"Error: Could not reshape tensor {wav_tensor.shape} to [1, N]. Skipping chunk.")
            return None


    # It's good practice to ensure data is within [-1, 1] if saving as standard PCM later,
    # but torchaudio.save handles this well for float inputs.
    # We can add optional clamping if needed, but usually not required for torchaudio.
    # torch.clamp_(wav_tensor_cpu, -1.0, 1.0)

    return wav_tensor_cpu
# --- -------------------------------- ---

def split_text_into_chunks(long_text: str, max_chars_per_chunk: int = 400):
    sentences = nltk.sent_tokenize(long_text)
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        sentence_to_add = sentence.strip()
        if not sentence_to_add:
            continue
        if len(current_chunk) + len(sentence_to_add) + 1 > max_chars_per_chunk and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence_to_add + " "
        else:
            current_chunk += sentence_to_add + " "
        if i == len(sentences) - 1 and current_chunk.strip():
            chunks.append(current_chunk.strip())
    return chunks


def text_to_speech_long_torchaudio(
    long_text: str,
    tts_model: ChatterboxTTS, # Contains sample_rate (tts_model.sr)
    output_filename: str = "output_long.wav",
    max_chars_per_chunk: int = 300,
    inter_chunk_silence_ms: int = 350,
    audio_prompt_path: str = None,
    # device is implicit from tts_model's device
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    save_individual_chunks: bool = False
):
    """
    Converts long text to speech using torchaudio for all audio operations.
    """
    print("Splitting text into chunks...")
    text_chunks = split_text_into_chunks(long_text, max_chars_per_chunk)

    if not text_chunks:
        print("No text to synthesize after chunking.")
        return

    all_audio_tensors = []
    num_chunks = len(text_chunks)
    sample_rate = tts_model.sr
    print(f"Synthesizing {num_chunks} chunks using sample rate {sample_rate} Hz...")

    for i, chunk_text in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}...")
        chunk_tensor = chatterbox_tts_generate_tensor(
            chunk_text,
            tts_model,
            audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature
        )

        if chunk_tensor is None:
            print(f"Skipping chunk {i+1} due to generation error.")
            continue

        if save_individual_chunks:
            chunk_filename = f"chunk_{i+1}.wav"
            try:
                ta.save(chunk_filename, chunk_tensor, sample_rate)
                print(f"Saved individual chunk: {chunk_filename}")
            except Exception as e:
                print(f"Error saving chunk {chunk_filename} with torchaudio: {e}")

        all_audio_tensors.append(chunk_tensor)

        if i < num_chunks - 1 and inter_chunk_silence_ms > 0:
            silence_duration_samples = int(sample_rate * inter_chunk_silence_ms / 1000.0)
            # Ensure silence tensor has same dtype and device (CPU for concatenation)
            # and shape [1, num_samples]
            silence_tensor = torch.zeros((1, silence_duration_samples),
                                         dtype=chunk_tensor.dtype,
                                         device=chunk_tensor.device) # Should be CPU
            all_audio_tensors.append(silence_tensor)

    if not all_audio_tensors:
        print("No audio tensors were generated.")
        return

    print("Concatenating audio tensors...")
    try:
        # Ensure all tensors are on the same device (CPU) and 2D [1, N] before concatenating
        # chatterbox_tts_generate_tensor already returns CPU tensors.
        final_audio_tensor = torch.cat(all_audio_tensors, dim=1) # Concatenate along the time dimension
    except Exception as e:
        print(f"Error concatenating tensors: {e}")
        print("Details of tensors being concatenated:")
        for idx, t in enumerate(all_audio_tensors):
            print(f"Tensor {idx}: shape={t.shape}, dtype={t.dtype}, device={t.device}")
        return


    print(f"Exporting final audio to {output_filename} using torchaudio...")
    try:
        ta.save(output_filename, final_audio_tensor, sample_rate)
        print(f"Successfully exported to {output_filename}")
    except Exception as e:
        print(f"Error saving final audio with torchaudio: {e}")
    print("Done!")

# --- Main Execution ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")

    print("Loading ChatterboxTTS model...")
    try:
        # Make sure ChatterboxTTS model is initialized on the correct device
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"Model loaded successfully on {device}. Sample rate: {model.sr} Hz.")
    except Exception as e:
        print(f"Failed to load ChatterboxTTS model: {e}")
        exit()

    your_long_text = """In a quiet alley of Lahore, in the early spring of 2002, a boy named Zayan was born under the soft hum of ceiling fans and the prayerful whispers of his mother. From the very beginning, Zayan was different. While other children screamed and giggled in the streets, he sat in corners — observing, thinking, feeling.
He barely spoke to anyone outside his family. His father, a government clerk, wanted him to be outspoken and bold like his elder brother, but Zayan found comfort in silence. His world was built not of loud words but of quiet thoughts, books, stars, and numbers.
As he grew, life wasn’t easy.
Zayan was deeply sensitive. A classmate's unkind word could echo in his mind for days. When his favorite teacher once scolded him for not raising his hand in class, he didn't sleep that night. But he wasn't weak — his sensitivity made him more aware, more human.
At home, his introversion was misunderstood. Relatives labeled him odd. Even his parents, loving but practical, urged him to “go out more” and “stop reading all day.” He wanted to scream: "I'm trying to build a world in my head that I can survive in."
But instead, he just smiled and nodded, and went back to his room.
Behind those quiet eyes, a fire burned. Zayan was ambitious. He wanted to do something meaningful — to be more than what the world thought he could be.
He stumbled upon a video titled “Artificial Intelligence Explained” when he was 16. Something clicked. “Machines that learn… like the brain?” It was as if the universe had whispered his purpose into his ears.
From that day, Zayan taught himself Python on a borrowed laptop. Late at night, while his family slept, he studied neural networks, watched YouTube lectures, and read research papers by flashlight. His hard work wasn’t for applause — it was survival. It was his escape.
But dreams don’t bloom easily.
Their internet connection was unstable. His parents couldn’t afford a new laptop. Some nights, he walked to a friend’s house just to download a library or a dataset.
At university, he couldn’t afford private academies. While others had mentors and networks, Zayan had only StackOverflow, GitHub, and faith. He often sat silently in lectures, understanding more than most, but never speaking.
He applied for internships — most ignored him. He sent cold emails — silence. He watched people with better connections fly ahead while he stumbled in the dark. His confidence began to shake. Maybe he wasn't meant for greatness.
But then, he remembered: The flame doesn’t speak. It just burns.
One evening, while debugging a TensorFlow model for a small freelance gig, he received an email:
“Your solution to the problem we posted on Kaggle was impressive. We'd like to talk.”
It was a researcher from a European AI lab. Zayan had unknowingly submitted a brilliant idea — something that caught the eye of a PhD scholar.
That conversation turned into collaboration. That collaboration turned into a research internship. And soon, a fully-funded scholarship.
By 25, Zayan had published two papers in international journals. He still didn’t talk much. At conferences, he quietly sipped tea in corners. But when he took the stage, explaining his model for low-resource NLP systems in South Asian languages, the room fell silent — not out of pity, but awe.
He later built an open-source framework for AI education in Urdu, so kids like him — introverted, poor, unheard — could learn.
Zayan never forgot the struggles, the rejections, the nights of self-doubt. They didn’t break him. They shaped him.
Years later, a young boy in a village school would open a laptop, connect to a free course built by “some guy named Zayan”, and begin his journey — quiet, reserved, ambitious — just like Zayan once was.
Because sometimes, the most powerful flames don’t roar.
They whisper And they light up the world anyway."""


    AUDIO_PROMPT_PATH = "/kaggle/input/taskchatterbox/test-2.wav" # Update this path
    if not os.path.exists(AUDIO_PROMPT_PATH):
        print(f"Warning: Audio prompt file '{AUDIO_PROMPT_PATH}' not found. Proceeding without audio prompt.")
        # AUDIO_PROMPT_PATH = None # Explicitly set to None if preferred for clarity

    MAX_CHARS_PER_CHUNK = 550
    INTER_CHUNK_SILENCE_MS = 100
    
    tts_exaggeration = 0.5
    tts_cfg_weight = 0.5
    tts_temperature = 0.8

    text_to_speech_long_torchaudio(
        long_text=your_long_text,
        tts_model=model, # Pass the whole model
        output_filename="chatterbox_long_output_torchaudio3.wav",
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
        inter_chunk_silence_ms=INTER_CHUNK_SILENCE_MS,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=tts_exaggeration,
        cfg_weight=tts_cfg_weight,
        temperature=tts_temperature,
        save_individual_chunks=True # Set to True to save and check individual chunks
    )


    