# chatterbox/run_tts_test.py

import torchaudio
import torch
from pathlib import Path
import time
import sys

# --- Robust Path Setup ---
# Add the project root to the Python path to allow importing the chatterbox module
try:
    # Assuming this script is in the 'chatterbox' directory, the project root is one level up.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from chatterbox.tts import ChatterboxTTS
except (ImportError, NameError):
    print("Error: Could not import ChatterboxTTS.")
    print("Please ensure this script is located in the 'chatterbox' directory and that the main project structure is intact.")
    sys.exit(1)


# --- Configuration ---

# 1. Define the texts you want to generate in a batch.
texts_to_generate = [
    "Artists create a sequence of drawings to visualize the animation, shot by shot, much like a comic book. This process helps the entire team understand the director's vision, plan the pacing of the story, and identify potential issues before any resource-intensive digital work commences. It serves as the essential blueprint that guides all subsequent stages of production, ensuring a cohesive final product.",
    "Following this, the rigging process gives the models a digital skeleton, or armature, which allows animators to pose and move them realistically. This technical yet artistic step is fundamental for creating believable movement, as a well-constructed rig provides the intuitive controls necessary for bringing static models to life.",
    "Simultaneously, the texturing process involves painting and applying surface details, such as skin, fabric, or metal, to the models. These detailed maps, known as textures, determine how light interacts with the surfaces, adding a layer of realism and visual richness. This combination of movement and surface artistry transforms simple geometric shapes into compelling characters.",
    "Rendering is the computationally intensive process of calculating the final image from all the data, turning the 3D scene into a sequence of 2D frames. Finally, compositing combines these rendered layers with visual effects and color grading in post-production, seamlessly integrating every component to achieve the final, stunning look of the completed animation.",
]

# 2. Define the path to the voice prompt audio file.
#    This uses the project root to create a reliable path.
AUDIO_PROMPT_PATH = project_root / "assets" / "audio_sample1.wav"

# 3. Define the output directory for the generated audio files.
OUTPUT_DIR = Path(__file__).resolve().parent / "tts_test_outputs"


def main():
    """Main function to run the TTS test."""

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # --- Validate Inputs ---
    if not AUDIO_PROMPT_PATH.is_file():
        print(f"Error: Audio prompt file not found at '{AUDIO_PROMPT_PATH}'")
        return
    
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Audio files will be saved in: '{OUTPUT_DIR}'")

    # --- Model Loading ---
    print("\nLoading Chatterbox TTS model... (This may take a moment)")
    start_time = time.time()
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds.")

    # --- Audio Generation ---
    print(f"\nGenerating audio for a batch of {len(texts_to_generate)} texts...")
    start_time = time.time()
    
    # The `generate` method handles batching when given a list of strings.
    wavs_batch = model.generate(
        texts_to_generate,
        audio_prompt_path=str(AUDIO_PROMPT_PATH)
    )
    
    generation_time = time.time() - start_time
    print(f"Batch generation completed in {generation_time:.2f} seconds.")

    # --- Saving Outputs ---
    print("\nSaving generated audio files...")
    for i, wav_tensor in enumerate(wavs_batch):
        output_filename = OUTPUT_DIR / f"output_batch_{i+1}.wav"
        torchaudio.save(str(output_filename), wav_tensor, model.sr)
        print(f"  - Saved: {output_filename.name}")

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    main()