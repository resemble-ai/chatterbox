"""
Real-world example: Audiobook generator with character voices.

This demonstrates how to use conditional caching in a practical application
where multiple voices are used to narrate different parts of a story.
"""

import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from conditional_cache import ConditionalCache

# Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

print("Loading TTS model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("Model loaded!\n")

# Initialize cache with disk persistence
cache = ConditionalCache(cache_dir="./audiobook_cache", auto_save=True)

# Define character voices (audio_file, exaggeration)
VOICES = {
    "narrator": ("julia-whelan.wav", 0.3),  # Calm, steady narrator voice
    # "hero": ("earl-nightingale.wav", 0.5),  # Uncomment if you have this file
    # "villain": ("another-voice.wav", 0.7),  # More dramatic
}

# Story script with speaker annotations
STORY_SCRIPT = [
    ("narrator", "It was a dark and stormy night when everything changed. The rain poured down in sheets, and thunder rumbled in the distance."),
    ("narrator", "Sarah stood at the window, watching the storm rage outside. She knew that by morning, nothing would ever be the same again."),
    ("narrator", "Suddenly, a knock echoed through the house. Sarah's heart raced as she approached the door."),
    ("narrator", "The stranger on the doorstep spoke in a low voice."),
]

# Generation parameters
LANGUAGE = "en"
CFG_WEIGHT = 0.5

def generate_audiobook(script, voices, output_dir="./audiobook_output"):
    """
    Generate audiobook from script with multiple character voices.
    
    Args:
        script: List of (speaker, text) tuples
        voices: Dictionary mapping speaker names to (audio_file, exaggeration)
        output_dir: Directory to save generated audio files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("AUDIOBOOK GENERATION")
    print("=" * 80)
    print(f"Segments: {len(script)}")
    print(f"Voices: {len(voices)}")
    print(f"Output: {output_path}\n")
    
    # Track voice usage for statistics
    voice_usage = {speaker: 0 for speaker in voices.keys()}
    
    # Generate each segment
    all_segments = []
    
    for i, (speaker, text) in enumerate(script, 1):
        if speaker not in voices:
            print(f"⚠ Warning: Unknown speaker '{speaker}' - skipping segment {i}")
            continue
        
        voice_file, exaggeration = voices[speaker]
        
        # Check if voice file exists
        if not Path(voice_file).exists():
            print(f"⚠ Warning: Voice file not found '{voice_file}' - skipping segment {i}")
            continue
        
        voice_usage[speaker] += 1
        
        print(f"[{i}/{len(script)}] {speaker}: \"{text[:60]}...\"")
        
        # Get or prepare conditionals (uses cache automatically)
        cache.get_or_prepare(
            model, 
            voice_file, 
            exaggeration=exaggeration,
            verbose=False  # Reduce output clutter
        )
        
        # Generate audio
        wav = model.generate(
            text,
            language_id=LANGUAGE,
            exaggeration=exaggeration,
            cfg_weight=CFG_WEIGHT,
        )
        
        # Save individual segment
        segment_file = output_path / f"segment_{i:03d}_{speaker}.wav"
        ta.save(segment_file, wav, model.sr)
        
        all_segments.append((wav, speaker))
        print(f"    ✓ Saved: {segment_file.name}\n")
    
    # Concatenate all segments into single audiobook file
    if all_segments:
        print("=" * 80)
        print("Combining segments into audiobook...")
        
        # EFFICIENT CONCATENATION: Move to CPU first, collect in list, single cat at end
        # This avoids O(n²) memory copying and MPS memory fragmentation
        cpu_segments = []
        for seg, speaker in all_segments:
            # Ensure tensor is on CPU to free MPS memory
            cpu_seg = seg.detach().cpu() if seg.device.type != 'cpu' else seg
            cpu_segments.append(cpu_seg)
        
        # Single concatenation in system RAM
        combined_audio = torch.cat(cpu_segments, dim=1)
        
        audiobook_file = output_path / "complete_audiobook.wav"
        ta.save(audiobook_file, combined_audio, model.sr)
        
        duration_seconds = combined_audio.shape[1] / model.sr
        print(f"✓ Complete audiobook saved: {audiobook_file}")
        print(f"  Duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
        print(f"  Segments: {len(all_segments)}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)
    
    print("\nVoice usage:")
    for speaker, count in voice_usage.items():
        print(f"  {speaker}: {count} segments")
    
    print(f"\n{cache}")
    
    print("\nCache statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Audiobook generation complete!")
    return output_path


def add_silence_between_segments(segments, silence_duration=0.5, sample_rate=24000):
    """
    Add silence between audio segments for better pacing.
    
    Args:
        segments: List of audio tensors
        silence_duration: Duration of silence in seconds
        sample_rate: Sample rate of audio
    """
    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros(1, silence_samples)
    
    result = []
    for i, segment in enumerate(segments):
        result.append(segment)
        if i < len(segments) - 1:  # Don't add silence after last segment
            result.append(silence)
    
    return torch.cat(result, dim=1)


def generate_with_pauses(script, voices, output_dir="./audiobook_output", pause_duration=0.5):
    """
    Generate audiobook with pauses between segments.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Generating audiobook with pauses...")
    
    segments = []
    
    for i, (speaker, text) in enumerate(script, 1):
        if speaker not in voices:
            continue
        
        voice_file, exaggeration = voices[speaker]
        if not Path(voice_file).exists():
            continue
        
        print(f"[{i}/{len(script)}] {speaker}...")
        
        cache.get_or_prepare(model, voice_file, exaggeration=exaggeration, verbose=False)
        wav = model.generate(text, language_id=LANGUAGE, exaggeration=exaggeration, cfg_weight=CFG_WEIGHT)
        
        segments.append(wav)
    
    # Add pauses
    combined = add_silence_between_segments(segments, pause_duration, model.sr)
    
    output_file = output_path / "audiobook_with_pauses.wav"
    ta.save(output_file, combined, model.sr)
    
    print(f"✓ Saved: {output_file}")
    return output_file


# Main execution
if __name__ == "__main__":
    # Pre-warm cache with all voices (optional, but speeds up generation)
    print("Pre-warming cache with all voices...\n")
    for speaker, (voice_file, exaggeration) in VOICES.items():
        if Path(voice_file).exists():
            print(f"  Loading {speaker} voice...")
            cache.get_or_prepare(model, voice_file, exaggeration=exaggeration)
        else:
            print(f"  ⚠ Skipping {speaker} - file not found: {voice_file}")
    
    print(f"\n{cache}\n")
    
    # Generate the audiobook
    output_dir = generate_audiobook(STORY_SCRIPT, VOICES)
    
    # Optional: Generate version with pauses
    print("\n" + "=" * 80)
    print("Generating version with pauses between segments...")
    print("=" * 80 + "\n")
    generate_with_pauses(STORY_SCRIPT, VOICES, pause_duration=0.5)
    
    print("\n" + "=" * 80)
    print("All files saved to:", output_dir.absolute())
    print("=" * 80)
    
    print("""
Next time you run this script, it will be much faster because:
  1. Voice conditionals are cached in memory
  2. Voice conditionals are saved to disk (./audiobook_cache/)
  3. On restart, they load from disk instantly
  
Try adding more voices to VOICES dictionary and extending STORY_SCRIPT!
    """)
