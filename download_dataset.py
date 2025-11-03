#!/usr/bin/env python3
"""
Download dataset from Hugging Face and prepare it for lora.py (TESTED & WORKING)

This version has been tested and confirmed working with your dataset.
Audio is stored as bytes in the Parquet file and decoded using soundfile.
"""
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import shutil
import sys
import io

def download_and_prepare_dataset(
    repo_name: str,
    output_dir: str = "./audio_data",
    use_auth_token: bool = False
):
    """
    Download dataset from HF and structure it for lora.py
    
    Args:
        repo_name: HF dataset repo (e.g., "MrEzzat/arabic-tts-dataset")
        output_dir: Output directory (should match AUDIO_DATA_DIR in lora.py)
        use_auth_token: Set to True for private datasets
    """
    print(f"{'='*70}")
    print(f"Downloading Dataset from Hugging Face Hub")
    print(f"{'='*70}")
    print(f"Repository: {repo_name}")
    print(f"Output directory: {output_dir}")
    print(f"Authentication: {'Required' if use_auth_token else 'Not required'}")
    print(f"{'='*70}\n")
    
    # Load dataset from HF
    print(f"Downloading dataset from {repo_name}...")
    print("(This may take a while - downloading 83MB)\n")
    
    try:
        if use_auth_token:
            dataset = load_dataset(repo_name, split='train', token=True)
        else:
            dataset = load_dataset(repo_name, split='train')
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if the repository name is correct")
        print(f"  2. For private datasets, run: huggingface-cli login")
        print(f"  3. Set USE_AUTH_TOKEN = True for private datasets")
        sys.exit(1)
    
    print(f"\n✓ Downloaded {len(dataset)} samples\n")
    
    # Create output directory structure
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    
    # Check if directory exists
    if output_path.exists():
        print(f"⚠️  Warning: Directory {output_path} already exists")
        response = input("Remove and recreate? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(1)
        print(f"Removing existing directory...")
        shutil.rmtree(output_path)
    
    # Create directories
    print(f"\nCreating directory structure...")
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    print(f"✓ Created {output_path}")
    print(f"✓ Created {audio_dir}")
    
    # Import soundfile for saving audio
    try:
        import soundfile as sf
    except ImportError:
        print("\n❌ Error: soundfile library not found")
        print("Install it with: pip install soundfile")
        sys.exit(1)
    
    # Prepare metadata
    print(f"\nProcessing and saving audio files...")
    print("This will take a few minutes...\n")
    metadata_rows = []
    
    # Process each sample
    successful = 0
    failed = 0
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            
            # Generate filename
            filename = f"utterance_{idx+1:04d}.wav"
            audio_path = audio_dir / filename
            
            # Extract audio data from bytes
            # The audio is a dict with 'bytes' and 'path' keys
            audio = sample['audio']
            audio_bytes = audio['bytes']
            
            # Decode audio from bytes using soundfile
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Validate audio
            if audio_data is None or len(audio_data) == 0:
                print(f"  ⚠️  Empty audio for sample {idx}")
                failed += 1
                continue
            
            # Save audio file
            sf.write(str(audio_path), audio_data, sample_rate)
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            
            # Add to metadata
            metadata_rows.append({
                'file_name': f"audio/{filename}",
                'transcription': sample['transcription'],
                'duration_seconds': duration,
                'topic': sample.get('topic', ''),
                'utterance_type': sample.get('utterance_type', ''),
                'topic_category': sample.get('topic_category', '')
            })
            
            successful += 1
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} samples...")
        
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to process sample {idx}: {e}")
            failed += 1
            continue
    
    print(f"\n✓ Saved {successful} audio files")
    if failed > 0:
        print(f"  ⚠️  Failed: {failed} samples")
    
    if successful == 0:
        print(f"\n❌ No audio files were saved! Something went wrong.")
        sys.exit(1)
    
    # Save metadata.csv
    print(f"\nCreating metadata.csv...")
    metadata_path = output_path / "metadata.csv"
    df = pd.DataFrame(metadata_rows)
    df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata.csv with {len(df)} entries")
    
    # Verify structure
    print(f"\n{'='*70}")
    print(f"Verifying dataset structure...")
    print(f"{'='*70}")
    
    audio_files = list(audio_dir.glob("*.wav"))
    
    checks = [
        (output_path.exists(), f"Output directory exists: {output_path}"),
        (metadata_path.exists(), f"metadata.csv exists: {metadata_path}"),
        (audio_dir.exists(), f"audio/ directory exists: {audio_dir}"),
        (len(audio_files) > 0, f"Audio files present: {len(audio_files)} files"),
        (len(df) == len(audio_files), f"Metadata matches audio files: {len(df)} entries"),
    ]
    
    all_passed = True
    for passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {message}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print(f"\n❌ Verification failed! Please check the errors above.")
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Dataset prepared successfully!")
    print(f"{'='*70}")
    print(f"\nDataset location: {output_path.absolute()}")
    print(f"\nStructure:")
    print(f"  {output_path}/")
    print(f"  ├── metadata.csv ({len(df)} entries)")
    print(f"  └── audio/")
    print(f"      └── {len(audio_files)} .wav files")
    
    # Statistics
    if len(df) > 0:
        total_duration = df['duration_seconds'].sum()
        avg_duration = df['duration_seconds'].mean()
        
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Total duration: {total_duration/60:.2f} minutes")
        print(f"  Average duration: {avg_duration:.2f} seconds")
        
        # Check topic distribution
        if 'topic_category' in df.columns and df['topic_category'].notna().any():
            print(f"\nTopic distribution:")
            for category, count in df['topic_category'].value_counts().items():
                if category:  # Skip empty categories
                    print(f"  {category}: {count} samples")
    
    print(f"\n{'='*70}")
    print(f"Ready to train!")
    print(f"{'='*70}")
    print(f"\nTo start training:")
    print(f"  1. Update AUDIO_DATA_DIR in lora.py to: '{output_dir}'")
    print(f"  2. Run: python lora.py")
    print(f"{'='*70}\n")
    
    return output_path

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - UPDATE THESE VALUES
    # ========================================================================
    
    # Hugging Face repository name (format: "username/dataset-name")
    REPO_NAME = "MrEzzat/arabic-tts-dataset"
    
    # Output directory (should match AUDIO_DATA_DIR in lora.py)
    OUTPUT_DIR = "audio_data"
    
    # Set to True for private datasets (requires: huggingface-cli login)
    USE_AUTH_TOKEN = False
    
    # ========================================================================
    
    # Run download
    try:
        download_and_prepare_dataset(REPO_NAME, OUTPUT_DIR, USE_AUTH_TOKEN)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

