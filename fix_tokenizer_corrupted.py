#!/usr/bin/env python3
"""
Fix corrupted tokenizer.json file
"""

import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

print("=" * 60)
print("Fix Corrupted tokenizer.json")
print("=" * 60)
print()

model_dir = Path("checkpoints_lora/merged_model")
tokenizer_path = model_dir / "tokenizer.json"

print(f"Tokenizer path: {tokenizer_path.absolute()}")
print(f"Exists: {tokenizer_path.exists()}")

if tokenizer_path.exists():
    size = tokenizer_path.stat().st_size
    print(f"Size: {size} bytes ({size / 1024:.2f} KB)")
    
    if size == 0:
        print("❌ File is EMPTY (0 bytes)!")
    elif size < 1000:
        print("⚠️  File is suspiciously small!")
    else:
        print("✅ File size looks OK")
        # Try to read it
        try:
            with open(tokenizer_path, 'r') as f:
                content = f.read(100)
                print(f"Content preview: {content[:50]}...")
                print("✅ File appears to be readable")
                print()
                print("File seems OK. The issue might be elsewhere.")
                print("Try checking the Chatterbox source code for tokenizer loading.")
                exit(0)
        except Exception as e:
            print(f"❌ Cannot read file: {e}")
else:
    print("❌ File doesn't exist!")

print()
print("=" * 60)
print("Downloading fresh tokenizer.json from HuggingFace...")
print("=" * 60)
print()

# Backup old file if it exists
if tokenizer_path.exists():
    backup_path = model_dir / "tokenizer.json.backup"
    print(f"Backing up old file to: {backup_path}")
    shutil.move(str(tokenizer_path), str(backup_path))
    print("✅ Backup created")
    print()

# Download fresh tokenizer
try:
    print("📥 Downloading from ResembleAI/chatterbox...")
    downloaded_path = hf_hub_download(
        repo_id="ResembleAI/chatterbox",
        filename="tokenizer.json",
        force_download=True  # Force fresh download
    )
    print(f"✅ Downloaded to: {downloaded_path}")
    
    # Copy to model directory
    print(f"📋 Copying to: {tokenizer_path}")
    shutil.copy(downloaded_path, tokenizer_path)
    
    # Verify
    if tokenizer_path.exists():
        size = tokenizer_path.stat().st_size
        print(f"✅ Copied successfully!")
        print(f"   Size: {size} bytes ({size / 1024:.2f} KB)")
        
        # Try to read it
        try:
            with open(tokenizer_path, 'r') as f:
                content = f.read(100)
                print(f"   Content preview: {content[:50]}...")
            print("✅ File is readable!")
            print()
            print("=" * 60)
            print("🎉 tokenizer.json fixed!")
            print("=" * 60)
            print()
            print("Now run: python diagnose_and_fix.py")
            print()
        except Exception as e:
            print(f"❌ Still cannot read file: {e}")
    else:
        print("❌ Copy failed!")
        
except Exception as e:
    print(f"❌ Download failed: {e}")
    print()
    print("Manual fix:")
    print("1. Visit: https://huggingface.co/ResembleAI/chatterbox/blob/main/tokenizer.json")
    print("2. Click 'Download'")
    print(f"3. Save to: {tokenizer_path}")
    print()

