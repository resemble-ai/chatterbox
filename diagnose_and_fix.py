#!/usr/bin/env python3
"""
Diagnose and fix fine-tuned model loading issues
"""

import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

print("=" * 60)
print("Fine-Tuned Model Diagnostic & Fix")
print("=" * 60)
print()

# Check model directory
model_dir = Path("checkpoints_lora/merged_model")
print(f"Model directory: {model_dir.absolute()}")
print(f"Exists: {model_dir.exists()}")
print()

if not model_dir.exists():
    print("❌ Model directory doesn't exist!")
    print("Please run training first: python lora_fixed.py")
    exit(1)

# List all files
print("Files in merged_model:")
files_found = {}
for file in sorted(model_dir.iterdir()):
    size = file.stat().st_size / (1024 * 1024)  # MB
    files_found[file.name] = file
    print(f"  {'✅' if size > 0 else '❌'} {file.name:30s} ({size:>8.1f} MB)")
print()

# Check required files
required_files = {
    "t3_mtl23ls_v2.safetensors": "T3 model (main)",
    "ve.pt": "Voice encoder",
    "s3gen.pt": "S3 generator",
    "tokenizer.json": "Tokenizer",
}

print("Checking required files:")
missing_files = []
for filename, description in required_files.items():
    if filename in files_found:
        file_path = files_found[filename]
        size = file_path.stat().st_size
        if size > 0:
            print(f"  ✅ {filename:30s} - {description}")
        else:
            print(f"  ❌ {filename:30s} - {description} (0 bytes - corrupted!)")
            missing_files.append(filename)
    else:
        print(f"  ❌ {filename:30s} - {description} (missing!)")
        missing_files.append(filename)
print()

# Fix missing t3_mtl23ls_v2.safetensors
if "t3_mtl23ls_v2.safetensors" in missing_files and "t3_cfg.pt" in files_found:
    print("🔧 Fixing: Converting t3_cfg.pt to t3_mtl23ls_v2.safetensors...")
    try:
        import torch
        from safetensors.torch import save_file
        
        t3_state = torch.load(model_dir / "t3_cfg.pt", map_location='cpu')
        save_file(t3_state, str(model_dir / "t3_mtl23ls_v2.safetensors"))
        print("  ✅ Converted successfully!")
        missing_files.remove("t3_mtl23ls_v2.safetensors")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    print()

# Fix missing tokenizer.json
if "tokenizer.json" in missing_files:
    print("🔧 Fixing: Downloading tokenizer.json from HuggingFace...")
    try:
        tokenizer_path = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="tokenizer.json"
        )
        shutil.copy(tokenizer_path, model_dir / "tokenizer.json")
        print("  ✅ Downloaded and copied successfully!")
        missing_files.remove("tokenizer.json")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    print()

# Final check
if missing_files:
    print("❌ Still missing files:")
    for f in missing_files:
        print(f"  - {f}")
    print()
    print("Cannot proceed. Please check training output for errors.")
    exit(1)

print("=" * 60)
print("✅ All required files present!")
print("=" * 60)
print()

# Now test loading
print("Testing model loading...")
print()

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try with absolute path
    model_path_abs = str(model_dir.absolute())
    print(f"Loading from: {model_path_abs}")
    
    model = ChatterboxMultilingualTTS.from_local(model_path_abs, device=device)
    
    print("✅ Fine-tuned model loaded successfully!")
    print()
    print("=" * 60)
    print("🎉 SUCCESS!")
    print("=" * 60)
    print()
    print("Your fine-tuned model is ready to use!")
    print()
    print("Run: python test_finetuned.py")
    print()
    
except Exception as e:
    print(f"❌ Loading failed: {e}")
    print()
    print("Detailed error:")
    import traceback
    traceback.print_exc()
    print()
    print("=" * 60)
    print("Troubleshooting:")
    print("=" * 60)
    print()
    print("1. Check if all files are valid (not corrupted)")
    print("2. Try running from the project root directory")
    print("3. Check the error message above for specific issues")
    print()

