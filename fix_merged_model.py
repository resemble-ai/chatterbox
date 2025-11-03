#!/usr/bin/env python3
"""
Fix the merged model format to match what ChatterboxMultilingualTTS.from_local expects
"""

import torch
from pathlib import Path
from safetensors.torch import save_file

print("=" * 60)
print("Fixing Merged Model Format")
print("=" * 60)
print()

merged_dir = Path("checkpoints_lora/merged_model")

# Check if t3_cfg.pt exists
t3_cfg_path = merged_dir / "t3_mtl23ls_v2.pt"
if not t3_cfg_path.exists():
    print(f"❌ Error: {t3_cfg_path} not found!")
    print("Make sure training has completed and merged model was saved.")
    exit(1)

print(f"✅ Found: {t3_cfg_path}")

# Load the t3 state dict
print("Loading T3 model state...")
t3_state = torch.load(t3_cfg_path, map_location='cpu')
print(f"✅ Loaded {len(t3_state)} parameters")

# Save as safetensors format
output_path = merged_dir / "t3_mtl23ls_v2.safetensors"
print(f"\nSaving to: {output_path}")
save_file(t3_state, str(output_path))

# Verify the file was created
if output_path.exists():
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Created: {output_path} ({size_mb:.1f} MB)")
    print()
    print("=" * 60)
    print("✅ Model format fixed!")
    print("=" * 60)
    print()
    print("You can now test your model:")
    print("  python test_arabic_tts.py")
    print()
else:
    print(f"❌ Failed to create {output_path}")
    exit(1)

