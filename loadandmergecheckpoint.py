import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Import Chatterbox components
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
import shutil

# Hardcoded configuration - MODIFY THESE
CHECKPOINT_PATH = "./checkpoints_lora/checkpoint_epoch7_step1248.pt"  # Path to your checkpoint
OUTPUT_DIR = "./checkpoints_lora/merged_model"  # Where to save the merged model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA configuration (must match training config)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALayer(nn.Module):
    """LoRA adapter layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Proper initialization
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/np.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_dropout(x)
        result = result @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    """Inject LoRA layers into the model"""
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                if min(module.in_features, module.out_features) < rank:
                    continue
                    
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                lora_layer = lora_layer.to(device)
                lora_layers[name] = lora_layer
                
                original_forward = module.forward
                def make_new_forward(orig_forward, lora):
                    def new_forward(x):
                        return orig_forward(x) + lora(x)
                    return new_forward
                
                module.forward = make_new_forward(original_forward, lora_layer)
    
    return lora_layers


def merge_lora_weights(model: ChatterboxTTS, lora_layers: Dict[str, LoRALayer]):
    """Merge LoRA weights into the base model"""
    with torch.no_grad():
        for name, lora_layer in lora_layers.items():
            # Find the corresponding linear layer in the model
            parts = name.split('.')
            module = model.t3.tfmr
            for part in parts[:-1]:
                module = getattr(module, part)
            linear_layer = getattr(module, parts[-1])
            
            # Compute LoRA update: W' = W + BA * scaling
            lora_update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
            
            # Add to original weights
            linear_layer.weight.data += lora_update
    
    return model


def save_merged_model(model: ChatterboxTTS, output_dir: Path):
    """Save the merged model components"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each component
    print("Saving model components...")
    torch.save(model.ve.state_dict(), output_dir / "ve.pt")
    torch.save(model.t3.state_dict(), output_dir / "t3_cfg.pt")
    torch.save(model.s3gen.state_dict(), output_dir / "s3gen.pt")
    
    # Copy tokenizer
    print("Copying tokenizer...")
    tokenizer_path = Path(hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json"))
    shutil.copy(tokenizer_path, output_dir / "tokenizer.json")
    
    # Save conditionals if they exist
    if model.conds:
        model.conds.save(output_dir / "conds.pt")
    
    print(f"Saved merged model to {output_dir}")


def main():
    """Main function to load checkpoint and merge LoRA weights"""
    print(f"Loading and merging checkpoint from: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    
    # Load the checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Step: {checkpoint['step']}")
    print(f"  - Loss: {checkpoint['loss']:.4f}")
    print(f"  - LoRA weights found: {len(checkpoint['lora_state_dict'])}")
    
    # Load base model
    print("\nLoading base Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    
    # Inject LoRA layers
    print("\nInjecting LoRA layers...")
    lora_layers = inject_lora_layers(
        model.t3.tfmr,
        TARGET_MODULES,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    print(f"Injected {len(lora_layers)} LoRA layers")
    
    # Load LoRA weights from checkpoint
    print("\nLoading LoRA weights from checkpoint...")
    loaded_count = 0
    for name, param in checkpoint['lora_state_dict'].items():
        # Extract layer name from parameter name (remove .lora_A or .lora_B)
        layer_name = name.rsplit('.', 1)[0]
        param_type = name.rsplit('.', 1)[1]
        
        if layer_name in lora_layers:
            if param_type == 'lora_A':
                lora_layers[layer_name].lora_A.data = param.to(DEVICE)
            elif param_type == 'lora_B':
                lora_layers[layer_name].lora_B.data = param.to(DEVICE)
            loaded_count += 1
    
    print(f"Loaded {loaded_count} LoRA parameters")
    
    # Merge LoRA weights into base model
    print("\nMerging LoRA weights into base model...")
    model = merge_lora_weights(model, lora_layers)
    
    # Save merged model
    output_path = Path(OUTPUT_DIR)
    print(f"\nSaving merged model to {output_path}...")
    save_merged_model(model, output_path)
    
    print("\n" + "=" * 50)
    print("SUCCESS! Merged model saved.")
    print("\nTo use the merged model:")
    print(f"  model = ChatterboxTTS.from_local('{OUTPUT_DIR}', device='{DEVICE}')")


def verify_merged_model(output_dir: str):
    """Optional: Verify the merged model can be loaded"""
    print("\n" + "-" * 50)
    print("Verifying merged model can be loaded...")
    
    try:
        model = ChatterboxTTS.from_local(output_dir, DEVICE)
        print("✓ Model loaded successfully!")
        
        # Test generation with a simple text
        test_text = "Testing the merged model."
        print(f"\nGenerating test audio: '{test_text}'")
        wavs = model.generate(test_text)
        print(f"✓ Generated audio with shape: {wavs[0].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading merged model: {e}")
        return False


if __name__ == "__main__":
    main()