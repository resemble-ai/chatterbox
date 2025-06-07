# my_activations.py
import torch.nn as nn
import torch
from math import pi

class Snake(nn.Module):
    """Exactly the same formula used in diffusers"""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((), alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2

_TABLE = {
    "relu":  nn.ReLU,
    "silu":  nn.SiLU,   # alias: swish
    "swish": nn.SiLU,
    "gelu":  nn.GELU,
    "mish":  nn.Mish,
    "snake": Snake,
}

def get_activation(name: str):
    """Return a freshly-instantiated activation layer, matching diffusers’ registry."""
    name = name.lower()
    if name not in _TABLE:
        raise ValueError(f"Unknown activation '{name}'.  Known: {list(_TABLE)}")
    return _TABLE[name]()
