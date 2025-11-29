import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_optimal_dtype(device=None):
    """
    Returns the optimal dtype for the given device.
    - CUDA: bfloat16 (best performance)
    - MPS: float16 (bfloat16 has limited support, many ops fall back to CPU)
    - CPU: float32 (most compatible)
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.bfloat16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.float16
        else:
            return torch.float32
    
    device_str = str(device).lower()
    if 'cuda' in device_str:
        return torch.bfloat16
    elif 'mps' in device_str:
        return torch.float16
    else:
        return torch.float32


def get_optimal_dtype_str(device=None):
    """Returns the optimal dtype as a string for config."""
    dtype = get_optimal_dtype(device)
    if dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float16:
        return "float16"
    else:
        return "float32"
