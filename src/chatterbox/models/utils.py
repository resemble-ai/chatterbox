import torch


def get_device() -> str:
    """
    Get the best available device for PyTorch computations.
    
    Priority order:
    1. CUDA - if available (fastest for most operations)
    2. MPS - if available on Apple Silicon (good performance on Mac)
    3. CPU - fallback (always available)
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
