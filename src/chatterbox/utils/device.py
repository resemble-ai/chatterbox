import torch


def resolve_device(device=None):
    """Resolve the target device, with auto-detection if not specified.

    Args:
        device: Target device string or torch.device. If None, auto-detects
                the best available device (cuda > mps > cpu).
    """
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = str(device)
    if device == "mps" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled. Falling back to CPU.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device. Falling back to CPU.")
        return "cpu"

    return device
