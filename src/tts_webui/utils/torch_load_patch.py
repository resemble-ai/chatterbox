"""
Utility module to monkeypatch torch.load to always use weights_only=False.
This addresses the issue with PyTorch 2.6 changing the default value of weights_only from False to True.
"""
import functools
import torch
import logging

# Store the original torch.load function
original_torch_load = torch.load

@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    """
    Wrapper for torch.load that always sets weights_only=False.
    This addresses the issue with PyTorch 2.6 changing the default value of weights_only from False to True.
    """
    # Explicitly set weights_only to False if not provided
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    
    return original_torch_load(*args, **kwargs)

def apply_torch_load_patch():
    """
    Apply the monkeypatch to torch.load.
    """
    torch.load = patched_torch_load
    logging.info("Applied monkeypatch to torch.load to always use weights_only=False")

def restore_original_torch_load():
    """
    Restore the original torch.load function.
    """
    torch.load = original_torch_load
    logging.info("Restored original torch.load function")
