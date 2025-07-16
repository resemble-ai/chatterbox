import numpy as np
import torch

import os
import random

# from transformers import set_seed


def set_seed(seed: int = 0):
    """Set the seed
    seed = 0         Generate a random seed
    seed = -1        Disable deterministic algorithms
    0 < seed < 2**32 Set the seed
    Args:
        seed: integer to use as seed
    Returns:
        integer used as seed
    """

    original_seed = seed

    # See for more informations: https://pytorch.org/docs/stable/notes/randomness.html
    if seed == -1:
        # Disable deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Enable deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if seed <= 0:
        # Generate random seed
        # Use default_rng() because it is independent of np.random.seed()
        seed = np.random.default_rng().integers(1, 2**32 - 1)

    assert 0 < seed < 2**32

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return original_seed if original_seed != 0 else seed
