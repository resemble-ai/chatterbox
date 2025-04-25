# Common transformations used by synthesizers
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def pack(arrays, seq_len: int=None, pad_value=0):
    """
    Given a list of length B of array-like objects of shapes (Ti, ...), packs them in a single tensor of
    shape (B, T, ...) by padding each individual array on the right.

    :param arrays: a list of array-like objects of matching shapes except for the first axis.
    :param seq_len: the value of T. It must be the maximum of the lengths Ti of the arrays at
    minimum. Will default to that value if None.
    :param pad_value: the value to pad the arrays with.
    :return: a (B, T, ...) tensor
    """
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # Convert lists to np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # Convert to tensor and handle device
    device = None
    if isinstance(arrays[0], torch.Tensor):
        tensors = arrays
        device = tensors[0].device
    else:
        tensors = [torch.as_tensor(array) for array in arrays]

    # Fill the packed tensor with the array data
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)

    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor

    return packed_tensor
