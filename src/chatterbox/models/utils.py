import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_map_location(device):
    """Resolve ``map_location`` for ``torch.load`` given a target device.

    Accepts either a ``str`` (``"cpu"``, ``"cuda"``, ``"npu"``, ...) or a
    ``torch.device``. CUDA-saved checkpoints cannot be deserialized onto a
    non-CUDA device directly, so for every non-CUDA accelerator we load onto
    CPU first and move the model to the target device afterwards (the caller
    performs ``.to(device)``).
    """
    device = torch.device(device)
    return None if device.type == "cuda" else torch.device("cpu")