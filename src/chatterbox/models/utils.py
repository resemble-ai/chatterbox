class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_optimal_dtype_str(device=None):
    """Return an optimal dtype string for a given device.

    - MPS: use `float16` (better support than bfloat16)
    - CUDA: use `bfloat16` (optimal for Ampere+)
    - CPU/other: use `float32`

    The `device` argument may be a `torch.device`, a device string like
    `'cuda:0'` or `'mps'`, or `None` (treated as CPU).
    """
    try:
        import torch
    except Exception:
        return "float32"

    if device is None:
        device_type = "cpu"
    elif isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
        if ":" in device_type:
            device_type = device_type.split(":", 1)[0]

    device_type = (device_type or "cpu").lower()

    if device_type == "mps":
        return "float16"
    if device_type == "cuda":
        return "bfloat16"
    return "float32"
