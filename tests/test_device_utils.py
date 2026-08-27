import pytest
import torch

from chatterbox.models.utils import load_map_location


@pytest.mark.parametrize(
    "device, expected",
    [
        ("cpu", torch.device("cpu")),
        (torch.device("cpu"), torch.device("cpu")),
        ("mps", torch.device("cpu")),
        (torch.device("mps"), torch.device("cpu")),
        ("npu", torch.device("cpu")),
        (torch.device("npu"), torch.device("cpu")),
        ("xpu", torch.device("cpu")),
        ("cuda", None),
        (torch.device("cuda"), None),
        (torch.device("cuda:0"), None),
    ],
)
def test_load_map_location(device, expected):
    got = load_map_location(device)
    if expected is None:
        assert got is None
    else:
        assert got == expected


def test_cpu_device_object_does_not_fall_through_to_cuda():
    # Regression test for https://github.com/resemble-ai/chatterbox/issues/533
    # ``torch.device("cpu")`` does not compare equal to the string ``"cpu"``,
    # so ``device in ["cpu", "mps"]`` is False and ``from_local`` used to set
    # ``map_location=None``, crashing on CPU-only machines.
    assert torch.device("cpu") not in ["cpu", "mps"]
    assert load_map_location(torch.device("cpu")) == torch.device("cpu")