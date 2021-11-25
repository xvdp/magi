"""@xvdp"""
import torch
from magi.transforms.functional_base import get_sample_like
# pylint: disable=no-member
torch.set_default_dtype(torch.float32) # reset float16 if set by previous test

def test_sample_like():
    data = torch.linspace(0.1, 0.9, 100, dtype=torch.float32).view(1,4, 5,5)
    x = get_sample_like(0.5, data)
    assert x.shape == torch.Size([1, 1, 1, 1])

    data = torch.linspace(0.1, 0.9, 100, device='cuda', dtype=torch.float16).view(1,4, 5,5)
    x = get_sample_like([0.5,0.5,0.5,0.5], data)
    assert x.shape == torch.Size([1, 4, 1, 1])
    assert x.device == data.device
    assert x.dtype == data.dtype
