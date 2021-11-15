"""@xvdp"""
import torch
from magi.transforms.functional_app import normtorange
from magi.transforms import NormToRange

# pylint: disable=no-member
def test_normtorange():
    data = torch.linspace(0.1, 2, 100)

    x = normtorange(data.clone())
    assert x.min().item() == 0.
    assert x.max().item() == 1.

    x = normtorange(data.clone())
    assert x.min().item() == 0.
    assert x.max().item() == 1.

def test_normtorange_channels():
    data = torch.linspace(0.1, 2, 100).view(1,4,5,5)
    x = normtorange(data.clone(), minimum=[0, 0, 0, 0], maximum=[1,1,1,1])
    for i in range(4):
        assert x[0,i].min().item() == 0. and x[0,i].max().item() == 1.

def test_NormToRange_channels():
    data = torch.linspace(0.1, 2, 100).view(1,4,5,5)
    N = NormToRange(expand_dims=(1,), for_display=True)
    x = N(data)
    for i in range(4):
        assert x[0,i].min().item() == 0. and x[0,i].max().item() == 1.
