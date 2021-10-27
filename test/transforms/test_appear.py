"""@xvdp
"""
import pytest
import torch
from magi.transforms import Normalize, UnNormalize, NormToRange
from magi.datasets import Noise, ImageNet, WIDER
from magi import config

# pylint: disable=no-member
IMAGENET_ROOT = config.load_dataset_path("ImageNet")
WIDER_ROOT = config.load_dataset_path("ImageNet")

# def get_dset_item(dset=None):
#     dsets = [ImageNet, WIDER, Noise]

def test_normalize():

    N = Normalize()
    if IMAGENET_ROOT is not None:
        dset = ImageNet(mode='val')
    else:
        dset = Noise()

    item = dset.__getitem__()
    mean = item[0].mean().item()
    std = item[0].std().item()

    N(item)

    nmean = item[0].mean().item()
    nstd = item[0].std().item()

    assert nmean != mean
    assert nstd != std


def test_norm_grad():
    # check default options, config.FOR_DISPLAY is False
    N = Normalize()
    assert N.for_display is None
    assert config.FOR_DISPLAY == False
    
    # check that setting grad, overrids config.FOR_DISPLAY
    if IMAGENET_ROOT is not None:
        dset = ImageNet(mode='val', grad=True, for_display=True)
    else:
        dset = Noise(grad=True, for_display=True)
    assert not config.FOR_DISPLAY, "grad shouldve overriten for display, didnt"
    d = dset.__getitem__()
    print(type(d[0]))
    assert d[0].requires_grad, f"grad should be set, it isnt {d[0]}, {d[0].requires_grad}"

    x = N(d)
    assert torch.all(torch.eq(d[0], x[0])).item(), f"norm w grad should clobber input"

def test_not_for_display():
    """
    """
    N = Normalize()
    dset = ImageNet(mode='val', for_display=False)
    assert not config.FOR_DISPLAY, f"dataset should change for display"
    d = dset.__getitem__()
    x = N(d)
    assert torch.all(torch.eq(d[0], x[0])).item(), f"norm not for display should clobber input"

def test_for_display_local():
    # check that passing 'for_display' on calls without gradient overrides for display but does not set Config
    dset = ImageNet(mode='val', for_display=False)
    N = Normalize(for_display=True)
    assert not config.FOR_DISPLAY, f"transform should NOT change FOR_DISPLAY"
    assert N.for_display, f"transform should pass its own for_display"
    d = dset.__getitem__()
    x = N(d)
    assert not torch.all(torch.eq(d[0], x[0])).item(), f"norm for display should NOT clobber input"

def test_block_clone_grad():
    dset = ImageNet(mode='val', for_display=True)
    assert config.FOR_DISPLAY, f"FOR_DISPLAY should be False, globally"
    N = Normalize()
    d = dset.__getitem__()
    d[0].requires_grad =True
    x = N(d)
    assert not config.FOR_DISPLAY, f"FOR_DISPLAY should be False, globally, "


# TODO NormToRange- 0,1, -4,10

# def test_for_display_grad():
#     dset = ImageNet(mode='val', for_display=True)
#     N = Normalize()
#     assert config.FOR_DISPLAY, f"transform should NOT change FOR_DISPLAY"
#     d = dset.__getitem__()
#     x = N(d)
#     assert not torch.all(torch.eq(d[0], x[0])).item(), f"norm for display should NOTclobber input"




# @pytest.mark.xfail
