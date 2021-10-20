"""@xvdp"""
import os.path as osp
import torch
from torchvision import transforms as TT
from magi.datasets import Noise

# pylint: disable=no-member

def test_noise_default():
    n = Noise()
    assert n._counter == 0
    item = n[0]
    assert n._counter == 1
    item = n[0]
    assert n._counter == 2

def test_noise():
    noise_type = ("normal", "uniform")
    n = Noise(size=(128,128,128), noise_type=noise_type)
    n.classes == noise_type
    a = n[0]
    assert a[0].ndim == 5
    assert a[1] in (0,1)
    a = n[0]
    assert a[1] in (0,1)
    a = n[0]
    assert a[1] in (0,1)


    