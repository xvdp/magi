import torch
import numpy as np
from torch._C import device
import pytest
from magi.utils import get_broadcastable, broadcast_tensors, slicer


# pylint: disable=no-member
def test_broadcast_tensors():
    a = -0.1556
    b = [[ 0.1201], [-0.7997]]
    c = torch.randn([2,3], device='cuda')
    d = torch.ones((2, 1, 3, 4), dtype=torch.float16)
    e = np.zeros([1, 1, 1, 4])

    out = broadcast_tensors(a,b,c,d,e)
    assert out[3].dtype == torch.float32
    assert out[2].device.type == 'cpu'
    assert all(o.shape == out[0].shape for o in out)
    return out
    
    # f = broadcast_tensors(a,b,c,d,e)

def test_broadcast_tensors2():
    shapes = [(1,), (2, 3), (2, 1), (2, 1, 3, 4), (1, 1, 1, 4)]
    out = broadcast_tensors(*[torch.randn(s) for s in shapes])
    assert all(o.shape == out[0].shape for o in out)
    return out


def test_broadcast_cast():
    shapes = [(1,), (2, 3), (2, 1), (2, 1, 3, 4), (1, 1, 1, 4)]
    out = broadcast_tensors(*[torch.randn(s) for s in shapes], dtype=torch.float16, device='cuda')
    assert out[0].device.type == 'cuda'
    assert out[0].dtype == torch.float16
    assert all(o.shape == out[0].shape for o in out)
    return out

def test_broadcast_front_alinged():
    shapes = [(4,), (1,3,1), (2,1,3,1), (2,1,4)]
    out = broadcast_tensors(*[torch.randn(s) for s in shapes], align=1)
    assert all(o.shape == out[0].shape for o in out)
    return out

def test_broadcast_tensors3():
    a = -0.1556
    b = [5, 2]
    c = torch.tensor(6)
    d = torch.ones((2, 3, 3, 4))
    e = torch.tensor([4, 3])
    out =  broadcast_tensors(a, b, c, d, e)
    assert all(o.shape == out[0].shape for o in out)


def test_get_broadcastable_tensors():
    x = torch.randn([1, 3], dtype=torch.half)
    other = torch.ones([1, 3, 20, 20], device="cuda")
    x = get_broadcastable(x, other)
    assert x.shape == (1, 3, 1, 1)
    assert x.device == other.device
    assert x.dtype == other.dtype

def test_get_broadcastable_axis1():
    x = [0.3, 0.3, 1]
    other = torch.ones([1, 3, 20, 20], device="cuda")
    x = get_broadcastable(x, other)
    assert x.shape == (1, 3, 1, 1)
    assert x.device == other.device
    assert x.dtype == other.dtype


def test_get_broadcastable_axis0():
    x = [0.3, 0.3, 1]
    other = torch.ones([3, 20, 20])
    x = get_broadcastable(x, other, axis=0)
    assert x.shape == (3, 1, 1)
    assert x.device == other.device
    assert x.dtype == other.dtype


def test_slicer():
    ones = torch.ones([2, 3, 10, 20])
    sc = slicer(ones.shape, [0, 3], [1, 2])
    assert ones[sc].shape == torch.Size([1, 3, 10, 1])

    sc=slicer(ones.shape, [0, 1, 2, 3], [1, 2, 3, 4])
    assert ones[sc].shape == torch.Size([1, 1, 1, 1])

@pytest.mark.xfail
def test_slicer_unequal_dims_indices():
    ones = torch.ones([2, 3, 10, 20])
    sc = slicer(ones.shape, [0, 1, 2, 3], [1, 2])
    
@pytest.mark.xfail
def test_slicer_shape_toolong():
    ones = torch.ones([2,3,10,20])
    sc = slicer(ones.shape, [0, 4], [1, 4])
