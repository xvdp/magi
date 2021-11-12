"""@xvdp
"""
import random
import pytest
import torch
import numpy as np
from magi.utils import torch_dtype, get_broadcastable, tensor_apply, broadcast_tensors

# pylint: disable=no-member
def test_tensor_apply():
    
    z = torch.linspace(-1, 2, 28).view(2, 2, 7)
    z00 = tensor_apply(z, "min", hold_axis=2)[0]
    assert torch.equal(z00, z[0,0])

    z_00 = tensor_apply(z, "min", hold_axis=0)[0]
    assert torch.equal(z_00, z[:,0,0])

    z0_0 = tensor_apply(z, "min", hold_axis=1)[0]
    assert torch.equal(z0_0, z[0,:,0])

    assert tensor_apply(z, "min", hold_axis=0, keepdims=True)[0].shape == (2,1,1)
    assert tensor_apply(z, "min", hold_axis=1, keepdims=True)[0].shape == (1,2,1)
    assert tensor_apply(z, "min", hold_axis=2, keepdims=True)[0].shape == (1,1,7)

def test_tensor_apply_list():
    x = torch.linspace(-1, 2, 60).view(2, 3, 2, 5)
    assert tensor_apply(x, hold_axis=[1,3], func="max", keepdims=True)[0].shape == (1,3,1,5)
    assert tensor_apply(x, hold_axis=[2,3], func="max", keepdims=True)[0].shape == (1,1,2,5)
    assert tensor_apply(x, hold_axis=[1,2,3], func="max", keepdims=True)[0].shape == (1,3,2,5)
    assert torch.equal(tensor_apply(x, hold_axis=[0,1,2,3], func="max", keepdims=True), x)

    assert torch.equal(tensor_apply(x, hold_axis=[1,2,3], func="max", keepdims=True)[0], x.max(axis=0, keepdims=True)[0])
    assert torch.equal(tensor_apply(x, hold_axis=[1,2,3], func="min", keepdims=True)[0], x.min(axis=0, keepdims=True)[0])
    assert torch.equal(tensor_apply(x, hold_axis=[0,2,3], func="mean", keepdims=True), x.mean(axis=1, keepdims=True))
    assert torch.equal(tensor_apply(x, hold_axis=[0,1,3], func="prod", keepdims=True), x.prod(axis=2, keepdims=True))

def test_get_broadcastable():
    x = torch.randn([2,3,30,30], dtype=torch.half, device="cuda")
    y = get_broadcastable(5, x)
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert y.ndim == x.ndim
    y = x*y

def test_reduce_edge_cond():
    tensor = torch.randn([random.randint(1,2), 3,*torch.randint(2,128, (random.randint(0, 4),))], dtype=torch.float)
    xshape = [[1, tensor.shape[i], random.randint(2,5)][random.randint(0,2)] for i in range(tensor.ndim)][:random.randint(2, tensor.ndim)]
    x = torch.ones(xshape, dtype=torch.int)
    print(f"tensor.shape {tensor.shape}, x.shape {xshape}")
    x = get_broadcastable(x, tensor)
    assert x.dtype == tensor.dtype
    assert x.ndim == tensor.ndim
    x = tensor*x

def test_reduce_edge_cond_np():

    tensor = torch.randn([random.randint(1,2), 3,*torch.randint(2,128, (random.randint(0, 4),))], dtype=torch.float)
    xshape = [[1, tensor.shape[i], random.randint(2,5)][random.randint(0,2)] for i in range(tensor.ndim)][:random.randint(2, tensor.ndim)]
    x = np.ones(xshape)
    print(f"tensor.shape {tensor.shape}, x.shape {xshape}")
    x = get_broadcastable(x, tensor)
    assert x.dtype == tensor.dtype
    assert x.ndim == tensor.ndim
    x = tensor*x

def test_reduce_broadcast_axis():

    tensor = torch.randn([3,3,3,4])
    x = torch.tensor([4.,5., 5.], device='cuda')
    for i in range(3):
        y = get_broadcastable(x, tensor, axis=i)
        assert y.shape[i] == len(x)
        t = tensor.sub(y).div(y)
    y = get_broadcastable(x, tensor, axis=3)
    assert y.shape[i] == 1
    t = tensor.sub(y).div(y)

def test_torch_dtype():
    dtype = torch_dtype(None)
    assert dtype is None

    dtype = torch_dtype(None, force_default=True)
    assert dtype == torch.get_default_dtype()


    dtype = torch_dtype("half")
    assert dtype == torch.float16

    
    dtype = torch_dtype(["double", None, torch.float32, "uint8", "int", "int64"])
    assert dtype[0] == torch.float64
    assert dtype[1] is None
    assert dtype[2] == torch.float32
    assert dtype[3] == torch.uint8
    assert dtype[4] == torch.int32
    assert dtype[5] == torch.int64





@pytest.mark.xfail(reason="dtype passes")
def test_torch_dtype_fail():
    _invalid = "int34"
    dtype = torch_dtype(_invalid)
    assert dtype is not None, f"dtype should be none with invalid dtype {_invalid}, got {dtype}"


