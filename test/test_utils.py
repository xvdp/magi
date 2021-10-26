"""@xvdp
"""
import random
import pytest
import torch
import numpy as np
from magi.utils import torch_dtype, reduce_to

# pylint: disable=no-member
def test_reduce_to():
    x = torch.randn([2,3,30,30], dtype=torch.half, device="cuda")
    y = reduce_to(5, x)
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert y.ndim == x.ndim
    y = x*y

def test_reduce_edge_cond():
    tensor = torch.randn([random.randint(1,2), 3,*torch.randint(2,128, (random.randint(0, 4),))], dtype=torch.float)
    xshape = [[1, tensor.shape[i], random.randint(2,5)][random.randint(0,2)] for i in range(tensor.ndim)][:random.randint(2, tensor.ndim)]
    x = torch.ones(xshape, dtype=torch.int)
    print(f"tensor.shape {tensor.shape}, x.shape {xshape}")
    x = reduce_to(x, tensor)
    assert x.dtype == tensor.dtype
    assert x.ndim == tensor.ndim
    x = tensor*x

def test_reduce_edge_cond_np():

    tensor = torch.randn([random.randint(1,2), 3,*torch.randint(2,128, (random.randint(0, 4),))], dtype=torch.float)
    xshape = [[1, tensor.shape[i], random.randint(2,5)][random.randint(0,2)] for i in range(tensor.ndim)][:random.randint(2, tensor.ndim)]
    x = np.ones(xshape)
    print(f"tensor.shape {tensor.shape}, x.shape {xshape}")
    x = reduce_to(x, tensor)
    assert x.dtype == tensor.dtype
    assert x.ndim == tensor.ndim
    x = tensor*x

def test_reduce_broadcast_axis():

    tensor = torch.randn([3,3,3,4])
    x = torch.tensor([4.,5., 5.], device='cuda')
    for i in range(3):
        y = reduce_to(x, tensor, axis=i)
        assert y.shape[i] == len(x)
        t = tensor.sub(y).div(y)
    y = reduce_to(x, tensor, axis=3)
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


