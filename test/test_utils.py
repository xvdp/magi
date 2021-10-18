"""@xvdp
"""
import pytest
from magi.utils import torch_dtype
import torch

# pylint: disable=no-member
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
