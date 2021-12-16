"""@xvdp test file for utils/imageio.py
"""
import os
import os.path as osp
import numpy as np
import torch
import pytest

from magi.utils.imageio import get_cache_name, open_url, np_fix_channels
from _test_common import assert_msg, source_url


# pylint: disable=no-member
torch.set_default_dtype(torch.float32) # reset float16 if set by previous test


# open_url()
def test_openio_open_url_torch_no_cache():
    url = source_url()
    if url:
        dtype="float32"
        img = open_url(url, cache_name=None, dtype=dtype, out_type="torch", channels=None)
        assert_msg("dtype", torch.__dict__[dtype], img.dtype, name=url, fun="imageio.open_url")

        dtype="float16"
        img = open_url(url, cache_name=None, dtype=dtype, out_type="torch", channels=None)
        assert_msg("dtype", torch.__dict__[dtype], img.dtype, name=url, fun="imageio.open_url")

def test_openio_open_url_numpy_no_cache():
    url = source_url()
    if url:
        dtype="float32"
        out_type="numpy"
        img = open_url(url, cache_name=None, dtype=dtype, out_type=out_type, channels=None)
        assert_msg("dtype", dtype, img.dtype.name, name=url, fun="imageio.open_url")

        dtype="uint8"
        out_type="numpy"
        img = open_url(url, cache_name=None, dtype=dtype, out_type=out_type, channels=None)
        assert_msg("dtype", dtype, img.dtype.name, name=url, fun="imageio.open_url")

def test_openio_open_url_numpy_cache():
    url = source_url()
    if url:

        cached = get_cache_name(url)
        dtype="float32"
        out_type="numpy"
        img = open_url(url, cache_name=cached, dtype=dtype, out_type=out_type, channels=None)
        assert_msg("dtype", dtype, img.dtype.name, name=url, fun="imageio.open_url")
        assert osp.isfile(cached), f" file '{cached}' not found"

        dtype="uint8"
        out_type="numpy"
        channels=1
        img = open_url(url, cache_name=cached, dtype=dtype, out_type=out_type, channels=channels)
        assert_msg("dtype", dtype, img.dtype.name, name=url, fun="imageio.open_url")
        assert_msg("channels", channels, int(img.shape[-1]), name=url, fun="imageio.open_url")
        if osp.isfile(cached):
            os.remove(cached)

def test_openio_np_fix_channels():
    # pytest test_openio.py::test_openio_np_fix_channels
    # uint8, 3->1, 4>1, 3>4, 1>3
    # float32
    # pylint: disable=too-many-function-args
    _z = np.random.randint(0, 255, 100*100*3, dtype="uint8").reshape(100,100,3)
    _z[0,0] = 255
    _z[-1,-1] = 0
    channels = 1
    _z1 = np_fix_channels(_z, channels)
    assert _z1.dtype == _z.dtype
    assert channels == _z1.shape[-1], f"channels {channels} ?= {_z1.shape[-1]}"
    assert _z1.max() == 255, f"max changed from {_z.max()} to {_z1.max()}"

