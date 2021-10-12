"""@xvdp
modify WIDER_ROOT to ensure runs on wider 
"""
import os.path as osp
import torch
import pytest
from magi.datasets import WIDER


WIDER_ROOT = "/media/z/Elements1/data/Face/WIDER"
def _wider_folder():
    return osp.isdir(WIDER_ROOT)

@pytest.mark.skipif(not _wider_folder(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_widerload():
    W = WIDER(data_root=WIDER_ROOT)


    item = W.images[99]
    name = item.get("tags", "name")[0]
    assert osp.isfile(name), f"image not found, {name}"
    assert len(item) == 12

    data = W.__getitem__()
    assert data is not None

    assert isinstance(data[0], torch.Tensor)