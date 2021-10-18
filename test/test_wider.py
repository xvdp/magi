"""@xvdp
modify WIDER_ROOT to ensure runs on wider
"""
import os.path as osp
import torch
from torchvision import transforms as TT
import pytest
from magi.datasets import WIDER

# pylint: disable=no-member
WIDER_ROOT = "/media/z/Elements1/data/Face/WIDER"
def _wider_folder():
    return osp.isdir(WIDER_ROOT)

def wider(**kwargs):
    return WIDER(data_root=WIDER_ROOT, **kwargs)

@pytest.mark.skipif(not _wider_folder(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_widerload():
    W = wider()

    item = W.samples[99]
    name = item.get("tags", "name")[0]
    assert osp.isfile(name), f"image not found, {name}"
    assert len(item) == len(W._tags[1:])
    assert item.tags == W._tags[1:]
    assert item.meta == W._meta[1:]
    assert item.dtype == W._dtype[1:]

    item = W.__getitem__()
    assert item is not None
    assert isinstance(item[0], torch.Tensor)
    assert item.tags == W._tags
    assert item.meta == W._meta
    assert item.dtype == W._dtype


@pytest.mark.skipif(not _wider_folder() or not torch.cuda.is_available(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_wider_cuda():
    W = WIDER(data_root=WIDER_ROOT, device="cuda", dtype="float16")
    data = W.__getitem__()
    assert data[0].dtype == torch.float16
    assert data[0].device.type == "cuda"

@pytest.mark.skipif(not _wider_folder(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_wider_torchtransform():
    transf = TT.CenterCrop(size=(224,224))
    W = WIDER(data_root=WIDER_ROOT, device="cuda", dtype="float16", transforms=transf, channels=1)
    data = W.__getitem__()
    assert tuple(data[0].shape) == (1,1,224,224)

@pytest.mark.skipif(not _wider_folder(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_wider_test():
    W = WIDER(data_root=WIDER_ROOT, mode="test")
    idx = 0
    assert len(W.samples[idx]) == 5
    data = W.__getitem__(idx)
    assert len(W.samples[idx]) == len(data) - 1, "__getitem__ should return deepcopy, not instance in memory"

@pytest.mark.skipif(not _wider_folder(), reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_wider_tags():
    W = wider(tags=['blur'], mode="train")
    assert W.samples[0].tags == ['bbox', 'name', 'blur']
    assert W[0].tags == ['image', 'bbox', 'blur']

