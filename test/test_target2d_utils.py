import pytest
import torch
from magi.datasets import WIDER
from magi.config import load_dataset_path
from magi.utils.target2d_utils import *




# pylint: disable=no-member
torch.set_default_dtype(torch.float32) # reset float16 if set by previous test

WIDER_ROOT = load_dataset_path("WIDER")

def wider(**kwargs):
    return WIDER(data_root=WIDER_ROOT, **kwargs)

@pytest.mark.skipif(WIDER_ROOT is None, reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_ij():
    W = wider()
    data = W.__getitem__()
    assert data[1].shape == ij__ji(data[1]).shape, ij__ji(data[1].numpy()).shape

    assert data[1].view(-1)[-1] == ij__ji(data[1]).view(-1)[-2]
    assert data[1].view(-1)[0] == ij__ji(data[1]).view(-1)[1]


def test_to_yxhwa():
    W = WIDER()
    data = W.__getitem__()
    c1 = data[1]
    form = data.form[1]
    c2, form2 = pos_offset__pos_pos_mode(c1, form)
    assert form2 == 'xyxy'
    assert torch.all(c2.view((-1,2))[-2] == c1.view((-1,2))[-2])
    assert torch.all(c2.view((-1,2))[-1] == c1.view((-1,2))[-1] + c1.view((-1,2))[-2])


    c3, form3 = pos_pos__center_offset_angle_mode(c2, form2)
    assert form3 == 'xywha'
    assert c3.shape[-1] == 5

    c4, form4 = ij__ji_mode(c3, form3)
    assert form4 == 'yxhwa'
    assert c4.view(-1,5)[0][1] == c3.view(-1,5)[0][0]
    assert c4.view(-1,5)[0][3] == c3.view(-1,5)[0][2]
