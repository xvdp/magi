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


def test_flip_modes():
    flip_modes = ['yxhw', 'xywh', 'yxhwa', 'xywha', 'yxyx', 'xyxy', 'xpath', 'ypath']
    for i, m in enumerate([m for m in config.BoxMode.__members__]):
        assert mode_ij__ji(m) == flip_modes[i]

@pytest.mark.skipif(WIDER_ROOT is None, reason=f"Could not find Wider Folder, adjust WIDER_ROOT in {__file__} to test")
def test_conversions():
    W = WIDER()
    data = W.__getitem__()
    c1 = data[1]
    n1 = c1.numpy()
    form1 = data.form[1]

    c2, form2 = pos_offset__pos_pos_mode(c1, form1)
    n2 = pos_offset__pos_pos(n1)
    assert torch.equal(torch.as_tensor(n2), c2)
    assert c2.is_contiguous()

    c3, form3 = pos_pos__pos_offset_mode(c2, form2)
    assert torch.equal(c3, c1) and form3 == form1
    n3 = pos_pos__pos_offset(n2)
    assert torch.equal(torch.as_tensor(n3), c3)
    assert c3.is_contiguous()

    c4, form4 = pos_pos__path_mode(c2, form2)
    n4 = pos_pos__path(n2)
    assert torch.equal(torch.as_tensor(n4), c4)
    assert c4.is_contiguous()

    c5, form5 = path__pos_pos_mode(c4, form4)
    n5 = path__pos_pos(n4)
    assert torch.equal(torch.as_tensor(n5), c5)
    assert torch.equal(c2, c5) and form5 == form2
    assert c5.is_contiguous()


    print("convert to ypath")
    c6, form6 = ij__ji_mode(c4, form4)
    n6 = ij__ji(n4)
    assert torch.equal(torch.as_tensor(n6), c6)
    assert c6.is_contiguous()

    c7, form7 = path__pos_pos_mode(c6, form6)
    n7 = path__pos_pos(n6)
    assert torch.equal(torch.as_tensor(n7), c7)
    assert c7.is_contiguous()

    c8, form8 = pos_pos__pos_offset_mode(c7, form7)
    n8 = pos_pos__pos_offset(n7)
    assert torch.equal(torch.as_tensor(n8), c8)
    assert c8.is_contiguous()

    c9, form9 = ij__ji_mode(c8, form8)
    n9 = ij__ji(n8)
    assert c9.is_contiguous()

    assert torch.equal(torch.as_tensor(n9), c9)

    assert torch.equal(c1, c9) and form1 == form9

