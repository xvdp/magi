"""@xvdp
modify WIDER_ROOT to ensure runs on wider
"""
import os.path as osp
import torch
import pytest
from magi import config
from magi.datasets import ImageNet
from magi.features import Item

# pylint: disable=no-member
IMAGENET_ROOT = config.load_dataset_path("ImageNet")


@pytest.mark.skipif(IMAGENET_ROOT is None, reason=f"Could not find Wider Folder, adjust IMAGENET_ROOT in {__file__} to test")
def test_default():

    I = ImageNet()
    assert I.keep_names == ['image', 'target_index']
    assert len(I.classes) == 1000, f"expected 1000 classes in '{I.data_root}' '{I.mode}'"


    assert isinstance(I.samples[0], Item), f"expected Item, found {type(I.samples[0])}"
    filename = I.samples[0].get(names="filename")[0]
    assert osp.isfile(filename), f"file {filename} not found"

    item = I[0]
    assert item.names == I.keep_names, f"expected elements {I.keep_names}, found {item.names}"
    img = item.get(names="image")[0]
    assert isinstance(img, torch.Tensor), f"expected tensor, got {type(img)}"


@pytest.mark.skipif(IMAGENET_ROOT is None, reason=f"Could not find Wider Folder, adjust IMAGENET_ROOT in {__file__} to test")
def test_subset_ordered():

    subset = 5
    I = ImageNet(mode="val", names=["image", "filename", "image_index", "target_folder", "target_name", "target_index"], ordered=2, subset=subset)

    assert len(I.classes) == subset

    assert I.samples[0][-1] == 0
    assert I.samples[2][-1] == 1
    assert I.samples[4][-1] == 2


@pytest.mark.skipif(IMAGENET_ROOT is None, reason=f"Could not find Wider Folder, adjust IMAGENET_ROOT in {__file__} to test")
def test_test_mode():

    # I = ImageNet(mode="test")
    # assert I.__getitem__().names == ['image']

    I = ImageNet(mode="test", names=["image", "filename", "image_index", "target_folder", "target_name", "target_index"])
    assert I.__getitem__().names == ['image', 'filename', 'image_index'], f"test dataset should yield no class information"


@pytest.mark.skipif(IMAGENET_ROOT is None, reason=f"Could not find Wider Folder, adjust IMAGENET_ROOT in {__file__} to test")
def test_grad_override_for_display():

    # I = ImageNet(mode="test")
    # assert I.__getitem__().names == ['image']
    config.set_for_display(True)

    I = ImageNet(mode="val", grad=True)

    d = I.__getitem__()
    assert d[0].requires_grad

    assert not config.FOR_DISPLAY
