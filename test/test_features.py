"""@xvdp
tests to feature classes for dataloaders
"""
import pytest
import torch
from magi.features import Item, ListDict

# def test_del_Item():
#     """ should del Item not delete items it contains as well?"""
#     tensor = torch.randn(1,3,54,54)
#     dic = {"a":0, "b":1, "c":2}
#     string = "inigomontoya"
#     ls = [12,12,12]
#     data = Item([tensor, ls, dic, 12., 0, string],
#                     tags=["image", "list", "dict", "int", "float", "str"])
#     del data
#     assert tensor is None, f"did deleting data not delete tensor? {tensor.shape}"


# pylint: disable=no-member
def test_listdict():

    _ld = ListDict([torch.randn([1,3,45,45]), 1,2,3,4 ], some_key="some_value", some_other_key=42)
    assert len(_ld) == 5
    assert len(_ld.__dict__) == 2
    assert _ld.some_key == "some_value"

    _ld.new_key = "zztop"
    assert len(_ld.__dict__) == 3

    _ld.clear()
    assert len(_ld) == len(_ld.__dict__) == 0, f"expected data cleared"

def test_empty_Item():
    data = Item()

def test_filled_Item():
    data = Item([torch.randn([1,3,45,45]), torch.randint(2000, (3,4,2)), 2, 1], tags=["image", "boxes", "image_id", "class_id"])

    assert data.keys == ["tags"]
    assert data.tags == ["image", "boxes", "image_id", "class_id"]
    assert len(data) == len(data.tags)

def test_Item_pop():
    data = Item([torch.randn([1,3,45,45]), torch.randint(2000, (3,4,2)), 2, 1], tags=["image", "boxes", "image_id", "class_id"])

    data.pop(0)
    assert data.tags == ["boxes", "image_id", "class_id"]
    assert len(data) == len(data.tags)

def test_Item():
    data = Item([torch.randn([1,3,45,45]), torch.randint(2000, (3,4,2)), 2, 1], tags=["image", "boxes", "image_id", "class_id"])
    data.pop(0)

    # append to self and tags list
    data.append("johnson", tags="name")
    assert len(data) == len(data.tags)

    # add new key
    data.info = ["some", "form", "of", "information"]
    assert data.keys == ["tags", "info"]

    # append to list with same tag and key
    data.append(20123.1, tags="name", info="also needs to be extended")
    assert len(data) == len(data.tags) == len(data.info)

    # return subset of elements with key == val, eg. data.tag == "name"
    subset = data.get(tags="name")
    assert subset == ["johnson", 20123.1]

    subset = data.get(tags=["name", "class_id"])
    assert subset == [1, "johnson", 20123.1]
    # extend
    data.extend([1, 3.14], tags=["one", "pi"], info=["int", "float"])

    # remove
    data.remove("johnson")
    assert len(data) == len(data.tags) == len(data.info)


def test_Item_get_intersect():
    item = Item([12,53.5,0.01], names=['eero', 'eero', 'saarinen'], dtype=['int', 'float', 'float'])

    sub = item.get(names='eero', dtype='float')
    assert sub == [53.5]

    sub = item.get_indices(names='eero', dtype='int')
    assert sub == [0]

def _get_data_item():
    tensor = torch.randn(1,3,45,45)
    dic = {"a":0, "b":1, "c":2}
    string = "inigomontoya"
    ls = [12,12,12]
    return Item([tensor, ls, dic, 12., 0, string],
                    tags=["image", "list", "dict", "int", "float", "str"])

def test_keep_by_name():
    tensor = torch.randn(1,3,45,45)
    dic = {"a":0, "b":1, "c":2}
    data = Item([tensor, [12,12,12], dic, 12., 0, "inigomontoya"],
                    tags=["image", "list", "dict", "int", "float", "str"])

    data.keep(tags=["image", "dict"])
    assert len(data) == len(data.tags) == 2
    assert torch.all(torch.eq(data[0], tensor))
    assert data[1] == dic

def test_keep_by_name2():
    tensor = torch.randn(1,3,45,45)
    dic = {"a":0, "b":1, "c":2}
    ls = [12,12,12]
    string = "inigomontoya"

    data = Item([tensor, ls, dic, 12., 0, string],
                     tags=["image", "list", "dict", "int", "float", "str"])

    data.keep(tags=["list", "str"])
    assert len(data) == len(data.tags) == 2
    assert data[0] == ls
    assert data[1] == string

def test_keep_by_index():
    tensor = torch.randn(1,3,54,54)
    dic = {"a":0, "b":1, "c":2}
    string = "inigomontoya"
    ls = [12,12,12]
    data = Item([tensor, ls, dic, 12., 0, string],
                    tags=["image", "list", "dict", "int", "float", "str"])

    data.keep([-1, 1])
    assert len(data) == len(data.tags) == 2
    assert data[0] == ls
    assert data[1] == string

def test_cast_partial_list():
    tensor = torch.randn(1,3,54,54)
    dic = {"a":0, "b":1, "c":2}
    string = "inigomontoya"
    ls = [12,12,12]
    data = Item([tensor, ls, dic, 12., 0, string],
                    tags=["image", "list", "dict", "int", "float", "str"])

    out = list(data)[1:3]
    del data

    assert len(out) == 2


def test_keep_none():
    data = _get_data_item()

    data.keep(None)
    assert len(data) == len(data.tags) == 6
    data.keep()
    assert len(data) == len(data.tags) == 6

def test_to_tensor_all():
    data = Item([[[0,1],[0,2]], [0.1,.1,.4], [[[1]],[[.1]]], 12., 0, "asdf"], tags=["image", "list", "dict", "int", "float", "str"])
    data.to_torch()
    for i in [0,1,2,3,4]:
        assert isinstance(data[i], torch.Tensor), f"data[i]: {data[i]}"
    assert isinstance(data[5], str)

    # re assign cuda and torch
    data.to_torch(device="cuda", dtype="float16")
    for i in [0,1,2,3,4]:
        assert data[i].device.type == "cuda" and data[i].dtype == torch.float16


def test_to_tensor_dtype():

    dtype = dtype=["float32", "uint8", "float64", "float16", "str", None]
    data = Item([[[0,1],[0,2]], [1,1,4], [[[1]],[[.1]]], 12., 0, "asdf"], tags=["image", "list", "dict", "int", "float", "str"],dtype=dtype)
    data.to_torch()

    for i in [0,1,2,3]:
        assert data[i].dtype == torch.__dict__[dtype[i]], f"found, { data[i].dtype }"


    data.to(device='cuda')
    for i in [0,1,2,3]:
        assert data[i].device.type == "cuda",  f"found {data[i].device}, {data[i].dtype}"

    data.to(dtype=torch.float32, device="cpu")
    for i in [0,1,2,3]:
        assert data[i].device.type == "cpu" and data[i].dtype == torch.float32, f"found {data[i].device}, {data[i].dtype}"


    assert isinstance(data[4], int), "should not have been converted it was tagged as dtype 'str' which is not in torch.__dict__'"
    assert isinstance(data[5], str)


def test_to_tensor_cuda_exclude_deepclone():
    data = Item([[[0,1],[0,2]], [0.1,.1,.4], [[[1]],[[.1]]], 12., 0, "asdf"],tags=["image", "list", "dict", "int", "float", "str"])

    data.to_torch(device="cuda", exclude=[1,3,-2])
    for i in [0,2]:
        assert isinstance(data[i], torch.Tensor)
    for i in [1,3,4]:
        assert not isinstance(data[i], torch.Tensor)

    clone = data.deepclone()

    clone.remove(data[0])
    assert len(clone.tags) == len(clone) == 5
    assert len(data.tags) == len(data) == 6

    del data[0]
    assert len(data.tags) == len(data) == 5


def test_to_torch_doesnt_break_grad():
    z = torch.randn([1,3,34,34])
    z.requires_grad = True

    data = Item([z, 1, [1,2,3], [[34,34],[12,2]] ], names=["tensor", "one", "list2d", "list2d"])
    data.to_torch()
    assert data[0].requires_grad, f"expected grad"
    for i in range(1, len(data[1:])):
        assert not data[i].requires_grad, f"expected no grad on item {i} {data[i].shape} "


def test_Item_reverse():
    data = Item([1,2,3,4], mykey=['want', 'too', 'tree', 'for'], _privatekey=[34,22])
    data.reverse()
    assert data.mykey == ['for', 'tree', 'too', 'want'] and data == [4,3,2,1]
    assert data._privatekey == [34,22], "private keys should not be reversed"

@pytest.mark.xfail
def test_Item_failtosort():
    data = Item(range(5), tags=['r', 'wq', 'e', 'q', 'r'], _taggy = [3,2])
    data.sort() # pylint: disable=not-callable

@pytest.mark.xfail
def test_Item_failtoaddprivates():
    data = Item(range(5), tags=['r', 'wq', 'e', 'q', 'r'], _taggy = [3,2])
    data.keys = [5,5,5,5,5]
