"""@xvdp
funcional appearance
    functions that only modify content values of data
    without changing data size or relative location of values
"""
from typing import Union
import numpy as np
import torch
from koreto import memory_profiler
from torch.distributed.distributed_c10d import broadcast
from .. import config
from ..utils import reduce_to
from ..features import Item
_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)
_vector = (np.ndarray, torch.Tensor)

# pylint: disable=no-member
# def normalize(data: Union[torch.Tensor, list], mean: Union[_torchable], std: Union[_torchable],
#                inplace: int=0, axis: int=1) -> Union[torch.Tensor, Item]:
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         data    (Item or Tensor)

#         mean    (_torchable): per channel or image mean
#         std     (_torchable): per channel or image std

#         inplace (bool[False]) default from Normalize [None] defers to config.INPLACE [True]
#                 if used with grad needs be False
#         axis    (int [1]) axis over which mean & std are broadcast if (mean|std) are 1d

#     Returns:
#         list [norm tensor, data[1], data[2]]
#     """
#     if isinstance(data, torch.Tensor):
#         data = _normalize(data, mean, std, inplace, axis)

#     elif isinstance(data, Item): # requires Item to be labeled with meta= data_<>d keys
#         for i in data.get_indices(meta=["data_1d", "data_2d", "data_3d"]):
#             data[i] = _normalize(data[i], mean, std, inplace, axis)

#     elif isinstance(data, list):
#         for i in [i for i in range(len(data)) if data[i].is_floating_point() and data[i].ndim >=3]:
#             data[i] = _normalize(data[i], mean, std, inplace, axis)

#     return data

# def _normalize(x: torch.Tensor, mean: Union[_torchable], std: Union[_torchable],
#                inplace: bool=False, axis: int=1) -> torch.Tensor:
#     """ Robust normalization, mean and std can be any dim and shape
#     Args
#         x   tensor to normalize
#     """
#     mean = reduce_to(mean, x, axis=axis)
#     std = reduce_to(std, x, axis=axis)
#     if inplace or (inplace is None and config.INPLACE):
#         return x.sub_(mean).div_(std)

def normalize(data: Union[_torchable], mean: Union[_torchable], std: Union[_torchable], for_display: bool=False,
              profile: bool=False) -> Union[_torchable]:
    """  normalize item or tensor
    """
    if not profile and isinstance(data, Item):
        return normalize_item(data, mean, std, for_display)
    if not profile and isinstance(data, torch.Tensor):
        return normalize_tensor(data, mean, std)

    if profile and isinstance(data, Item):
        return _normalize_item_profile(data, mean, std, for_display)
    if profile and isinstance(data, torch.Tensor):
        return _normalize_tensor_profile(data, mean, std)

def normalize_item(data: Item, mean: Union[_torchable], std: Union[_torchable], for_display: bool=False) -> Item:
    """
    Args
        data        Item
        mean, std.  tensors, appropriately broadcasted, same dtype and device
        for_display bool
    """
    _normalizable = ['data_1d', 'data_2d', 'data_3d']
    indices = data.get_indices(meta=_normalizable)
    assert indices, f"normalzion of Item requires Item.meta in {_normalizable}"

    if not for_display:
        for i in indices:
            data[i] = normalize_tensor(data[i], mean, std)
        return data

    out = data.deepclone()
    for i in indices:
        out[i] = normalize_tensor(out[i], mean, std)
    return out

def _match_tensor(x: Union[_torchable], tensor: torch.Tensor) -> torch.Tensor:
    """ match ndim, device, dtype of tensor"""
    if not torch.is_tensor(x) or x.ndim != tensor.ndim or x.dtype != tensor.dtype or x.device != tensor.device:
        x = reduce_to(x, tensor)
    return x

def normalize_tensor(x: torch.Tensor, mean: Union[_torchable], std: Union[_torchable]) -> torch.Tensor:
    """
    Args
        x           data tensor
        mean, std.  appropriately broadcasted tensors, same dtype and device
    """
    mean = _match_tensor(mean, x)
    std = _match_tensor(std, x)
    if x.requires_grad:
        return x.sub(mean).div(std)
    return x.sub_(mean).div_(std)

@memory_profiler
def _normalize_item_profile(data: Item, mean: Union[_torchable], std: Union[_torchable], for_display: bool=False) -> Item:
    return normalize_item(data, mean, std, for_display)
@memory_profiler
def _normalize_tensor_profile(data: torch.Tensor, mean: Union[_torchable], std: Union[_torchable]) -> torch.Tensor:
    return normalize_tensor(data, mean, std)

# other overly compact ways of normalizing
# def norm(x, mean, std, inplace):
#     return getattr(getattr(x, f'sub{"_"*inplace}')(mean), f'div{"_"*inplace}')(std)
# norml = lambda x, mean=0.5, std= 0.23, p="": getattr(getattr(x, f'sub{p}')(mean), f'div{p}')(std)
