"""@xvdp

transform() main functional wraps other functionals handing input type, cloning
transform_profile() lightweight memory and call stack debugger

"""
from typing import Union, Callable, Any
import logging
import numpy as np
import torch
from koreto import memory_profiler, Col

from .. import config
from ..features import Item
from ..utils import get_broadcastable

from .transforms_base import Randomize
_tensoritem = (torch.Tensor, Item)
_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)
_vector = (np.ndarray, torch.Tensor)

#pylint: disable=no-member

###
# functional generic transform
#
def transform(data: Union[_tensoritem], func: Callable, meta_keys: list=None,
              for_display: bool=False, **kwargs)-> Union[_tensoritem]:
    """ Apply generic functional transform
    Args
        data        Item or tensor
        func        function applied to tensor

    Args called if data is Item
        for_display bool [False], if True, CLONE
        meta_keys   list of meta keys to be transformed

    Args to transform function
        **kwargs    function arguments
    """
    # Apply transform to tensor
    if isinstance(data, torch.Tensor):
        return func(data, **kwargs)

    # Clone if requested
    if for_display:
        if any(t.requires_grad for t in data if isinstance(t, torch.Tensor)):
            logging.warning(f"{Col.YB}Attemtpt to clone with grad on {func.__name__} stopped, config.FOR_DISPLAY -> False {Col.AU}")
            config.set_for_display(False)
        else:
            data = data.deepclone()

    # Apply transform to Item indices
    indices = data.get_indices(meta=meta_keys)
    assert indices, f"Cannot Transform, missing Item.meta keys in {meta_keys}"
    for i in indices:
        data[i] = func(data[i], **kwargs)
    return data

@memory_profiler
def transform_profile(data: Union[_tensoritem], func: Callable, meta_keys: list=None,
              for_display: bool=False, **kwargs)-> Union[_tensoritem]:
    """ nvml, torch.cuda.stats, torch.profiler digest on transform """
    return transform(data=data, func=func, meta_keys=meta_keys, for_display=for_display, **kwargs)


def get_value(x: Any, tensor:torch.Tensor, axis: int=1,
              batch_size: int=1) -> torch.Tensor:
    """ Convert 'x' to tensor with same dtype, device, ndim as tensor
    with x.shape[i] == tensor.shape[i] or axis i reduced to 1 by mean

    Args
        x       (list, tuple, int, float, ndarray torch.Tensor, Randomize)
            if ndim == 1 and len() > 1 and len() == len(tensor.shape[axis]

        if x is Randomize class, sample new value

        tensor  torch.Tensor) tensor to match
            if tensor or ndarray, axis size equal x or reduced to mean

        axis    (int [min(1 | tensor.ndim-1)]) only necessary for broadcasing where x.ndim <= axis
    """
    if torch.is_tensor(x) and x.ndim == tensor.ndim and x.dtype == tensor.dtype and x.device == tensor.device:
        return x

    if isinstance(x, Randomize):
        x, p = x.sample(batch_size)
    # TODO apply p
    # TODO what if rebuilt sampler

    return get_broadcastable(x, tensor=tensor, axis=axis)
