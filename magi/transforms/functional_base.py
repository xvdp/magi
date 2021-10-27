"""@xvdp

transform() main functional wraps other functionals handing input type, cloning
transform_profile() lightweight memory and call stack debugger

"""
from typing import Union, Callable
import logging
import torch
from koreto import memory_profiler, Col

from .. import config
from ..features import Item
_TensorItem = Union[torch.Tensor, Item]
###
# functional generic transform
#
def transform(data: _TensorItem, func: Callable, meta_keys: list=None,
              for_display: bool=False, **kwargs)-> _TensorItem:
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
def transform_profile(data: _TensorItem, func: Callable, meta_keys: list=None,
              for_display: bool=False, **kwargs)-> _TensorItem:
    """ nvml, torch.cuda.stats, torch.profiler digest on transform """
    return transform(data=data, func=func, meta_keys=meta_keys, for_display=for_display, **kwargs)
