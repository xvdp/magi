"""@xvdp

transform() main functional wraps other functionals handing input type, cloning
transform_profile() lightweight memory and call stack debugger

"""
from typing import Union, Callable, Optional
import logging
import numpy as np
import torch
from koreto import memory_profiler, Col

from .. import config
from ..features import Item
from ..utils import get_broadcastable
from .transforms_rnd import Distribution, Probs

tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor, Distribution)
Tensorish = Union[int, float, list, tuple, np.ndarray, torch.Tensor, Distribution]
TensorItem = Union[torch.Tensor, Item]

# pylint: disable=no-member

###
#
# functional generic transform handling cloning and memory profiling
#
def transform(data: TensorItem,
              func: Callable,
              kind_keys: Optional[list] = None,
              for_display: bool = False, **kwargs) -> TensorItem:
    """ Apply generic functional transform
    Args
        data        Item or tensor
        func        function applied to tensor
            2 args required: data (tensor), mode (str), units

    Args called if data is Item
        for_display bool [False], if True, CLONE
        kind_keys   list of kind keys to be transformed

    Args to transform function
        **kwargs    function arguments
    """
    mode = None if 'mode' not in kwargs else kwargs.pop('mode')

    # Apply transform to tensor
    if isinstance(data, torch.Tensor):
        return func(data, mode=mode, **kwargs)

    # Clone if requested
    if for_display:
        if any(t.requires_grad for t in data if isinstance(t, torch.Tensor)):
            logging.warning(f"{Col.YB}Attemtpt to clone with grad on {func.__name__} stopped, config.FOR_DISPLAY -> False {Col.AU}")
            config.set_for_display(False)
        else:
            data = data.deepclone()

    # Apply transform to Item indices
    indices = data.get_indices(kind=kind_keys)
    assert indices, f"Cannot Transform, missing Item.kind keys in {kind_keys}"
    for i in indices:
        data[i] = func(data[i], mode=data.form[i], **kwargs)
    return data

@memory_profiler
def transform_profile(data: TensorItem,
                      func: Callable,
                      kind_keys: Optional[list] = None,
                      for_display: bool = False, **kwargs) -> TensorItem:
    """ nvml, torch.cuda.stats, torch.profiler digest on transform """

    kw = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kw[key] = f"tensor({value.tolist()})" if value.numel() < 12 else f"{value.shape})"
        elif isinstance(value, Item):
            kw[key] = f"Item({value[0].shape}...{len(value)})"
        else:
            kw[key] = value
    print(f"\n@memory_profiler\n{func.__name__}({kw})".replace(" ", ""))

    return transform(data=data, func=func, kind_keys=kind_keys, for_display=for_display, **kwargs)

###
#
# distrubution samplers
#
def get_bernoulli_like(x: Tensorish, like: TensorItem) -> torch.Tensor:

    _reinit = isinstance(x, float) and x > 0 and x < 1
    _reinit |= torch.is_tensor(x) and not torch.all(x[x!=1] == 0)
    _reinit |= isinstance(x, (list, tuple)) and not all(i in (0, 1) for i in x)
    if _reinit:
        x = Probs(p=x, expand_dims=0)
    return get_sample_like(x, like=like)

def get_sample_like(x: Tensorish, like: TensorItem) -> torch.Tensor:
    """ sample from Value, Probs
    if x is tensor of correct dtype and device, acts as a pass thru
    Args
        x,      torch.Tensor, Values, Probs
        like    torch.Tensor, item
    """
    if isinstance(like, list): # list or Item, first element has a tensor in it
        like = like[0]
    assert torch.is_tensor(like), f"{Col.YB}expected tensor, got {type(like)}{Col.AU}"

    if isinstance(x, (int, float, list, tuple)):
        x = get_broadcastable(x, other=like)

    if isinstance(x, torch.Tensor):
        _to = get_dtype_device_update(x, like)
        if _to:
            x = x.to(**_to)
        size_diff = like.ndim - x.ndim
        if size_diff > 0 and x.ndim > 1:
            x = x.view(*x.shape, *[1]*size_diff)
        elif size_diff < 0 and x.ndim > 1:
            x = x.view(x.shape[:size_diff])
        return x

    if isinstance(x, Distribution):
        _to = get_dtype_device_update(x, like)
        if _to:
            x.to(**_to)
        return x.sample(like.shape)

    raise NotImplementedError(f"Expected, tensor or distribution sampler, got {type(x)}")

def get_dtype_device_update(x: Union[torch.Tensor, Distribution], like: torch.Tensor) -> dict:
    out = {}
    if like.dtype != x.dtype:
        out["dtype"] = like.dtype
    if like.device != x.device:
        out["device"] = like.device
    return out

def p_all(p: Union[int, float, torch.Tensor]) -> bool:
    """ True if p is 1, 1.0 or torch.ones()
        not strictly equal, float(1) & int(1) but python returns True
        .item() fetches from cuda
    """
    if isinstance(p, (int, float)):
        return p == 1
    return p.sum().item() == p.numel()

def p_none(p: Union[int, float, torch.Tensor]) -> bool:
    """ True if p is 0, 0.0 or torch.zeros()"""
    if isinstance(p, (int, float)):
        return not p
    return not p.sum().item()

def p_some(p: Union[int, float, torch.Tensor]) -> bool:
    """ True if p is tensor with some but not all values == 1."""
    if torch.is_tensor(p) and p.numel() > p.sum().item() > 0:
        return True
    return False
