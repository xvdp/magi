"""@xvdp

Appearance functionals
    only modify content values of data
    without changing data size or relative location of values
Functionals are tagged for subtypes,
eg, normalize(), normalize_tensor(), normalize_proc()

Functionals
    normalize()
    unnormalize()
    normtorange()
    saturate()

"""
from typing import Union, Optional
import numpy as np
import torch

from ..features import Item
from ..utils import to_saturation, get_broadcastable, broadcast_tensors, tensor_apply_vals
from .functional_base import transform, transform_profile, p_all, p_none, get_sample_like, get_bernoulli_like

_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor, type)
_Tensorish = Union[_tensorish]
_TensorItem = Union[torch.Tensor, Item]
# pylint: disable=no-member

###
#
# functional for Normalize or MeanCenter
#
def normalize(data: _TensorItem,
              mean: _Tensorish,
              std: _Tensorish,
              p: _Tensorish = 1,
              for_display: bool = False,
              profile: bool = False) -> _TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        tensor | Item
        mean        tensor, iterable convertible to tensor
        std         tensor, iterable convertible to tensor
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    mean = get_sample_like(mean, like=data)
    std = get_sample_like(std, like=data)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=normalize_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std,
                      p=p)

def normalize_tensor(x: torch.Tensor,
                     mean: _Tensorish,
                     std: _Tensorish,
                     p: _Tensorish=1) -> torch.Tensor:
    """
    Args
        x           tensor
        mean, std.  appropriately broadcasted tensors, same dtype and device
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    mean = get_sample_like(mean, like=x)
    std = get_sample_like(std, like=x)

    if p_all(p):
        return normalize_proc(x, mean, std)

    return torch.lerp(x, normalize_proc(x, mean, std, inplace=True), p)


def normalize_proc(x: torch.Tensor,
                   mean: torch.Tensor,
                   std: torch.Tensor,
                   inplace: Optional[bool] = None) -> torch.Tensor:
    if x.requires_grad or not inplace:
        return x.sub(mean).div(std)
    return x.sub_(mean).div_(std)


###
#
# functional for UnNormalize or UnMeanCenter
#
def unnormalize(data: _TensorItem,
                mean: _Tensorish,
                std: _Tensorish,
                p: Union[type, _Tensorish] = 1,
                for_display: bool = False,
                profile: bool = False)-> _TensorItem:
    """  unnormalize item or tensor | profile memory optional """
    p = get_bernoulli_like(p, like=data)
    mean = get_sample_like(mean, like=data)
    std = get_sample_like(std, like=data)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=unnormalize_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std,
                      p=p)

def unnormalize_tensor(x: torch.Tensor,
                       mean: _Tensorish,
                       std: _Tensorish,
                       clip: bool = False,
                       p: _Tensorish = 1) -> torch.Tensor:
    """
    Args
        x           tensor
        mean, std.  appropriately broadcasted tensors, same dtype and device
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    mean = get_sample_like(mean, like=x)
    std = get_sample_like(std, like=x)

    if p_all(p):
        return unnormalize_tensor_proc(x, mean, std, clip)

    return torch.lerp(x, unnormalize_tensor_proc(x, mean, std, clip, inplace=True), p)

def unnormalize_tensor_proc(x: torch.Tensor,
                            mean: _Tensorish,
                            std: _Tensorish,
                            clip: bool = True,
                            inplace: Optional[bool] = None) -> torch.Tensor:

    if x.requires_grad or not inplace:
        x = x.mul(std).add(mean)
        if clip:
            x = x.clamp(0., 1.)
    else:
        x.mul_(std).add_(mean)
        if clip:
            x.clamp_(0., 1.)
    return x
###
#
# functional for NormToRange
#
def normtorange(data: _TensorItem,
                minimum: Union[type, _Tensorish] = 0,
                maximum: Union[type, _Tensorish] = 1,
                p: Union[type, _Tensorish] = 1,
                excess_only: bool = False,
                for_display: bool = False,
                profile: bool = False) -> _TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        minimum     int | float [0.]
        maximum     int | float [1.]
        excess_only bool [False] - if minimum < data.min() and maximum > data.max(), dont modify
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    minimum = get_sample_like(minimum, like=data)
    maximum = get_sample_like(maximum, like=data)

    # print(f'{inspect.stack()[0].function}()')
    # print('  minimum', minimum.shape)
    # print('  maximum', maximum.shape)
    # print('  p', p.shape)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=normtorange_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], minimum=minimum, maximum=maximum,
                      p=p, excess_only=excess_only)

def normtorange_tensor(x: torch.Tensor,
                       minimum: _Tensorish,
                       maximum: _Tensorish,
                       p: _Tensorish = 1,
                       excess_only: bool = False) -> torch.Tensor:
    """
    Args
        x           (tensor)
        minimum     (int, float,tuple,list,tensor)
        maximum     (int, float,tuple,list,tensor)
        p           tensor, list, float
        excess_only bool    [False] # True: do not expand to min, max, only contract
        axis        int, independent axis: 0: sample. 1: channels
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    minimum = get_sample_like(minimum, like=x)
    maximum = get_sample_like(maximum, like=x)

    if p_all(p):
        return normtorange_proc(x, minimum, maximum, excess_only=excess_only)

    return torch.lerp(x, normtorange_proc(x, minimum, maximum, excess_only=excess_only,
                                          inplace=False), p)


def normtorange_proc(x: torch.Tensor,
                     minimum: torch.Tensor,
                     maximum: torch.Tensor,
                     excess_only: bool = False,
                     inplace: Optional[bool] = None) -> torch.Tensor:
    """
    Args
        x           (tensor)
        minimum     (int, float,tuple,list,tensor)
        maximum     (int, float,tuple,list,tensor)
        excess_only (bool [False]) # True: do not expand to min, max, only contract
        inplace     (bool [None]) # True for lerping partial probs
    """
    # minimum and maximum can be per sample or for entire batch
    minimum = get_broadcastable(minimum, other=x, axis=1)
    maximum = get_broadcastable(maximum, other=x, axis=1)
    if minimum.shape != maximum.shape:
        minimum, maximum = broadcast_tensors(minimum, maximum)

    hold_axes = [i for i in range(len(minimum.shape)) if minimum.shape[i] > 1]
    x_minimum = tensor_apply_vals(x, "min", hold_axes, keepdims=True)
    x_maximum = tensor_apply_vals(x, "max", hold_axes, keepdims=True)

    if excess_only:
        minimum = torch.maximum(minimum, x_minimum)
        maximum = torch.maximum(maximum, x_maximum)

    # minimize reductions
    x_maximum = x_maximum.sub(x_minimum)
    maximum = maximum.sub(minimum)

    if x.requires_grad or not inplace:
        return x.sub(x_minimum).div(x_maximum).mul(maximum).add(minimum)

    return x.sub_(x_minimum).div_(x_maximum).mul_(maximum).add_(minimum)

###
#
# functional for Saturate
#
def saturate(data: _TensorItem,
             sat: Union[int, float, torch.Tensor, type],
             p: Union[int, torch.Tensor, type] = 1,
             for_display: bool = False,
             profile: bool = False)->  _TensorItem:

    p = get_bernoulli_like(p, like=data[0])
    sat = get_sample_like(sat, like=data[0])

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=saturate_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'],
                      sat=sat, p=p)

def saturate_tensor(x: torch.Tensor, sat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """ if p saturate x to sat*saturation of channels of x
    0: grayscale, 2, oversaturate, -1 flip saturation
    if len p > 1 independently saturate
    Args
        x   tensor shape NC...
        sat tensor shape 1 | N | NC
        p   tensor shape 1 | N
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    sat = get_sample_like(sat, like=x)

    if p_all(p):
        return to_saturation(x, sat)

    return torch.lerp(x, to_saturation(x, sat), p)

    # # TODO profile if lerp, loop or multiprocess
    # for i, _p in enumerate(p):
    #     if p_all(_p):
    #         x[i:i+1] = to_saturation(x[i:i+1], sat[i:i+1])
    # return x

###
#
# functional for Gamma
#
def gamma(data: _TensorItem,
                value: Union[type, _Tensorish] = 1,
                p: Union[type, _Tensorish] = 1,
                from_gamma: _Tensorish = 2.2,
                for_display: bool = False,
                profile: bool = False) -> _TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        minimum     int | float [0.]
        maximum     int | float [1.]
        excess_only bool [False] - if minimum < data.min() and maximum > data.max(), dont modify
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    values = get_sample_like(value, like=data)
    from_gamma = get_broadcastable(from_gamma, other=data[0], axis=1)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=gamma_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], values=values,
                      from_gamma=from_gamma, p=p)


def gamma_tensor(x: torch.Tensor, 
                 values: torch.Tensor, p: torch.Tensor,
                 from_gamma: _Tensorish = 2.2) -> torch.Tensor:
    """ if p saturate x to sat*saturation of channels of x
    0: grayscale, 2, oversaturate, -1 flip saturation
    if len p > 1 independently saturate
    Args
        x   tensor shape NC...
        sat tensor shape 1 | N | NC
        p   tensor shape 1 | N
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    values = get_sample_like(values, like=x)
    from_gamma = get_broadcastable(from_gamma, other=x, axis=1)

    print(f"gamma_tensor(p={p}, from_gamma={from_gamma}")
            
    if p_all(p):
        return gamma_proc(x, values, from_gamma=from_gamma)

    return torch.lerp(x, gamma_proc(x, values, from_gamma=2.2, inplace=False), p)

def gamma_proc(x, values=1.0, from_gamma=2.2, inplace: Optional[bool] = None):

    if x.requires_grad or not inplace:
        return x.pow(from_gamma/values)
    return x.pow_(from_gamma/values)
