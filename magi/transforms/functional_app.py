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
    gamma()

TODO could be further simplified:
    - normalize and unnormalize are nearly indentical
    - func, func_tensor, func_proc
"""
from typing import Union, Optional
import torch

from ..utils import to_saturation, get_broadcastable, broadcast_tensors, tensor_apply_vals
from .functional_base import transform, transform_profile, Tensorish, TensorItem
from .functional_base import p_none, p_all, get_sample_like, get_bernoulli_like

# pylint: disable=no-member

###
#
# functional for Normalize or MeanCenter
#
def normalize(data: TensorItem,
              mean: Tensorish,
              std: Tensorish,
              p: Tensorish = 1,
              for_display: bool = False,
              profile: bool = False) -> TensorItem:
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
                      kind_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std,
                      p=p)

def normalize_tensor(x: torch.Tensor,
                     mean: Tensorish,
                     std: Tensorish,
                     p: Tensorish=1,
                     mode: str = 'NCHW') -> torch.Tensor:
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

    return torch.lerp(x, normalize_proc(x, mean, std, inplace=False), p)

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
def unnormalize(data: TensorItem,
                mean: Tensorish,
                std: Tensorish,
                p: Union[type, Tensorish] = 1,
                for_display: bool = False,
                profile: bool = False)-> TensorItem:
    """  unnormalize item or tensor | profile memory optional """
    p = get_bernoulli_like(p, like=data)
    mean = get_sample_like(mean, like=data)
    std = get_sample_like(std, like=data)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=unnormalize_tensor, for_display=for_display,
                      kind_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std,
                      p=p)

def unnormalize_tensor(x: torch.Tensor,
                       mean: Tensorish,
                       std: Tensorish,
                       clip: bool = False,
                       p: Tensorish = 1,
                       mode: str = 'NCHW') -> torch.Tensor:
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

    return torch.lerp(x, unnormalize_tensor_proc(x, mean, std, clip, inplace=False), p)

def unnormalize_tensor_proc(x: torch.Tensor,
                            mean: Tensorish,
                            std: Tensorish,
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
def normtorange(data: TensorItem,
                minimum: Union[type, Tensorish] = 0,
                maximum: Union[type, Tensorish] = 1,
                p: Union[type, Tensorish] = 1,
                for_display: bool = False,
                profile: bool = False) -> TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        minimum     int | float [0.]
        maximum     int | float [1.]
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    minimum = get_sample_like(minimum, like=data)
    maximum = get_sample_like(maximum, like=data)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=normtorange_tensor, for_display=for_display,
                      kind_keys=['data_1d', 'data_2d', 'data_3d'], minimum=minimum,
                      maximum=maximum, p=p)

def normtorange_tensor(x: torch.Tensor,
                       minimum: Tensorish,
                       maximum: Tensorish,
                       p: Tensorish = 1,
                       mode: str = 'NCHW') -> torch.Tensor:
    """
    Args
        x           (tensor)
        minimum     (int, float,tuple,list,tensor)
        maximum     (int, float,tuple,list,tensor)
        p           tensor, list, float
        axis        int, independent axis: 0: sample. 1: channels
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    minimum = get_sample_like(minimum, like=x)
    maximum = get_sample_like(maximum, like=x)

    if p_all(p):
        return normtorange_proc(x, minimum, maximum)

    return torch.lerp(x, normtorange_proc(x, minimum, maximum, inplace=False), p)

def normtorange_proc(x: torch.Tensor,
                     minimum: torch.Tensor,
                     maximum: torch.Tensor,
                     inplace: Optional[bool] = None) -> torch.Tensor:
    """
    Args
        x           (tensor)
        minimum     (int, float,tuple,list,tensor)
        maximum     (int, float,tuple,list,tensor)
        softclamp   (float [None]) 1 -> tanh, inf -> step function
        inplace     (bool [None]) # True for lerping partial probs
    """
    # minimum and maximum can be per sample or for entire batch
    minimum = get_broadcastable(minimum, other=x, axis=1)
    maximum = get_broadcastable(maximum, other=x, axis=1)
    if minimum.shape != maximum.shape:
        minimum, maximum = broadcast_tensors(minimum, maximum)
    hold_axis = [i for i in range(len(minimum.shape)) if minimum.shape[i] > 1]

    x_minimum = tensor_apply_vals(x, "min", hold_axis, keepdims=True)
    x_maximum = tensor_apply_vals(x, "max", hold_axis, keepdims=True)

    x_delta = x_maximum.sub(x_minimum)
    delta = maximum.sub(minimum)

    if x.requires_grad or not inplace:
        return x.sub(x_minimum).div(x_delta).mul(delta).add(minimum)

    return x.sub_(x_minimum).div_(x_delta).mul_(delta).add_(minimum)

###
#
# functional for Saturate
#
def saturate(data: TensorItem,
             sat: Union[int, float, torch.Tensor, type],
             p: Union[int, torch.Tensor, type] = 1,
             for_display: bool = False,
             profile: bool = False)->  TensorItem:

    p = get_bernoulli_like(p, like=data[0])
    sat = get_sample_like(sat, like=data[0])

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=saturate_tensor, for_display=for_display,
                      kind_keys=['data_1d', 'data_2d', 'data_3d'],
                      sat=sat, p=p)

def saturate_tensor(x: torch.Tensor,
                    sat: torch.Tensor,
                    p: torch.Tensor,
                    mode: str = 'NCHW') -> torch.Tensor:
    """ if p saturate x to sat*saturation of channels of x
    0: grayscale, 2, oversaturate, -1 flip saturation
    if len p > 1 independently saturate
    Args
        x   tensor shape NC...
        sat tensor shape 1 | N | NC
        p   tensor shape 1 | N
    # TODO profile if lerp, loop or multiprocess
        for i, _p in enumerate(p):
            if p_all(_p):
                x[i:i+1] = to_saturation(x[i:i+1], sat[i:i+1])
        return x
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    sat = get_sample_like(sat, like=x)

    if p_all(p):
        return to_saturation(x, sat)

    return torch.lerp(x, to_saturation(x, sat), p)

###
#
# functional for Gamma
#
def gamma(data: TensorItem,
                value: Union[type, Tensorish] = 1,
                p: Union[type, Tensorish] = 1,
                from_gamma: Tensorish = 2.2,
                for_display: bool = False,
                profile: bool = False) -> TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        value       Value, tensor, float    target gamma
        p           Probs, tensor, float    probability
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    values = get_sample_like(value, like=data)
    from_gamma = get_broadcastable(from_gamma, other=data[0], axis=1)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=gamma_tensor, for_display=for_display,
                      kind_keys=['data_1d', 'data_2d', 'data_3d'], values=values,
                      from_gamma=from_gamma, p=p)


def gamma_tensor(x: torch.Tensor,
                 values: torch.Tensor, p: torch.Tensor,
                 from_gamma: Tensorish = 2.2,
                 mode: str = 'NCHW') -> torch.Tensor:
    """ lerp(x, x**(2.2/values)m p) """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    values = get_sample_like(values, like=x)
    from_gamma = get_broadcastable(from_gamma, other=x, axis=1)

    if p_all(p):
        return gamma_proc(x, values, from_gamma=from_gamma)

    return torch.lerp(x, gamma_proc(x, values, from_gamma=2.2, inplace=False), p)

def gamma_proc(x, values=1.0, from_gamma=2.2, inplace: Optional[bool] = None):
    """ x**(2.2/values) """
    if x.requires_grad or not inplace:
        return x.pow(from_gamma/values)
    return x.pow_(from_gamma/values)

###
#
# functional for SoftClamp
#
def softclamp(data: TensorItem,
                soft: Union[type, Tensorish] = 1,
                p: Union[type, Tensorish] = 1,
                inflection: Union[type, Tensorish] = 0.5,
                for_display: bool = False,
                profile: bool = False) -> TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        soft        float [0]: 1 adds soft max min
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    p = get_bernoulli_like(p, like=data)
    soft = get_sample_like(soft, like=data)
    inflection = get_sample_like(inflection, like=data)

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=softclamp_tensor, for_display=for_display,
                      kind_keys=['data_1d', 'data_2d', 'data_3d'], soft=soft,
                      inflection=inflection, p=p)

def softclamp_tensor(x: torch.Tensor,
                 soft: Union[float, int],
                 p: torch.Tensor,
                 inflection: Union[type, Tensorish] = 0.5,
                 mode: str = 'NCHW') -> torch.Tensor:
    """ piecewise tanh clamp
    Args
        x           tensor
        soft        (int [1]): 0 -> linear 0-1, inf -> unit step function
        dims        (list, tuple)
    """
    p = get_bernoulli_like(p, like=x)

    if p_none(p):
        return x

    soft = get_sample_like(soft, like=x)
    inflection = get_sample_like(inflection, like=x)

    if p_all(p):
        return softclamp_proc(x, soft, inflection)

    return torch.lerp(x, softclamp_proc(x, soft, inflection), p)

def softclamp_proc(x: torch.Tensor, soft: Optional[torch.Tensor], inflection: torch.Tensor) -> torch.Tensor:
    """ piecewise tanh clamp
    Args
        x           tensor
        soft        tensor
    """
    if soft is None:
        return torch.clamp(x, 0, 1)
    dims = [i for i in range(len(soft.shape)) if soft.shape[i] > 1]
    x_max = tensor_apply_vals(x, "max", hold_axis=dims, keepdims=True)
    x_min = tensor_apply_vals(x, "min", hold_axis=dims, keepdims=True)

    # soft=1 match tangent
    if not torch.all(soft == 1):
        numax = torch.lerp(x_max, 1 + soft*x_max - soft, (x_max > 1).to(dtype=x_max.dtype))
        numin = torch.lerp(x_min, soft*x_min, (x_min < 0).to(dtype=x_max.dtype))

        delta = x_max.sub(x_min)
        nudelta = numax - numin
        x = x.sub(x_min).div(delta).mul(nudelta).add(numin)

        x_max = tensor_apply_vals(x, "max", hold_axis=dims, keepdims=True)
        x_min = tensor_apply_vals(x, "min", hold_axis=dims, keepdims=True)

    # upper inflection point
    _half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
    inflection = torch.min(torch.max(inflection, _half), torch.ones(1))

    _tol = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
    _ui = torch.min(torch.max(2 - x_max, inflection), 1 - _tol)
    _uidif = 1 - _ui

    # lower inflection point
    _li = torch.max(torch.min(-1*x_min, 1-inflection), _tol)

    return torch.where(x < _li, torch.tanh((x-_li)/_li)*_li + _li,
                       torch.where(x >= _ui, torch.tanh((x-_ui)/_uidif)*_uidif + _ui,
                                   x))
