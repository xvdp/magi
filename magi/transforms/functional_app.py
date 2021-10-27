"""@xvdp

Appearance functionals
    only modify content values of data
    without changing data size or relative location of values
Functionals are tagged for subtypes,
eg, normalize(), normalize_tensor()

Functionals
    normalize()

"""
from typing import Union
import numpy as np
import torch

from ..utils import ensure_broadcastable
from ..features import Item
from .functional_base import transform, transform_profile
_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]
_TensorItem = Union[torch.Tensor, Item]
# pylint: disable=no-member


###
# functional for Normalize or MeanCenter
#
def normalize(data: _TensorItem, mean: _Tensorish, std: _Tensorish, for_display: bool=False,
              profile: bool=False)-> _TensorItem:
    """  normalize item or tensor | profile memory optional
    Args
        data        tensor | Item
        mean        tensor, iterable convertible to tensor
        std         tensor, iterable convertible to tensor
        for_display bool [False] CLONES Item ( not applied if data is tensor )
        profile     bool [False] wraps transform in profiler
    """
    _transform = transform_profile if profile else transform
    return _transform(data=data, func=normalize_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std)

def normalize_tensor(x: torch.Tensor, mean: _Tensorish, std: _Tensorish)-> torch.Tensor:
    """
    Args
        x           tensor
        mean, std.  appropriately broadcasted tensors, same dtype and device
    """
    mean = ensure_broadcastable(mean, x)
    std = ensure_broadcastable(std, x)
    if x.requires_grad:
        return x.sub(mean).div(std)
    return x.sub_(mean).div_(std)

###
# functional for UnNormalize or UnMeanCenter
#
def unnormalize(data: _TensorItem, mean: _Tensorish, std: _Tensorish,
                for_display: bool=False, profile: bool=False)-> _TensorItem:
    """  unnormalize item or tensor | profile memory optional """

    _transform = transform_profile if profile else transform
    return _transform(data=data, func=unnormalize_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], mean=mean, std=std)

def unnormalize_tensor(x: torch.Tensor, mean: _Tensorish, std: _Tensorish,
                       clip: bool=False)-> torch.Tensor:
    """
    Args
        x           tensor
        mean, std.  appropriately broadcasted tensors, same dtype and device
    """
    mean = ensure_broadcastable(mean, x)
    std = ensure_broadcastable(std, x)
    if x.requires_grad: # not im place
        x = x.mul(std).add(mean)
        if clip:
            x = x.clamp(0., 1.)
    else:
        x.mul_(std).add_(mean)
        if clip:
            x.clamp_(0., 1.)
    return x

###
# functional for NormToRange
#
def normtorange(data: _TensorItem, minimum: Union[int, float]=0., maximum: Union[int, float]=1.,
                excess_only: bool=False, independent: bool=True, per_channel: bool=False,
                for_display: bool=False, profile: bool=False)-> _TensorItem:
    """  unnormalize item or tensor | profile memory optional
    Args
        data        Item | tensor
        minimum     int | float [0.]
        maximum     int | float [1.]
        excess_only bool [False] - if minimum < data.min() and maximum > data.max(), dont modify
        independent bool [True]  - normalize each batch item separately
        per_channel bool [False] - normalize each channel separately
    """
    _transform = transform_profile if profile else transform
    return _transform(data=data, func=normtorange_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], minimum=minimum, maximum=maximum,
                      excess_only=excess_only, independent=independent, per_channel=per_channel)

def normtorange_tensor(x: torch.Tensor, minimum: float, maximum: float,
                       excess_only: bool, independent: bool, per_channel: bool)-> torch.Tensor:
    """
    Args
        x           tensor
        independent     normalize each batch element on its own range
    """
    if independent:
        for i, _ in enumerate(x):
            x[i:i+1] = normtorange_tensor(x[i:i+1], minimum=minimum, maximum=maximum,
                                          excess_only=excess_only, independent=False,
                                          per_channel=per_channel)
        return x

    _to = {"dtype": x.dtype, "device": x.device}
    if per_channel:
        _sh = [1]*x.ndim
        _sh[1] = x.shape[1]
        _min = torch.tensor([x[:, i, ...].min() for i in range(_sh[1])], **_to).view(_sh)
        _max = torch.tensor([x[:, i, ...].max() for i in range(_sh[1])], **_to).view(_sh)

    _min = x.min()
    _max = x.max()

    if not excess_only or x.min().item() < minimum or x.max().item() > maximum:
        _denom = _max.sub(_min).add(minimum)
        _deneq0 = (_denom == 0).to(dtype=_denom.dtype)
        _denom.add_(_deneq0) # NaN prevent if denom == 0: denom = 1

    if x.requires_grad:
        return x.sub(_min).mul(maximum - minimum).div(_denom)
    return x.sub_(_min).mul_(maximum - minimum).div_(_denom)

###
# functional for Saturate
#
def saturate(data: _TensorItem, sat_a: float, sat_b: float, p: float,
             distribution: str, independent: bool, for_display: bool=False,
             profile: bool=False)->  _TensorItem:
    """
    """
    _transform = transform_profile if profile else transform
    return _transform(data=data, func=saturate_tensor, for_display=for_display,
                      meta_keys=['data_1d', 'data_2d', 'data_3d'], sat_a=sat_a,
                      sat_b=sat_b, p=p, distribution=distribution, independent=independent)

def saturate_tensor(x: torch.Tensor, sat_a: float, sat_b: float, p: float,
                    distribution: str, independent: bool)-> torch.Tensor:
    pass
        # self.sat_a = sat_a
        # self.sat_b = sat_b
        # self.p = p
        # self.distribution = distribution
        # self.independent = independent