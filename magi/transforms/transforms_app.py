"""@xvdp
Appearance Transforms dont change size or positions of data

"""
from typing import Union
import numpy as np
import torch
from .transforms_base import TransformApp
from . import functional_app as F
from .. import config

_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]

__all__ = ["Normalize", "MeanCenter", "UnMeanCenter", "UnNormalize", "NormToRange"]

# pylint: disable=no-member
def _as_tensor(values: _Tensorish, default: torch.Tensor, ndims: int=4,
                axis: int=1) -> torch.Tensor:
    """ Broadcast to ndims tensor, default 4, N,C,H,W
        for normalization over C default 1
    """
    if values is None:
        values = default
    shape = [1]*ndims
    shape[axis] = -1
    return torch.as_tensor(default, dtype=torch.__dict__[config.DTYPE]).view(*shape)

class Normalize(TransformApp):
    """ similar to torchvision.transforms.Normalize

    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean        (sequence [None], float)
                        sequence:   per channel mean
                        None:       ImageNet mean
                        int, float: target mean
        std         (sequence [None], float)
                        sequence:   per channel stdev
                        None:       ImageNet  stdev
                        int, float: target stdev
        for_display (bool [None]) bypasses config.FOR_DISPLAY, if true Items are cloned
            ..SHOULD NOT BE USED unless original tensors are required unmodified

        ndims       (int [4]) num of dims of the item no normalize
        axis        (int [1]) axis containing channels
    """
    def __init__(self, mean: _Tensorish=None, std: _Tensorish=None,
                 for_display: bool=None,  ndims: int=4, axis: int=1) -> None:
        super().__init__(for_display=for_display)
        self.mean = _as_tensor(mean, default=[0.4993829, 0.47735223, 0.42281782],
                               ndims=ndims, axis=axis)
        self.std = _as_tensor(std, default=[0.23530918, 0.23156014, 0.23460476],
                              ndims=ndims, axis=axis)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns (data - mean) / std same type as input
        Args:
            data: tensor or Item or list
            **kwargs    any argument from __init__, locally
            same type as input
            profile: bool [False] wraps func in @memory_profile
        """
        kw_call, kw_ = self.update_kwargs(**kwargs)
        kw_call['profile'] = False if 'profile' not in kw_ else kw_.pop('profile')
        return F.normalize(data, **kw_call)

# alias to normalize
MeanCenter = Normalize

class UnNormalize(TransformApp):
    """
    UnNormalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean        (sequence [None], float)
                        sequence:   per channel mean
                        None:       ImageNet mean
                        int, float: target mean
        std         (sequence [None], float)
                        sequence:   per channel stdev
                        None:       ImageNet  stdev
                        int, float: target stdev
        for_display (bool [None]) bypasses config.FOR_DISPLAY, if true Items are cloned
            ..SHOULD NOT BE USED unless original tensors are required unmodified

        ndims       (int [4]) num of dims of the item no normalize
        axis        (int [1]) axis containing channels
    """
    def __init__(self, mean: _Tensorish=None, std: _Tensorish=None,
                 for_display: bool=None,  ndims: int=4, axis: int=1) -> None:
        super().__init__(for_display=for_display)
        self.mean =_as_tensor(mean, default=[0.4993829, 0.47735223, 0.42281782],
                              ndims=ndims, axis=axis)
        self.std = _as_tensor(std, default=[0.23530918, 0.23156014, 0.23460476],
                              ndims=ndims, axis=axis)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns (data - mean) / std same type as input
        Args:
            data: tensor or Item or list
            **kwargs    any argument from __init__, locally
            same type as input
            profile: bool [False] wraps func in @memory_profile
        """
        kw_call, kw_ = self.update_kwargs(**kwargs)
        kw_call['profile'] = False if 'profile' not in kw_ else kw_.pop('profile')
        return F.unnormalize(data, **kw_call)

UnMeanCenter = UnNormalize

class NormToRange(TransformApp):
    """map tensor linearly to a range
    Args:
        minimum     (float [0.]) min value of normalization
        maximum     (float [1.]) max value of normalization
        excess_only (bool [False]) when True leave images within range untouched
        independent (bool [True]) when True normalize per item in batch
        per_channel (bool [False]) when True normalize per channel
        for_display (bool [None]) bypasses config.FOR_DISPLAY, if true Items are cloned
            ..SHOULD NOT BE USED unless original tensors are required unmodified
    """
    def __init__(self, minimum: Union[float, int]=0.0, maximum: Union[float, int]=1.0,
                 excess_only: bool=False, independent: bool=True, per_channel: bool=False,
                 for_display: bool=None)-> None:
        super().__init__(for_display=for_display)

        self.minimum = minimum
        self.maximum = maximum
        self.excess_only = excess_only
        self.independent = independent
        self.per_channel = per_channel

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    any argument from __init__, locally
            same type as input
            profile: bool [False] wraps func in @memory_profile
        """
        kw_call, kw_ = self.update_kwargs(**kwargs)
        kw_call['profile'] = False if 'profile' not in kw_ else kw_.pop('profile')
        return F.normtorange(data, **kw_call)


class Saturate(TransformApp):
    """Manipulates image saturation
        Saturate to 0 is equivalent to Desaturate
        Saturate to -1 inverts image saturations
        Saturate to 2 over saturates image modulated modulated with piecewise tanh
        Args:
            sat_a:          (float) saturation target
            sat_b:          (float, None) random saturation between _a and _b
            p:              (float, 1) 0-1 bernoulli probability augmentation occurin
            distribution:   (str ['normal']), None takes distribution mode from config.py
                                normal | uniform | bernoulli
    """
    def __init__(self, sat_a: float, p: float=1, sat_b: float=None, distribution: str="normal",
                 independent: bool=True, for_display: bool=None)-> None:
        super().__init__(for_display=for_display)

        self.sat_a = sat_a
        self.sat_b = sat_b
        self.p = p
        self.distribution = distribution
        self.independent = independent

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    any argument from __init__, locally
            same type as input
            profile: bool [False] wraps func in @memory_profile
        """
        kw_call, _ = self.update_kwargs(**kwargs)
        return F.saturate(data, **kw_call)
