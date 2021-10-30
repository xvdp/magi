"""@xvdp
Appearance Transforms dont change size or positions of data

TODO: p and distribution to Normalize: mean, std

Distribution transform: - for [mean, std]
-> value, p
->  distribution type:
    uniform: [value, other]
    normal: [value, sigma[3], clamp values eg,[0.1, None]
        eg std: normal [s3 - std - s3+] > 0.01


uniform[mean_0 - mean_1]


"""
from typing import Union
import numpy as np
import torch
from .transforms_base import Randomize, TransformApp
from . import functional_app as F
from .. import config

_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]

__all__ = ["Normalize", "MeanCenter", "UnMeanCenter", "UnNormalize", "NormToRange"]

# pylint: disable=no-member
def _as_tensor(values: _Tensorish, ndims: int=4, axis: int=1) -> torch.Tensor:
    """ Broadcast to ndims tensor, default 4, N,C,H,W
        for normalization over C default 1
    """
    shape = [1]*ndims
    shape[axis] = -1
    return torch.as_tensor(values, dtype=torch.__dict__[config.DTYPE]).view(*shape)

class Normalize(TransformApp):
    """ similar to torchvision.transforms.Normalize

    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

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
    def __init__(self, mean: _Tensorish=[0.4993829, 0.47735223, 0.42281782],
                 std: _Tensorish=[0.23530918, 0.23156014, 0.23460476], ndims: int=4, axis: int=1,
                 for_display: bool=None, **kwargs) -> None:
        super().__init__(for_display=for_display)
        self.mean = _as_tensor(values=mean, ndims=ndims, axis=axis)
        self.std = _as_tensor(values=std, ndims=ndims, axis=axis)

        # if "meanb" in kwargs:

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
    def __init__(self, mean: _Tensorish=[0.4993829, 0.47735223, 0.42281782],
                 std: _Tensorish=[0.23530918, 0.23156014, 0.23460476],  ndims: int=4, axis: int=1,
                 for_display: bool=None) -> None:
        super().__init__(for_display=for_display)
        self.mean =_as_tensor(values=mean, ndims=ndims, axis=axis)
        self.std = _as_tensor(values=std, ndims=ndims, axis=axis)

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
            sat_b:          (float, [None]) if None
            p:              (float, 1) 0-1 bernoulli probability augmentation occuring
            distribution:   (str ['Uniform']) Normal | or any
    """
    def __init__(self, p: float=1, sat_a: float=0., sat_b: float=None, distribution: str="Uniform",
                 per_sample: bool=True, per_channel: bool=False, for_display: bool=None)-> None:
        super().__init__(for_display=for_display)

        # self.sampler = Values(sat_a, sat_b, p, distribution, per_channel, per_sample)
        self.sat_a = sat_a
        self.sat_b = sat_b
        self.p = p
        self.distribution = distribution
        self.per_sample = per_sample
        self.per_channel = per_channel

        # dtype = data.dtype if 
        # sampler = Randomize(a=sat_a, b=sat_b, p=p, distribution=distribution, per_channel=per_channel, per_sample=per_sample)


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
