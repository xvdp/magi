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
from typing import Union, Optional

import numpy as np
import torch
from .transforms_base import TransformApp
from .transforms_rnd import Values, Probs
from . import functional_app as F
from .. import config

_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]

__all__ = ["Normalize", "MeanCenter", "UnMeanCenter", "UnNormalize", "NormToRange", "Saturate"]

# pylint: disable=no-member
def _as_tensor(values: _Tensorish, ndims: int = 4, axis: int = 1) -> torch.Tensor:
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
    def __init__(self,
                 mean: _Tensorish = (0.4993829, 0.47735223, 0.42281782),
                 std: _Tensorish = (0.23530918, 0.23156014, 0.23460476),
                 ndims: int = 4,
                 axis: int = 1,
                 for_display: Optional[bool] = None, **kwargs) -> None:
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
        kw_call = self.update_kwargs(**kwargs)
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
    def __init__(self,
                 mean: _Tensorish = (0.4993829, 0.47735223, 0.42281782),
                 std: _Tensorish = (0.23530918, 0.23156014, 0.23460476),
                 ndims: int = 4,
                 axis: int = 1,
                 for_display: Optional[bool] = None) -> None:
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
        kw_call = self.update_kwargs(**kwargs)
        return F.unnormalize(data, **kw_call)

UnMeanCenter = UnNormalize

class NormToRange(TransformApp):
    """Map tensor linearly to a range
    Like all Transforms it can be with probabilistic settings, by default is constant
    Args:
        minimum         (float, tensor [0.]) min value of normalization
        maximum         (float, tensor [1.]) max value of normalization
        softclamp       (float [0]) # > 0 returns smooth clamp with tanh, inf is unit step function

        exand_dims      (int, tuple, None [0]) 0: normalize per batch sample
                                               1: normalize per channel
                                               ... higher may not make sense
    Args for probabilistic use: unlikely case but like other transforms can be applied
        p:              (float, tensor, float, [1]) 0-1 bernoulli probability
        p_dims:         (int, tuple, None [0]) 0: one 'p' per batch sample Probs('expand_dims')
        distribution    (str ['Uniform']) | distribution in Values()

        minimum_b:      (float, tensor, [None]) if None - 'minimum' is returned with probability p
        maximum_b:      (float, tensor, [None]) if None - 'maximum' is returned with probability p
        **kwargs    any kwarg prefixed 'minimum_<arg>' is passed to Values(minimum) as <arg>
                    idem for 'maximum_', other kwargs are passed to both Values()

        for_display (bool [None]) bypasses config.FOR_DISPLAY, if true Items are cloned
            ..SHOULD NOT BE USED unless original tensors are required unmodified

    Examples
    >>> N = NormToRange() # most common case, normalizes data to range, 0-1, independent samples
    >>> N = NormToRange(minimum=0, maximum=1, expand_dims=(0,), p=1) # equivalent
    >>> N(img)

    # normalize with probabilities per channel, clone output insetad of overwriting
    >>> N = NormToRange(minimum=0, maximum=1, for_display=True, p=[0,0.5,1])

    # normalize each channel independently
    >>> N = NormToRange(minimum=0, maximum=1, expand_dims=(0,1))

    # normalize categorically over two different ranges, samples and channels independently
    >>> N = NormToRange(expand_dims=(0,1), maximum=1.2, minimum=-0.5,  maximum_b=0.8, minimum_b=0.2,
                        p=0.5, distribution='Categorical')
    """
    def __init__(self,
                 minimum: _Tensorish = 0.0,
                 maximum: _Tensorish = 1.0,
                 p: _Tensorish = 1.0,           # probability / optional args
                 p_dims: Union[None, int, tuple, list] = 0,     # vary probability over dims
                 distribution: Optional[str] = "Uniform",
                 expand_dims: Union[None, int, tuple, list] = 0, # vary norm over dims
                 minimum_b: _Tensorish = None,
                 maximum_b: _Tensorish = None,
                 for_display: Optional[bool] = None,            # clone before appying
                 **kwargs) -> None:                             # kwargs for transforms.Values()

        super().__init__(for_display=for_display)

        # kwargs prefixed 'minimum_' or 'maximum_' are passed to each Valueothers to both
        kw_min = {key:kwargs.pop(k) for k, key in self.extract_keys("minimum_", **kwargs).items()}
        kw_max = {key:kwargs.pop(k) for k, key in self.extract_keys("maximum_", **kwargs).items()}

        self.minimum = Values(a=minimum, b=minimum_b, expand_dims=expand_dims,
                              distribution=distribution, **{**kwargs, **kw_min})
        self.maximum = Values(a=maximum, b=maximum_b, expand_dims=expand_dims,
                              distribution=distribution, **{**kwargs, **kw_max})
        self.p = Probs(p=p, expand_dims=p_dims)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    overwritedes to class
                maximum, minimum, p,
                for_display (bool [False]) clones data
                profile     (bool [False]) wraps func in @memory_profile
        """
        kw_call = self.update_kwargs(**kwargs)
        return F.normtorange(data, **kw_call)


class Saturate(TransformApp):
    """ Alters saturation lerping desaturated image
    Saturate to 0 is equivalent to Desaturate
    Saturate to -1 inverts image saturations
    Saturate to 2 over saturates image modulated modulated with piecewise tanh
    Args:
        p:              (float, tensor, float, [1]) 0-1 bernoulli probability,
        p_dims:         (int, tuple, None [0]) 0: one 'p' per batch sample Probs('expand_dims')

        a:              (float, tensor [0]) saturation target: default desaturate
        b:              (float, tensor, [None]) if None - 'a' is returned with probability p

        distribution    (str ['Uniform']) Normal | Categorical | Cauchy | Laplace ...
        exand_dims      (int, tuple, None [0]) 0: one distribution sample per batch sample

        for_display     (bool [False]) clones before saturation
        **kwargs for saturation value distributions
            c,..., z not in (i, j, k, p) - values

    Examples:
    # Desaturate with probability of 1
    >>> Sat = Saturate() # or Saturate(a=0, p=1)

    # Uniform distribution over N,H (in an NCHW) vector, between saturation -3 and 3, with 60% prob
    >>> Sat = Saturate(a=-3,b=3, expand_dims=(0,2), distribution='Uniform', for_display=True, p=0.6)

    # Categorical distribution with equal 'probs' between saturations (-2, 0, 3, 10) with 90% prob
    >>> Sat = Saturate(a=-2, b=3, c=10, d=0, p=0.9, for_display=True, distribution="Categorical")

    # Gumbel distribution centered over 3std range of sat, -2,2, with RGB probs 0%, 50%, 100%
    >>> Sat = Saturate(a=-3,b=3,expand_dims=0, distribution='Gumbel', for_display=True, p=[0,0.5,1])

    # Call above definitions
    >>> Sat(tensor_img)

    # Override saturation value and probability on call
    >>> Sat(tensor_img, sat=-2), p=0.1)

    # Override saturation value per channel R*=-2, G*-0, B*=2 on call
    >>> Sat(m, sat=torch.tensor([[-2, 0, 2]]))
    """
    def __init__(self,
                 p: _Tensorish = 1.0,                           # probability
                 p_dims: Union[None, int, tuple, list] = 0,     # vary probability over dims
                 a: _Tensorish = 0.,                            # target saturation
                 b: Optional[_Tensorish] = None,                # second target
                 distribution: Optional[str] = "Uniform",
                 expand_dims: Union[None, int, tuple, list] = 0,# vary saturation over dims
                 for_display: Optional[bool] = None,            # clone before appying
                 **kwargs) -> None:                             # kwargs for transforms.Values()
        super().__init__(for_display=for_display)
        self.sat = Values(a=a, b=b, expand_dims=expand_dims, distribution=distribution, **kwargs)
        self.p = Probs(p=p, expand_dims=p_dims)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    overwritedes to class
                value   (float, tensor)
                p       (float, tensor)
                for_display (bool [False]) clones data
                profile     (bool [False]) wraps func in @memory_profile
        """
        kw_call = self.update_kwargs(**kwargs)
        return F.saturate(data, **kw_call)


class Gamma(TransformApp):
    """ Changes gamma assumes gamma 2.2
    To blend betweeen current and linear
    Args:
        a           (float >0  [1.0])   target gamma a
        b           (float > 0 [None])   target gamma b, if None int between input and a with prob p
                        b is float > 0, continuous sample distribution
        p           (float 0-1 [0.1])   bernoulli chance of transform
        distribution(str  ["Normal"])   in Values
                        None:       discrete between a and b, (or if b is None, a if p, else None)
    """
    def __init__(self,
                 p: _Tensorish = 1.0,                           # probability
                 p_dims: Union[None, int, tuple, list] = 0,     # vary probability over dims
                 a: _Tensorish = 1.0,                           # target gamma
                 b: Optional[_Tensorish] = None,                # second target
                 from_gamma: _Tensorish = 2.2,
                 distribution: Optional[str] = "Normal",
                 expand_dims: Union[None, int, tuple, list] = 0,# vary saturation over dims
                 for_display: Optional[bool] = None,            # clone before appying
                 **kwargs) -> None:                             # kwargs for transforms.Values()

        super().__init__(for_display=for_display)

        self.value = Values(a=a, b=b, expand_dims=expand_dims, distribution=distribution, **kwargs)
        self.from_gamma = from_gamma
        self.p = Probs(p=p, expand_dims=p_dims)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    overwritedes to class
                value   (float, tensor)
                p       (float, tensor)
                for_display (bool [False]) clones data
                profile     (bool [False]) wraps func in @memory_profile
        """
        kw_call = self.update_kwargs(**kwargs)
        return F.gamma(data, **kw_call)

class SoftClamp(TransformApp):
    """ SoftClamp
    Clamp between 0,1 with tanh
    Args:
        a           (float >0  [1.0])   target softness a
        b           (float > 0 [None])  target softness b
                        b is float > 0, continuous sample distribution
        inflection  (float >= 0.5 < 1. [0.5])
        p           (float 0-1 [0.1])   bernoulli chance of transform
        distribution(str  ["Normal"])   in Values
                        None:       discrete between a and b, (or if b is None, a if p, else None)
    """
    def __init__(self,
                 p: _Tensorish = 1.0,                           # probability
                 p_dims: Union[None, int, tuple, list] = 0,     # vary probability over dims
                 a: _Tensorish = 1.0,       # 0 -> torch.clamp(0,1), inf -> unit step function
                 b: Optional[_Tensorish] = None,                # second target softness
                 inflection: _Tensorish = 0.5,                  # inflection point 0.5 -> 1.
                 distribution: Optional[str] = "Normal",
                 expand_dims: Union[None, int, tuple, list] = 0,# vary softness over dims
                 for_display: Optional[bool] = None,            # clone before appying
                 **kwargs) -> None:                             # kwargs for transforms.Values()

        super().__init__(for_display=for_display)
        kw_infl = {key:kwargs.pop(k) for k, key in self.extract_keys("inflection_", **kwargs).items()}

        self.soft = Values(a=a, b=b, expand_dims=expand_dims, distribution=distribution, **kwargs)
        self.inflection = Values(a=inflection, expand_dims=expand_dims, distribution=distribution,
                                 **{**kwargs, **kw_infl})
        self.p = Probs(p=p, expand_dims=p_dims)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns data in range (minimum, maximum)
        Args:
            data: tensor or Item or list
            **kwargs    overwritedes to class
                soft    (float, tensor)
                p       (float, tensor)
                for_display (bool [False]) clones data
                profile     (bool [False]) wraps func in @memory_profile
        """
        kw_call = self.update_kwargs(**kwargs)
        return F.softclamp(data, **kw_call)
