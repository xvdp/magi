"""@xvdp
Transforms that dont change size or positions of data

"""
from typing import Union, Any
import numpy as np
import torch
from .transforms_base import Transform
from . import functional_app as F
from .. import config

_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)

# pylint: disable=no-member
class Normalize(Transform):
    """ similar to torchvision.transforms.Normalize
    difference:
        float indicates target standard deviation

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
    def __init__(self, mean: Union[_torchable]=None, std: Union[_torchable]=None,
                 for_display: bool=None,  ndims: int=4, axis: int=1) -> None:

        self.mean = self._as_tensor(mean, default=[0.4993829, 0.47735223, 0.42281782],
                                    ndims=ndims, axis=axis)
        self.std = self._as_tensor(std, default=[0.23530918, 0.23156014, 0.23460476],
                                   ndims=ndims, axis=axis)

        self.for_display = for_display if for_display is not None else config.FOR_DISPLAY

    @staticmethod
    def _as_tensor(values: Union[_torchable], default: torch.Tensor, ndims: int=4,
                   axis: int=1) -> torch.Tensor:
        """ Broadcast to ndims tensor, default 4, N,C,H,W
            for normalization over C default 1
        """
        if values is None:
            values = default
        shape = [1]*ndims
        shape[axis] = -1
        return torch.as_tensor(default, dtype=torch.__dict__[config.DTYPE]).view(*shape)

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        """ Returns (data - mean) / std same type as input
        Args:
            data: tensor or Item or list
            **kwargs    any argument from __init__, locally
            same type as input

            profile: bool [False] wraps func in @memory_profile
        """
        kw_call, kw_other = self.update_kwargs(**kwargs)
        kw_call['profile'] = False if 'profile' not in kw_other else kw_other['profile']
        return F.normalize(data, **kw_call)

# alias to normalize
MeanCenter = Normalize
