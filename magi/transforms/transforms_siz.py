"""@xvdp
Sizing Transforms, output tensor size is changed

"""

from typing import Union, Optional
import logging
import numpy as np
import torch
from torch.distributions import Bernoulli
import torchvision.transforms as TT
from koreto import Col

from .transforms_base import Transform
from .transforms_rnd import Values
from . import functional_siz as F
from .. import config


# pylint: disable=no-member
#####
#
# crops
#
def valid_dims(expand_dims, max_dim=1):
    if expand_dims is None:
        return expand_dims
    if isinstance(expand_dims, int):
        expand_dims = (expand_dims,)
    assert isinstance(expand_dims, (list, tuple)), f"expand dims, tuple expected, got {expand_dims}"
    if len(expand_dims) == 0:
        return None
    if max_dim is not None:
        assert max(expand_dims) <= max_dim, f"max epxandable dimesnion {max_dim}, got {expand_dims}"
    return expand_dims

class SqueezeCrop(Transform):
    """Crops the given Torch Image and Size
    Args:
        size (int, tuple of ints, None): output size of crop
            None: size is min side
            int: crop to square
        interploation   (str ['linear']) | 'cubic'

        ratio: (float) [0.5]) squeeze to crop ratio
            if ratio == 0: only squeezes
            if ratio == 1: only crops

        ratio_b (float [None]) if second ratio passed, squeeze to crop ratio can be prbabilistic
        distribution (str ['Uniform']) | in Values
        expand_dims (tuple, int [None]), max: 1, Sizing Transforms only expands Batch or channel dim

    TODO ratio and ratio_b, a, b, loc= scale= ..
    TODO targets -> annotation strategy for multiple channels
    TODO collapse identical dims
    """
    __type__ = "Sizing"
    def __init__(self,
                 size: Union[None, int, list, tuple] = None,
                 ratio: float = 0.5,
                 ratio_b: Optional[float] = None,
                 interpolation: str = "linear",
                 expand_dims: Union[None, int, tuple] = None,
                 distribution: str = "Uniform",
                 for_display: Optional[bool] = None,
                 **kwargs) -> None:
        super().__init__(for_display=for_display)

        self.size = size
        ratio = min(1, max(ratio, 0))
        ratio_b = ratio_b if ratio_b is None else min(1, max(ratio_b, 0))

        expand_dims = valid_dims(expand_dims)
        self.ratio = Values(a=ratio, b=ratio_b, expand_dims=expand_dims,
                            distribution=distribution, **kwargs)
        self.interpolation = interpolation

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
        return F.squeeze_crop(data, **kw_call)

# class ResizeCrop(Transform):
#     """
#     similar to CenterCrop
#     similar to torchvision.transforms.RandomResizeCrop
#     Given torch image to validate training. its kind of a silly crop
#         size            (tuple, int) expected output size of each edge
#         scale           (tuple) range of size of the origin size cropped
#         ratio           (tuple) range of aspect ratio of the origin aspect ratio cropped
#         interpolation   (str ["linear"]) in ['linear' 'cubic']
#         p               (int [1])   0: no randomness

#     """
#     __type__ = "Affine"
#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation="linear",
#                  p=1):
#         if isinstance(size, tuple):
#             self.size = size
#         else:
#             self.size = (size, size)
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("range should be of kind (min, max)")

#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio
#         self.p = p

#     def __call__(self, data, **kwargs):
#         """
#         Args:
#             data: tuple of
#                 tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
#                 target_tensor (tensor): annotations, interpolated
#                 labels
#             **kwargs, any of the __init__ args can be overridden
#         Returns:
#             tensor, target_tensor, labels
#         """
#         args, _ = update_kwargs(self, **kwargs)

#         return F.resize_crop(data, args["size"], args["scale"], args["ratio"],
#                              args["interpolation"], args["p"])

#     def __repr__(self):
#         return _make_repr(self)

# class CenterCrop(Transform):
#     """Crops the given Torch Image at the center.
#         size optional, if no size, crop to minimum edge
#         acts on tensors

#         random shift with choice of alpha(0-1) to edge and distribution
#         random zoom with choice of alpha(0-1) to target size, if size set
#     Args:
#         size (sequence or int, [None]): Desired output size of the crop.
#             If size is an int, a square crop (size, size) is made.
#             If size is None, square crop of length of min dimension is made.

#         center_choice   (int, [2])
#                             1: image center
#                             2: if targets exist: target items boudning box center
#                                 3: mean (1,2) ...
#         # shift then crop
#         shift           (float, [0.1]), maximum allowed center shift in random center crops
#                             recommended 0 < shift < 0.5
#         distribution    (enum config.RnDMode ["normal"]) "normal" / "uniform"

#         # zoom then crop
#         zoom            (float, [0]) 0-1; zoom, requires size to be set to work
#                             zoom=0 crops exact pixels, zoom=1, crops min dim and resizes
#         zdistribution   (enum config.RnDMode ["normal"]) "normal" / "uniform"
#                             normal is bounded normal between zoom 0, 1
#         zskew           (float, [0.5]), 0,1;  0.5 means normal
#                             1 skewed towards full zoom, 0 skewed towards no zoom
#     """
#     __type__ = "Affine"
#     def __init__(self, size=None, center_choice=2, shift=0.1, distribution="normal", zoom=0,
#                  zdistribution="normal", zskew=0.5):
#         if size is not None:
#             assert (isinstance(size, numbers.Number) and size > 0) or isinstance(size, iterable)
#             if isinstance(size, iterable):
#                 for _s in size:
#                     assert isinstance(_s, numbers.Number) and _s > 0
#         self.size = size
#         self.center_choice = center_choice
#         self.shift = shift
#         self.distribution = distribution
#         self.zoom = zoom
#         self.zdistribution = zdistribution
#         self.zskew = zskew

#     def __call__(self, data, **kwargs):
#         """
#         Args:
#             data: tuple of
#                 tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
#                 target_tensor (tensor): annotations, interpolated
#                 labels
#             **kwargs, any of the __init__ args can be overridden
#         Returns:
#             tensor, target_tensor, labels
#         """
#         args, _ = update_kwargs(self, **kwargs)

#         return F.center_crop(data, args["size"], args["center_choice"], args["shift"],
#                              args["distribution"], args["zoom"], args["zdistribution"],
#                              args["zskew"])

#     def __repr__(self):
#         return _make_repr(self)
