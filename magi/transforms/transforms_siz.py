"""@xvdp
Sizing Transforms, output tensor size is changed

"""

from typing import Union, Optional
import torch

from .transforms_base import Transform
from .transforms_rnd import Values, validate_dims
from . import functional_siz as F

# pylint: disable=no-member
#####
#
# crops

class SqueezeCrop(Transform):
    """Sizing Transform: Crops and resizes to square,
    Args:
        size (int, tuple of ints, None): output size of crop: NOT probailisic
            None: size is min side
            int: crop to square
        interploation   (str ['linear']) | 'cubic'

        ratio: (float) [0.5]) squeeze to crop ratio: probabilistic arg
            if ratio == 0: only squeezes
            if ratio == 1: only crops
        ratio_b (float [None]) if second ratio passed, squeeze to crop ratio can be prbabilistic
        distribution (str ['Uniform']) | in Values
        expand_dims (tuple, int [None]), max: 1, Sizing Transforms only expands Batch or channel dim
    TODO: add probabilistic crop shift and crop percentage of smallest side.
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

        expand_dims = validate_dims(expand_dims, allowed=(None, 0, 1),
                                    msg="On 'Sizing' Transforms, expand_dims")
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

class CropResize(Transform):
    """ Crop then Resize: similar to torchvision.transforms.RandomResizeCrop
    Statistically equivalent when run on defaults.
    There are several cockamamie defaults in this "standard transform":
        Aspect ratio LogUniform
        Crop size is determined by a random uniform distribution between 0 and random uniform scale
    The samplers here are set up to allow benchmarking with identical parameters - the default
    Control cropping randomness a bit more precisely, and expanding dimensions

    Args
        size                (tuple, int) output size of image: not probabilisitc

        scale               (int, float , tuple [(0.08, 1.])) area scales for intermediary crop
            # if int | float: scale is constant
        scale_distribution  (str ['Uniform'])

        ratio               (tuple, int [(3/4, 4/3)]) aspect ratios w/h of the intermediary crop
            # if int: ratio is constant
        ratio_distribution  (str ['LogNormal'])
            # LogNormal: to match RandomResizeCrop():  low ratio is ~1.75X more likely than high
            # Im not sure why this was done but to match defaults...

        variance      (float [1.]) 0 - 1 - 0: all batch elements are identical, 1: default
        interpolation   (str ["linear"]) in ['linear' 'cubic'] # scaling interpolation
        for_display     (bool [None])

    """
    __type__ = "Sizing"
    def __init__(self,
                 size: Union[int, tuple, list],
                 scale: Union[int, float, tuple] = (0.08, 1.0),
                 scale_distribution: str = "Uniform",
                 ratio: Union[int, float, tuple, list] = (3./4., 4./3.),
                 ratio_distribution: str = "LogUniform",
                 expand_dims: Union[None, int, tuple, list] = (0,),
                 variance: float = 1,
                 interpolation: str = "linear",
                 for_display: Optional[bool] = None,
                 **kwargs) -> None:
        super().__init__(for_display=for_display)

        # non probabilistic argument
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

        # constrain to batch and channel expansion
        expand_dims = validate_dims(expand_dims, allowed=(None, 0, 1),
                                    msg="On 'Sizing' Transforms, expand_dims")
        # crop target area sampler
        scale = (scale,) if isinstance(scale, (int, float)) else scale
        self.scale = Values(*scale, expand_dims=expand_dims,
                            distribution=scale_distribution, **kwargs)

        # aspect ratio sampler
        ratio = (ratio,) if isinstance(ratio, (int, float)) else ratio
        self.ratio = Values(*ratio, expand_dims=expand_dims,
                            distribution=ratio_distribution, **kwargs)

        # offset sampler; updates on data with high=(height-h, width-w)
        self.i = Values(low=0, high=100, distribution="Uniform")
        self.j = Values(low=0, high=100, distribution="Uniform")
        self.variance = variance

        self.interpolation = interpolation

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor, labels
        """
        kw_call = self.update_kwargs(**kwargs)
        return F.crop_resize(data, **kw_call)

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
