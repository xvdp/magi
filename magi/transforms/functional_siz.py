"""@xvdp
functional for sizing transforms
"""
from typing import Union, Optional
import torch
from torch import nn
from koreto import Col
from .functional_base import transform, transform_profile, get_sample_like, get_bernoulli_like
from .functional_base import Tensorish, TensorItem
from ..utils import get_mgrid, slicer, crop_targets, transform_target, assert_equal, assert_in
from ..utils import ij__ji_mode, pos_offset__pos_pos_mode, pos_pos__pos_offset_mode

# pylint: disable=no-member
###
#
# functional for SqueezeCrop()
#
def squeeze_crop(data: TensorItem,
                 size: Union[int, tuple],
                 ratio: Tensorish,
                 interpolation: str = 'linear',
                 align_corners: bool = True,
                 for_display: bool = False,
                 profile: bool = False) -> TensorItem:
    """ crops then squeezes
    Args
        data    tensor | Item with tensor zeroth element
        size    int | tuple
        ratio   int, float, tensor, Distribution sampler : probabilistic argument
        interpolation   str ['linear'] | 'cubic' | 'nearest' | 'area'
    """
    crop_start, crop_end = _get_squeeze_crop_dims(data, ratio)
    size = _resolve_size(data[0].shape, size)

    _transform = transform_profile if profile else transform
    data = _transform(data=data, func=crop_resize_tensor, for_display=for_display,
                      kind_keys=['data_2d'], crop_start=crop_start, crop_end=crop_end,
                      target_size=size, interpolation=interpolation, align_corners=align_corners)

    if data.get_indices(kind='pos_2d'):
        data = _transform(data=data, func=crop_resize_targets, for_display=False,
                        kind_keys=['pos_2d'], crop_start=crop_start, crop_end=crop_end,
                        target_size=size)
    return data

###
#
# functional for CropResize() # aka ResizeCrop(), RandomResizeCrop()
#
def crop_resize(data: TensorItem,
                size: tuple,
                scale: Tensorish,
                ratio: Tensorish,
                i: Tensorish,
                j: Tensorish,
                variance: float,
                interpolation: str = "linear",
                for_display: bool = False,
                profile: bool = False,
                align_corners: bool = True) -> TensorItem:
    """ crops then squeezes
    Args
        data    tensor | Item with tensor zeroth element
        size    int | tuple
        scale   Distribution sampler : crop area
        ratio   Distribution sampler : aspect ratios
        i       Distribution sampler : crop offset i
        j       Distribution sampler : crop offset i
        variance    float [1.] lerp attenuatio of difference between random scales

        interpolation   str ['linear'] | 'cubic' | 'nearest' | 'area'
    """
    size = _resolve_size(data[0].shape, size)
    crop_start, crop_end = _get_crop_resize_dims(data[0], scale, ratio, i, j, variance)

    _transform = transform_profile if profile else transform
    data = _transform(data=data, func=crop_resize_tensor, for_display=for_display,
                      kind_keys=['data_2d'], crop_start=crop_start, crop_end=crop_end,
                      target_size=size, interpolation=interpolation, align_corners=align_corners)

    if data.get_indices(kind='pos_2d'):
        data = _transform(data=data, func=crop_resize_targets, for_display=False,
                        kind_keys=['pos_2d'], crop_start=crop_start, crop_end=crop_end,
                        target_size=size)
    return data

###
#
# standard resize and crop resize functions
#
def resize_tensor(x: torch.Tensor,
                  size: Union[int, tuple, None] = None,
                  interpolation: str = 'linear',
                  align_corners: bool = True) -> torch.Tensor:
    """ Resizes tensor
    """
    size = _resolve_size(x.shape, size)
    if x.shape[2:] == size:
        return x

    interpolation = _resolve_interpolation_mode(x.ndim, interpolation)
    return nn.functional.interpolate(x, size=size, mode=interpolation,
                                     align_corners=align_corners).clamp(x.min(), x.max())

def crop_resize_tensor(x: torch.Tensor,
                        crop_start: torch.Tensor,
                        crop_end: torch.Tensor,
                        target_size: Union[int, tuple, None] = None,
                        interpolation: str = 'linear',
                        align_corners: bool = True,
                        mode: str = 'NCHW') -> torch.Tensor:
    """
    tensor func for crop_resize and squeeze_crop
    Args
        x           tensor
        crop_start  (tensor) shape N,C,1,1,2 # where N and C are 1 dim is not expanded
        crop_end    (tensor) same shape as crop_start
    """
    expand_dims = [i for i in range(len(crop_start.shape[:-1])) if crop_start.shape[i] > 1]
    dim_sizes = [crop_start.shape[i] for i in expand_dims]
    assert_in(expand_dims, (0,1), msg="On 'Sizing' Transforms, expand_dims")
    assert_equal(crop_start.shape, crop_end.shape, msg="Shapes crop_start, crop_end")

    crop_start = crop_start.view(-1, 2)
    crop_end = crop_end.view(-1, 2)
    target_size = _resolve_size(x.shape, target_size)

    # single crop and resize
    if not expand_dims:
        return resize_tensor(x[:, :, crop_start[0, 0]:crop_end[0, 0],
                               crop_start[0, 1]:crop_end[0, 1]],
                             target_size, interpolation, align_corners)

    # stack multpile crops into tensor slices
    # get_mgrid + slicer: cycle thru permutations of batch elems & channels
    out = torch.zeros(*x.shape[:2], *target_size, dtype=x.dtype, device=x.device)
    indices = get_mgrid(dim_sizes, indices=True)
    for i, index in enumerate(indices):
        _nc = slicer(x.shape, expand_dims, index)
        out[_nc] = resize_tensor(x[_nc][:, :, crop_start[i, 0]:crop_end[i, 0],
                                        crop_start[i, 1]:crop_end[i, 1]],
                                 target_size, interpolation, align_corners)
    return out

def crop_resize_targets(x: Union[torch.Tensor, list, tuple],
                        crop_start: torch.Tensor,
                        crop_end: torch.Tensor,
                        target_size: Union[int, tuple, None] = None,
                        mode: str = 'yxyx') -> torch.Tensor:
    """
    target func for crop_resize and squeeze_crop
    Args
        x           tensor, list # targets, expected in xpath mode
            handled internally as list of tensors
        crop_start  (tensor) shape N,C,1,1,2 # where N and C are 1 dim is not expanded
        crop_size   (tensor) same shapea as crop_start
    """
    #
    # convert targets to list of tensors
    to_tensor = False
    if torch.is_tensor(x):
        x = [x]
        to_tensor = True
    # convert to positions 'y' to match tensor
    _mode = mode
    if mode[0] == 'x':
        x, _mode = ij__ji_mode(x, mode)
    if 'w' in mode:
        x, _mode = pos_offset__pos_pos_mode(x, _mode)

    #
    # propagate, broadcast and expand to batch
    target_size = torch.as_tensor(target_size, dtype=crop_end.dtype)
    target_size, crop_end = torch.broadcast_tensors(target_size, crop_end)
    scale = (target_size/(crop_end - crop_start))
    # propagate scaling to targets as list
    if len(x) == len(scale):
        scale = [s.view(-1,2) for s in torch.split(scale, [1]*len(scale), 0)]
    else: # propagate scales if batch size > 1 but all samples scaled together
        scale = [scale.view(-1,2)]*len(x)
    crop_start = torch.cat([crop_start]*len(x))
    crop_end = torch.cat([crop_end]*len(x))

    #
    # targets for each batch sample, cropped and scaled independently
    for i, _x in enumerate(x):
        # if channels are transformed independently:
        if crop_start.shape[1] > 1: 
            _x = torch.cat([x[i]]*crop_start.shape[1])
        x[i], drop =  crop_targets(_x, crop_start[i], crop_end[i], _mode)

        x[i] = transform_target(x[i], scale=scale[i]).view(-1, 1, *x[i].shape[2:])
        if drop:
            keep = torch.tensor([i for i in range(len(x[i])) if i not in drop], dtype=torch.int64)
            x[i] = x[i][keep]

    #
    # restore original mode and format of annotations
    if mode[0] != _mode[0]:
        x, _mode = ij__ji_mode(x, _mode)
    if mode != _mode and 'w' in mode:
        x, _mode = pos_pos__pos_offset_mode(x, _mode)
    assert_equal(_mode, mode, msg="Failed to convert modes ")

    if to_tensor:
        x = x[0]
    return x

###
#
# helpers for squeeze_crop
#
def _get_squeeze_crop_dims(x: TensorItem, ratio: Tensorish) -> tuple:
    """ Returns crop start and end as in shape(*x.shape, 2)

    expand_dims=None            torch.Size([1, 1, 1, 1, 2]) torch.Size([1, 1, 1, 1, 2])
    expand_dims=0               torch.Size([2, 1, 1, 1, 2]) torch.Size([2, 1, 1, 1, 2])
    expand_dims=(0, 1)          torch.Size([2, 3, 1, 1, 2]) torch.Size([2, 3, 1, 1, 2])
    expand_dims=(1,)            torch.Size([1, 3, 1, 1, 2]) torch.Size([1, 3, 1, 1, 2])
    # expand_dims > 2 would fail 
    """
    # get tensor from item
    x = x if torch.is_tensor(x) else x[0]

    # random sampler sampler
    ratio = get_sample_like(ratio, like=x).unsqueeze(-1)
    img_size = torch.tensor(x.shape[2:]).long().unsqueeze(0) # -> (1, H, W)
    anisotropy = torch.sub(*x.shape[2:]) # H - W
    start = torch.clamp(torch.tensor([anisotropy, -anisotropy]),
                        0, img_size.max()).mul(ratio/2.).long()
    end = img_size.sub(start)
    return start, end

def _resolve_size(input_shape: tuple, target_size: Union[int, tuple, None] = None) -> tuple:
    """ Returns a tuple of same dims as input_shape[2:], ie, HW, HWD etc.
    Args
        input_shape     shape of tensor NC...
        target_size     (int, tuple, None) if None, min(input_shape[2:])
    """
    input_size = input_shape[2:] # ouput same dims as input
    if target_size is None:
        target_size = min(input_size)
    if isinstance(target_size, int):
        target_size = tuple([target_size] * len(input_size))

    assert_equal(len(target_size), len(input_size), msg="Input and target size ")

    return target_size

def _resolve_interpolation_mode(ndim: int, interploation: str) -> str:
    """ supported interpolations in torch
    'nearest' | 'area' | 'linear' | 'bilinear' | 'trilinear' | 'bicubic'
    """
    prefix = ("", "bi", "tri")[ndim - 3]
    modes = ("nearest", "area", f"{prefix}linear", f"{prefix}cubic")
    if interploation in modes:
        return interploation

    if interploation in ("linear", "cubic"):
        return f"{prefix}{interploation}"

    assert_in(interploation, modes, msg="Iterpolation mode")

###
#
# helpers for crop_resize
#
"""
ratio   torch.Size([10, 1, 1, 1, 1])
ratio[]  torch.Size([10, 1, 1, 1, 2])
scale  torch.Size([10, 1, 1, 1, 1])
end    torch.Size([10, 1, 1, 1, 2])
size tensor([ 768., 1024.])
end [[[[[679, 711]]]], [[[[486, 384]]]], [[[[802, 820]]]], [[[[755, 684]]]], [[[[832, 689]]]], [[[[746, 779]]]], [[[[515, 650]]]], [[[[284, 232]]]], [[[[719, 657]]]], [[[[706, 594]]]]]


"""

def _get_crop_resize_dims(x: TensorItem,
                          scale_sampler: Tensorish,
                          ratio_sampler: Tensorish,
                          i_sampler: Tensorish,
                          j_sampler: Tensorish,
                          variance: float = 1.) -> tuple:
    """ return crop start, crop end
    statistically equivalent to randomness on RandomResizeCrop crops

    """
    num_samples = 10 # match RandomResizeCrop statistics sample 10x then choose center crop
    x = x if torch.is_tensor(x) else x[0]
    assert torch.is_tensor(x), f"{Col.YB} expected tensor found {type(x)}{Col.AU}"
    assert x.ndim == 4, f"{Col.YB} expected NCHW tensor found shape{x.shape}{Col.AU}"

    # input image size
    size = torch.as_tensor(x.shape[2:], dtype=x.dtype)

    # sampling: sub-areas sizes of original image
    scale = (get_sample_like(scale_sampler, like=x, num_samples=num_samples)*torch.prod(size)).unsqueeze(-1)
    # sampling: aspect ratios: LogUniform!
    ratio = get_sample_like(ratio_sampler, like=x, num_samples=num_samples).unsqueeze(-1)
    ratio = torch.cat((1/ratio, ratio), -1) # stack at end

    # output shape
    shape = list(ratio.shape)
    shape[0] = shape[0]//num_samples
    requested_crops = scale.numel()//num_samples

    # crop_end: ( add start on return, this is crop_size)
    crop_end = torch.round(torch.sqrt(scale * ratio))

    # some aspect ratios have both sides smaller than original size
    some = torch.prod((size >= crop_end.view(-1, 2)).to(dtype=torch.int), dim=1)
    crop_start = None

    # randomized aspect ratios
    crops = torch.where(some==1)[0][:requested_crops]
    if len(crops):
        view = (-1, 2)
        low_var = 0
        if shape[1] > 1: # if channels are expanded apply variance lerping only to channels
            view = (-1, shape[1], 2)
            low_var = (slice(0,None), slice(0,1), slice(0,None))

        crop_end = crop_end.view(-1,2)[crops].view(view)
        crop_end = torch.lerp(crop_end[low_var], crop_end, variance).view(-1,2)

        i_sampler.__.high = size[0] - crop_end[:, 0]
        j_sampler.__.high = size[1] - crop_end[:, 1]
        crop_start = torch.round(torch.stack((i_sampler.sample(requested_crops),
                                              j_sampler.sample(requested_crops)),
                                              axis=-1)).view(view)
        crop_start = torch.lerp(crop_start[low_var], crop_start, variance).view(-1,2)

    # center crops: if not sufficient random samples filled
    if requested_crops > len(crops) :
        _crop_end = size.clone()
        img_ratio = size[1]/size[0]           # width/height
        if ratio_sampler.__.__class__.__name__ == "LogUniform":
            low = ratio_sampler.__.e_low
            high = ratio_sampler.__.e_high
        elif ratio_sampler.__.__class__.__name__ == "Uniform":
            low = ratio_sampler.__.low
            high = ratio_sampler.__.high
        else:
            assert ratio_sampler.__.__class__.__name__ in ('LogUniform', 'Uniform'), f"unsupported distribution {ratio_sampler.__}"

        if img_ratio < low:       # is vertical, w < 3h/4
            _crop_end[0] = _crop_end[1] / low
        elif img_ratio > high:    # is horizontal w > 4h/3
            _crop_end[1] = _crop_end[0] * high

        # expand center crop to
        num_centered = requested_crops - len(crops)
        _crop_end = torch.stack([torch.round(_crop_end)]*num_centered, axis=0)
        _crop_start = size.sub(_crop_end).div(2)

        if crop_start is None:
            crop_end = _crop_end
            crop_start = _crop_start
        else:
            # append center crops to randomly sourced ratios and shuffle
            shuffle = torch.randperm(requested_crops)
            crop_end = torch.cat((crop_end, _crop_end))[shuffle]
            crop_start = torch.cat((crop_start, _crop_start))[shuffle]

    crop_start = crop_start.view(shape).to(dtype=torch.int64)
    crop_end = crop_end.view(shape).add(crop_start).to(dtype=torch.int64)

    return crop_start, crop_end

# def _get_random_resize_crop_parameters(height=100, width=100, elems=1,
#                 scale=(0.08, 1.0), ratio=(3./4., 4./3.), p=1, num_tries=10):
#     """ statistically equivalent to RandomResizeCrop
#     A log uinform sampler for aspect ratio coupled with uniform samplers for area and  offset
#     Curious elaborate sampling.
#     """

#     low, high = sorted(ratio)
#     size = torch.as_tensor([height, width], dtype=torch.float32)
#     area = torch.prod(size)
#     print(f"area    {area}")


#     # ratio
#     AspectRatio_Sampler =  LogUniform(low=low, high=high)

#     # offset
#     IJ_Sampler = torch.distributions.Uniform(low=0, high=size)

#     #scale
#     Area_Sampler = torch.distributions.Uniform(low=scale[0]*area, high=scale[1]*area)

#     samples = AspectRatio_Sampler.sample([num_tries]*elems)
#     samples = torch.stack((1/samples, samples))

#     target_area = Area_Sampler.sample([num_tries]*elems) # area
#     print(f"target_area {target_area}")
#         # h = torch.round(torch.sqrt(target_area / samples)).to(dtype=torch.int)
#         # w = torch.round(torch.sqrt(target_area * samples)).to(dtype=torch.int)

#     hw = torch.round(torch.sqrt(target_area * samples)).to(dtype=torch.int64)
#     some = (width >= hw[1]).to(dtype=torch.int64) * (height >= hw[0]).to(dtype=torch.int64)

#     if torch.any(some):
#         index = some.tolist().index(1)
#         IJ_Sampler.high = size - hw[:,index]
#         ij = torch.round(IJ_Sampler.sample())
#         hw = hw[:, index]
#         # return torch.round(IJ_Sampler.sample(), hw[:,index]

#     else:
#         # central crop, compress & crop larger dimension
#         hw = size.clone()

#         in_ratio = float(width) / float(height)
#         if in_ratio < low: # vertical, w < 3h/4
#             hw[0] = round(width / low)
#         elif in_ratio > high: # horizontal w > 4h/3
#             hw[1] = round(height * high)
#         ij = size.sub(hw).div(2)

#     return ij.to(dtype=torch.int64), hw.to(dtype=torch.int64)

