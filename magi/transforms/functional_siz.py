"""@xvdp
functional for sizing transforms
"""
from typing import Union, Optional
import torch
from torch import nn
from koreto import Col
from .functional_base import transform, transform_profile, get_sample_like, Tensorish, TensorItem
from ..utils import get_mgrid, slicer

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

    crop_start, crop_size = get_crop_info(data, ratio)

    _transform = transform_profile if profile else transform
    data = _transform(data=data, func=squeeze_crop_tensor, for_display=for_display,
                      kind_keys=['data_2d'], crop_start=crop_start, crop_size=crop_size,
                      target_size=size, interpolation=interpolation, align_corners=align_corners)

    # data = _transform(data=data, func=squeeze_crop_targets, for_display=False,
    #                   kind_keys=['pos_2d'], size=size, start=start, crop_size=crop_size,
    #                   interpolation=interpolation)
    return data

def get_crop_info(x: TensorItem, ratio: Tensorish) -> tuple:
    """ returns crop start and size as in shape(*x.shape, 2)

    expand_dims=None            torch.Size([1, 1, 1, 1, 2]) torch.Size([1, 1, 1, 1, 2])
    expand_dims=0               torch.Size([2, 1, 1, 1, 2]) torch.Size([2, 1, 1, 1, 2])
    expand_dims=(0, 1)          torch.Size([2, 3, 1, 1, 2]) torch.Size([2, 3, 1, 1, 2])
    expand_dims=(1,)            torch.Size([1, 3, 1, 1, 2]) torch.Size([1, 3, 1, 1, 2])
    expand_dims=(0, 1, 2)       torch.Size([2, 3, 582, 1, 2]) torch.Size([2, 3, 582, 1, 2])
    expand_dims=(0, 2)          torch.Size([2, 1, 582, 1, 2]) torch.Size([2, 1, 582, 1, 2])
    expand_dims=(0, 1, 2, 3)    torch.Size([2, 3, 582, 1024, 2]) torch.Size([2, 3, 582, 1024, 2])
    # expand_dims = [i for i in range(len(start.shape[:-1])) if start.shape[i] > 1]
    """
    x = x if torch.is_tensor(x) else x[0]
    ratio = get_sample_like(ratio, like=x).unsqueeze(-1)
    img_size = torch.tensor(x.shape[2:]).long().unsqueeze(0) # H, W
    anisotropy = torch.sub(*x.shape[2:]) # H - W
    start = torch.clamp(torch.tensor([anisotropy, -anisotropy]),
                        0, img_size.max()).mul(ratio/2.).long()
    size = img_size.sub(start)
    return start, size

def resolve_size(input_shape: tuple, target_size: Union[int, tuple, None] = None) -> tuple:
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
    assert len(target_size) == len(input_size), f"size required to be len {len(input_size)} got {len(target_size)}"
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

    assert interploation in modes, f"invalid mode='{interploation}', expected {modes}"

def resize_tensor(x: torch.Tensor,
                  size: Union[int, tuple, None] = None,
                  interpolation: str = 'linear',
                  align_corners: bool = True) -> torch.Tensor:
    """ Resizes tensor """
    size = resolve_size(x.shape, size)
    if x.shape[2:] == size:
        return x

    interpolation = _resolve_interpolation_mode(x.ndim, interpolation)
    return nn.functional.interpolate(x, size=size, mode=interpolation,
                                     align_corners=align_corners).clamp(x.min(), x.max())


def squeeze_crop_tensor(x: torch.Tensor,
                        crop_start: torch.Tensor,
                        crop_size: torch.Tensor,
                        target_size: Union[int, tuple, None] = None,
                        interpolation: str = 'linear',
                        align_corners: bool = True) -> torch.Tensor:
    """
    Args
        x           tensor
        crop_start  (tensor) shape N,C,1,1,2 # where N and C are 1 dim is not expanded
        crop_size   (tensor) same shapea as crop_start

    """
    expand_dims = [i for i in range(len(crop_start.shape[:-1])) if crop_start.shape[i] > 1]
    if expand_dims:
        _msg = f"{Col.YB}Sizing transforms only allow expanding dims 0 | 1{expand_dims}{Col.AU}"
        assert max(expand_dims) <= 1, _msg

        # TODO collapse expanded dims if all are equal

    dims = [crop_start.shape[i] for i in expand_dims]
    crop_start = crop_start.view(-1, 2)
    crop_size = crop_size.view(-1, 2)

    assert crop_start.shape == crop_size.shape, f"mismatch dims crop {crop_size.ndim} and start {crop_start.ndim} "
    target_size = resolve_size(x.shape, target_size)

    # single crop and resize
    if not expand_dims:
        return resize_tensor(x[:, :, crop_start[0,0]:crop_size[0, 0],
                               crop_start[0, 1]:crop_size[0, 1]],
                             target_size, interpolation, align_corners)

    out = torch.zeros(*x.shape[:2], *target_size, dtype=x.dtype, device=x.device)

    indices = get_mgrid(dims, indices=True)
    for i, index in enumerate(indices):
        nc = slicer(x.shape, expand_dims, index)
        out[nc] = resize_tensor(x[nc][:, :, crop_start[i, 0]:crop_size[i, 0],
                                      crop_start[i, 1]:crop_size[i, 1]],
                                target_size, interpolation, align_corners)
    return out

# def crop_targets()
