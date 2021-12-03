"""@xvdp
functional for sizing transforms
"""
from typing import Union, Optional
import torch
from torch import nn
from koreto import Col
from .functional_base import transform, transform_profile, get_sample_like, Tensorish, TensorItem
from ..utils import get_mgrid, slicer, crop_targets, transform_target
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

    # expand crop sizes to batch, and expand dims 0, 1 if
    # sample ratios
    crop_start, crop_end = _get_crop_info(data, ratio)
    size = _parse_size_arg(data[0].shape, size)

    _transform = transform_profile if profile else transform
    data = _transform(data=data, func=squeeze_crop_tensor, for_display=for_display,
                      kind_keys=['data_2d'], crop_start=crop_start, crop_end=crop_end,
                      target_size=size, interpolation=interpolation, align_corners=align_corners)

    if data.get_indices(kind='pos_2d'):
        data = _transform(data=data, func=squeeze_crop_targets, for_display=False,
                        kind_keys=['pos_2d'], crop_start=crop_start, crop_end=crop_end,
                        target_size=size)
    return data


def squeeze_crop_tensor(x: torch.Tensor,
                        crop_start: torch.Tensor,
                        crop_end: torch.Tensor,
                        target_size: Union[int, tuple, None] = None,
                        interpolation: str = 'linear',
                        align_corners: bool = True,
                        mode: str = 'NCHW') -> torch.Tensor:
    """
    Args
        x           tensor
        crop_start  (tensor) shape N,C,1,1,2 # where N and C are 1 dim is not expanded
        crop_end    (tensor) same shape as crop_start
    """
    expand_dims = [i for i in range(len(crop_start.shape[:-1])) if crop_start.shape[i] > 1]
    if expand_dims:
        _msg = f"{Col.YB}Sizing transforms only allow expanding dims 0 | 1{expand_dims}{Col.AU}"
        assert max(expand_dims) <= 1, _msg

    dims = [crop_start.shape[i] for i in expand_dims]
    crop_start = crop_start.view(-1, 2)
    crop_end = crop_end.view(-1, 2)

    assert crop_start.shape == crop_end.shape, f"mismatch dims crop {crop_end.ndim} and start {crop_start.ndim} "
    target_size = _parse_size_arg(x.shape, target_size)

    # single crop and resize
    if not expand_dims:
        return resize_tensor(x[:, :, crop_start[0, 0]:crop_end[0, 0],
                               crop_start[0, 1]:crop_end[0, 1]],
                             target_size, interpolation, align_corners)

    # stack multpile crops into tensor slices
    # get_mgrid + slicer returns available permutations and slices tensor, batch, channel
    out = torch.zeros(*x.shape[:2], *target_size, dtype=x.dtype, device=x.device)
    indices = get_mgrid(dims, indices=True)
    for i, index in enumerate(indices):
        _nc = slicer(x.shape, expand_dims, index)
        out[_nc] = resize_tensor(x[_nc][:, :, crop_start[i, 0]:crop_end[i, 0],
                                        crop_start[i, 1]:crop_end[i, 1]],
                                 target_size, interpolation, align_corners)
    return out



def squeeze_crop_targets(x: Union[torch.Tensor, list, tuple],
                        crop_start: torch.Tensor,
                        crop_end: torch.Tensor,
                        target_size: Union[int, tuple, None] = None,
                        mode: str = 'yxyx') -> torch.Tensor:
    """
    Args
        x           tensor # targets, expeced in xpath mode
        crop_start  (tensor) shape N,C,1,1,2 # where N and C are 1 dim is not expanded
        crop_size   (tensor) same shapea as crop_start
    """
    # 
    # convert to positionts 'y' to match tensor
    _mode = mode
    if mode[0] == 'x':
        x, _mode = ij__ji_mode(x, mode)
    if 'w' in mode:
        x, _mode = pos_offset__pos_pos_mode(x, _mode)


    # TODO cleanup broadcasting; this is a mess
    # resolve scales
    target_size = torch.as_tensor(target_size, dtype=crop_end.dtype)
    target_size, crop_end = torch.broadcast_tensors(target_size, crop_end)
    scale = (target_size/(crop_end - crop_start))

    expanded_channels = crop_start.shape[1]

    # propagate
    if len(crop_start) > 1: # batch_size > 1 and 0 in expand_dims
        scale = [s.view(-1,2) for s in torch.split(scale, [1]*len(scale), 0)]

    elif len(x) > 1: # batch_size > 1, expand_dims in None, 1
        crop_start = torch.cat([crop_start]*len(x))
        crop_end = torch.cat([crop_end]*len(x))
        scale = [scale.view(-1,2)]*len(x)

    if torch.is_tensor(scale):
        scale = scale.view(-1,2)

    # crop
    drop = []
    for i, _x in enumerate(x):
        if expanded_channels > 1:
            _x = torch.cat([x[i]]*expanded_channels)
        x[i], _drop =  crop_targets(_x, crop_start[i], crop_end[i], _mode)
        drop.append(_drop)

    # scale
    if isinstance(scale, (list, tuple)):
        for i, _ in enumerate(scale):
            x[i] = transform_target(x[i], scale=scale[i]).view(-1, 1, *x[i].shape[2:])
            if drop[i]:
                x[i] = torch.cat([x[i][j] for j in range(len(x[i])) if j not in drop[i] ]) 
    else:
        x = transform_target(x, scale=scale).view(-1, 1, *x.shape[2:])
        if drop:
            x = torch.cat([x[j] for j in range(len(x)) if j not in drop[i] ]) 

    if mode[0] != _mode[0]:
        x, _mode = ij__ji_mode(x, _mode)
    if mode != _mode and 'w' in mode:
        x, _mode = pos_pos__pos_offset_mode(x, _mode)
    assert _mode == mode, f"failed to convert to original mode"

    return x

def log_tensor(x, indent="", msg=""):
    if torch.is_tensor(x):
        print(f"{indent}{msg}{tuple(x.shape)}")
    else:
        print(f"{indent}{msg}: list, len: {len(x)}")
        for e in x:
            log_tensor(e, indent+" ")

def _get_crop_info(x: TensorItem, ratio: Tensorish) -> tuple:
    """ returns crop start and end as in shape(*x.shape, 2)

    expand_dims=None            torch.Size([1, 1, 1, 1, 2]) torch.Size([1, 1, 1, 1, 2])
    expand_dims=0               torch.Size([2, 1, 1, 1, 2]) torch.Size([2, 1, 1, 1, 2])
    expand_dims=(0, 1)          torch.Size([2, 3, 1, 1, 2]) torch.Size([2, 3, 1, 1, 2])
    expand_dims=(1,)            torch.Size([1, 3, 1, 1, 2]) torch.Size([1, 3, 1, 1, 2])
    expand_dims=(0, 1, 2)       torch.Size([2, 3, 582, 1, 2]) torch.Size([2, 3, 582, 1, 2])
    expand_dims=(0, 2)          torch.Size([2, 1, 582, 1, 2]) torch.Size([2, 1, 582, 1, 2])
    expand_dims=(0, 1, 2, 3)    torch.Size([2, 3, 582, 1024, 2]) torch.Size([2, 3, 582, 1024, 2])
    # expand_dims = [i for i in range(len(start.shape[:-1])) if start.shape[i] > 1]
    """
    # get tensor from item
    x = x if torch.is_tensor(x) else x[0]
    # random sampler sampler
    ratio = get_sample_like(ratio, like=x).unsqueeze(-1)
    img_size = torch.tensor(x.shape[2:]).long().unsqueeze(0) # H, W
    anisotropy = torch.sub(*x.shape[2:]) # H - W
    start = torch.clamp(torch.tensor([anisotropy, -anisotropy]),
                        0, img_size.max()).mul(ratio/2.).long()
    end = img_size.sub(start)
    return start, end

def _parse_size_arg(input_shape: tuple, target_size: Union[int, tuple, None] = None) -> tuple:
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
    size = _parse_size_arg(x.shape, size)
    if x.shape[2:] == size:
        return x

    interpolation = _resolve_interpolation_mode(x.ndim, interpolation)
    return nn.functional.interpolate(x, size=size, mode=interpolation,
                                     align_corners=align_corners).clamp(x.min(), x.max())


# def crop_targets()
