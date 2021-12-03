"""@xvdp
bounding boxes and paths annotation conversions
"""
import enum
from typing import Union, Optional
import torch
from torch import Tensor
import numpy as np
from koreto import Col
from .. import config

_OptTensorable = Union[None, int, float, list, tuple, Tensor]
# pylint: disable=no-member

###
#
#   2d Transformations
#
# TODO: perspective, lens distort, 3d
# as a sparse tensor, or we convert back and forth on every transform
# TODO conversions of rotated boxes to pos_pos, and pos_offset
def transform_target(target: Union[Tensor, list, tuple],
                     matrix: Optional[Tensor] = None,
                     pos: _OptTensorable = None,
                     scale: _OptTensorable = None,
                     angle: _OptTensorable = None,
                     repos: _OptTensorable = None,
                     mode: Union[str, config.BoxMode] = "xpath") -> Union[Tensor, list]:
    """ Affine transform of 2d points by matrix
    Args
        target      (tensor | tuple of tensors) shape (batch, ..., n, 2)
            2d positions, shape ([batch, ... , 2])
        matrix      (tensor), shape (batch, 1, 3, 3) if matrix, ignore, pos, scale, repos, angle

        pos, scale, repos       (int, tuple, list, tensor)
            if int | float:  value is replicated for all dimensions all batches
            if tensor | list | tuple, & ndim == 1: replicated for all batches
            if ndim > 1: ndim has to equal batch_size
        .. expected in same order as targets, x,y or y,x

        angle:                  (int, tuple, list, tensor) in radians
            if tuple | list | tensor, ndim = 1 and len(angle) in (1, batch_size)
        .. counterclockwise (if mode passed and not xpath)

        mode: io mode of the targets: TODO: rotations for pos_pos, pos_offset, center_offset_angle
    """
    if all(t is None for t in (matrix, pos, scale, angle, repos)):
        return target

    _valid = (list, tuple, Tensor)
    assert isinstance(target, _valid), f"expected types {_valid}, got {type(target)}"
    mode = target_mode_str(mode)

    if matrix is None:
        matrix = make_transform2d(len(target), pos=pos, scale=scale, angle=angle,
                                  repos=repos, mode=mode, dtype=None)

    assert len(matrix) in (1, len(target))
    if isinstance(target, (list, tuple)):
        out  = []
        for i, tgt in enumerate(target):
            mat = matrix if len(matrix) == 1 else matrix[i:i+1]
            out.append(transform_target(tgt, mat))
        return out

    return from_transformable((matrix @ to_transformable(target)), shape=target.shape)

def _get_dtype(dtype: Union[None, str, torch.dtype] = None) -> torch.dtype:
    """resolved dtype str, dtype or default
    Args
        dtype   torch.dtype, str, None
    """
    _is = isinstance
    if _is(dtype, torch.dtype):
        return dtype
    if _is(dtype, str) and dtype in torch.__dict__ and _is(torch.__dict__[dtype], torch.dtype):
        return torch.__dict__[dtype]
    return torch.get_default_dtype()

def to_transformable(x: Tensor, homogeneous: bool = True) -> Tensor:
    """Outputs a pointset affine transformable -> (batch, n, dims, 1)
    Unsqueezeing last dimension, and either expanding dim 1 or
    compacting all dims 1 to -2
    Args
        x           tensor w shapes: (dims) | (batch, dims) | (batch, n, ..., m, dims)
        homogeneous bool [True] expand from dims to h coords
    Examples:
    in: [batch, n, m, dims] -> [batch, n*m, dims, 1]
    in: [dims] -> [1, 1, dims, 1]
    dims = x.shape(-1) or  x.shape(-1) + 1 if homogeneous
    """
    if x.ndim == 1:
        x = x.view(1,-1)
    if homogeneous:
        x = torch.cat((x, torch.ones((*x.shape[:-1], 1))), -1)
    return x.view(x.shape[0], -1, x.shape[-1], 1)

def from_transformable(x: Tensor, shape: tuple, homogeneous: bool = True) -> Tensor:
    """ restores tensor to previous shape
    removing
    """
    x = x.squeeze(-1) # remove dimension added for transformation
    if homogeneous:
        x = torch.split(x, (2,1), -1)[0]
    return x.view(shape)

def make_transform2d(batch_size: int = 1,
                     pos: _OptTensorable = None,
                     scale: _OptTensorable = None,
                     angle: _OptTensorable = None,
                     repos: _OptTensorable = None,
                     mode: Union[str, config.BoxMode] = 'xpath',
                     dtype: Union[None, str, torch.dtype] = None) -> Tensor:
    """Returns homogeneous transform shape (batch_size, 1, 3, 3), if no input -> Identity
    Transform order: translation (pos), scale, rotation (angle), translation (repos)
    Args:
        pos, scale, repos       (int, tuple, list, tensor)
            if int | float:  value is replicated for all dimensions all batches
            if tensor | list | tuple, & ndim == 1: replicated for all batches
            if ndim > 1: ndim has to equal batch_size
        angle:                  (int, tuple, list, tensor) in radians
            if tuple | list | tensor, ndim = 1 and len(angle) in (1, batch_size)
        dtype: if None, torch.get_default_dtype
    """
    dtype = _get_dtype(dtype)
    out = torch.stack([torch.eye(3, dtype=dtype)] * batch_size)
    if pos is not None:
        _assert_batch(batch_size, pos, ndim=2, msg="make_transform2d(pos= ")
        out[..., -1] = expand_vec2d(pos, dims=2, dtype=dtype, homogeneous=True)[..., -1]
    if scale is not None:
        _assert_batch(batch_size, scale, ndim=2, msg="make_transform2d(scale= ")
        out = out*expand_vec2d(scale, dims=2, dtype=dtype, homogeneous=True)
    if angle is not None:
        _assert_batch(batch_size, angle, ndim=1, msg="make_transform2d(angle= ")
        out = make_rotation(angle, mode) @ out
    if repos is not None:
        _assert_batch(batch_size, pos, ndim=2, msg="make_transform2d(repos= ")
        out[..., -1] += expand_vec2d(repos, dims=2, dtype=dtype, homogeneous=True)[..., -1]
    return out.unsqueeze(1)

def _assert_batch(batch_size, data, ndim=2, msg=""):
    """ could fail if data passed as list"""
    if torch.is_tensor(data):
        if data.ndim == ndim:
            assert data.shape[0] in (1, batch_size), f"{msg}mismatch, batch size got {data.shape[0]}, expeced {batch_size}"
        assert data.ndim <= ndim, f"too many dimensions {data.ndim} expected <= {ndim}"

def make_rotation(val, mode='xpath', homogeneous=True):
    """ make batch of counterclockwise rotations
    returns tensor shape (n, 3,3) where n = len(val)
    """
    if isinstance(val, (int, float)) or (torch.is_tensor(val) and val.ndim == 0):
        val = [val]
    cos = torch.cos(torch.as_tensor(val))
    sin = (1 if mode[0] == 'y' else -1)*torch.sin(torch.as_tensor(val))
    out = []
    for i, _ in enumerate(cos):
        out.append(_make_rotation2d(cos[i], sin[i], homogeneous))
    return torch.stack(out)

def _make_rotation2d(cos, sin, homogeneous=True):
    """ Make single 2d rotation
    """
    if homogeneous:
        return Tensor([[cos, -1*sin, 0], [sin, cos, 0], [0, 0, 1]])
    return Tensor([[cos, -1*sin], [sin, cos]])

def expand_vec2d(x: Union[int, float, list, tuple, Tensor],
                 dims: int = 2,
                 dtype: Union[None, str, torch.dtype] = None,
                 homogeneous: bool = True) -> Tensor:
    """ Expands to tensor sized [n, dims + int(homogeneous), 1]
    where n is 1 or len(x) if x.ndim > 1
    Args:
        x       (int, float, list, tuple, tensor) with max shape [n, dims]  
        dims    (int [2]) number of vector dimensions
        dtype   (str, torch.dtype, [None]) None: default_dtype
        homogeneous (bool [True]) - add extra dim 1
    # TODO will fail if dims or ndims > that expeceted output
    """
    ndim = 2 # constant, [batch,data]
    dtype = _get_dtype(dtype)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x *= dims
    elif isinstance(x, (int,float)) or (torch.is_tensor(x) and (x.ndim == 0 or x.ndim==1 and len(x) == 1)):
        x = [x] * dims
    x = torch.as_tensor(x, dtype=dtype)
    if x.ndim < ndim: # use unsqueeze recursive instead
        x = x.unsqueeze(0)
    if homogeneous:
        shape = list(x.shape)
        shape[-1] = 1
        x = torch.cat((x, torch.ones(shape, dtype=x.dtype)), -1)
    return x.unsqueeze(-1)

###
#
#
def crop_targets(x: Union[Tensor, list, tuple],
                 crop_start: torch.Tensor,
                 crop_end: torch.Tensor,
                 mode: str = 'ypath') -> tuple:
    """ crops tensor targets or list of targets to start and end
    Args
        x           tensor, shape (N,P,K,2):
                        N: crop sets
                        P: number of paths per set
                        K: number of points per path
                    list or tensors: on Item batches, paths are not merged but are lists

        crop_start  tensor, shape (N, 2) same order as x (yx or xy)
        crop_size   tensor, shape (N, 2)
        mode        assumes mode and crop_start/ crop_end are ordered similarly, ij or ji

    """
    # crops not implemented for pos_offset
    _mode = mode
    if mode in ('yxhw',  'xyhw'):
        x, _mode = pos_offset__pos_pos_mode(x, mode)

    # set of targets outside image
    drop = []

    # targets tensor
    if torch.is_tensor(x):
        shapes = f"{tuple(crop_start.shape)}: {tuple(crop_end.shape)}: {tuple(x.shape)}"
        assert len(crop_start) in (1, len(x)) and len(crop_end) in (1, len(x)), f"shape mismatch {shapes}"
        assert crop_start.shape[-1] == crop_end.shape[-1] == x.shape[-1], f"shape mismatch {shapes}"

        # make crops broadcastable to x -> (N,1,1,2)
        xview = lambda a, b: (b.shape[0], *[1]*(a.ndim - 2), b.shape[-1])
        crop_start = crop_start.view(xview(x, crop_start)).to(dtype=x.dtype, device=x.device)
        crop_end = crop_end.view(xview(x, crop_end)).to(dtype=x.dtype, device=x.device)
  
        # crop
        x = torch.clamp(x.sub(crop_start), torch.zeros(2), crop_end.sub(crop_start))

        # collect dropped targets
        for i, box in enumerate(x.view(-1, *x.shape[-2:])):
            if torch.all(box[:, 0]==box[0, 0]) or torch.all(box[:, 1]==box[0, 1]):
                drop.append(i)

        return x, drop

    # list of targets tensors: if batch is passed
    out = []
    for i, _x in enumerate(x):
        _crop_start = crop_start[i:i+1] if len(crop_start) == len(x) else crop_start
        _crop_end = crop_end[i:i+1] if len(crop_end) == len(x) else crop_end
        _x, _drop = crop_targets(_x, _crop_start, _crop_end, _mode)
        out.append(_x)
        drop.append(_drop)

    if mode in ('yxhw',  'xyhw'):
        out, _mode = pos_pos__pos_offset_mode(out, _mode)

    return out, drop

#
# BOXES/  PATHS / 2dtargets format conversions
#
def to_xpath(x: Union[Tensor, np.ndarray, list], mode: Union[str, config.BoxMode] = 'xyhw'):
    mode = target_mode_str(mode)
    if mode in ('xyhwa', 'yxhwa'):
        raise NotImplementedError("TODO, impplement rotation offset conversions")

    if mode[0] != 'x':
        x, mode = ij__ji_mode(x, mode)
    if mode == 'xywh':
        x, mode = pos_offset__pos_pos_mode(x, mode)
    if mode == 'xyxy':
        x, mode = pos_pos__path_mode(x, mode)
    return x

def from_xpath(x: Union[Tensor, np.ndarray, list], mode: Union[str, config.BoxMode] = 'xyhw'):
    """ TODO rotate boxes on centers on conversion
    """
    mode = target_mode_str(mode)
    if mode in ('xyhwa', 'yxhwa'):
        raise NotImplementedError("TODO, impplement rotation offset conversions")
    inmode = "xpath"
    if mode[0] !='x':
        x, inmode = ij__ji_mode(x, inmode)
    if inmode == mode:
        return x
    # to pos pos: no good, should rotate boxes on centers
    x, inmode = path__pos_pos_mode(x, inmode)
    if inmode == mode:
        return x
    return pos_pos__pos_offset(x)


def ij__ji(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """row major - col major
    requires last dimension to be size 2
    """
    if isinstance(x, list):
        return [ij__ji(_x) for _x in x]

    assert x.shape[-1] in (2, 5), f"expected shapes (...,2,2) or (...,5), found {x.shape}"

    if isinstance(x, np.ndarray):
        return _ij__ji_np(x)

    if x.shape[-1] == 2:
        x =  x.flip(x.ndim-1)
    else:
        xa, xb = x.split((4, 1), -1)
        xa = xa.reshape(-1, 4).reshape(-1, 2, 2).flip(-1).reshape(-1, 4).reshape(xa.shape)
        x = torch.cat((xa, xb), -1)
    return x

def _ij__ji_np(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 2:
        x = x[..., ::-1]
    else:
        x[..., :4] =  x[...,:4].reshape(-1, 4).reshape(-1, 2, 2)[..., ::-1].reshape(-1, 4).reshape(x[..., :4].shape)
    return np.ascontiguousarray(x)

def _clone(x: Union[Tensor, np.ndarray], inplace: bool = True):
    if inplace:
        return x
    if torch.is_tensor(x):
        return x.clone().detach()
    return x.copy()

def _shape_assert(x: Tensor, pos: int = 1, shape: tuple = (2,), msg: str = ""):
    msg = f"{msg} expected pos vector in format (...,*{shape}) got {tuple(x.shape)}"
    assert x.shape[-pos:] == shape, msg

def pos_pos__pos_offset(x: Union[Tensor, np.ndarray, list], inplace: bool = False) -> Union[Tensor, np.ndarray, list]:
    """ convert positions box to position offset box
    io tensor shape (..., 2, 2)
    """
    if isinstance(x, list):
        return [pos_pos__pos_offset(_x, inplace) for _x in x]

    _shape_assert(x, 2, (2, 2), "pos_pos__pos_offset")
    x = _clone(x, inplace)
    x[..., 1, :] -= (x[..., 0, :])
    return x

def pos_offset__pos_pos(x: Union[Tensor, np.ndarray, list], inplace: bool = False) -> Union[Tensor, np.ndarray, list]:
    """ convert position offset box to positions box
    io tensor shape (..., 2, 2)
    """
    if isinstance(x, list):
        return [pos_offset__pos_pos(_x, inplace) for _x in x]

    _shape_assert(x, 2, (2, 2), "pos_offset__pos_pos")
    x = _clone(x, inplace)
    x[..., 1, :] += (x[..., 0, :])
    return x

def pos_pos__path(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ convert positions box to positions path
    in tensor shape  (..., 2, 2)
    out tensor shape (..., 4, 2); out_numel = in_numel*2
    """
    if isinstance(x, list):
        return [pos_pos__path(_x) for _x in x]

    _shape_assert(x, 2, (2, 2), "pos_pos__path")
    paths = x.shape[:-2]
    order = torch.tensor([0, 1, 2, 1, 2, 3, 0, 3], dtype=torch.int64)
    return x.reshape(*paths, 4)[..., order].reshape(*paths[:2], 4, 2)

def path__pos_pos(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ convert positions path to positions box
    out tensor shape (..., 4, 2)
    in tensor shape  (..., 2, 2); out_numel = in_numel/2
    .. does not account for rotations if there are any
    """
    if isinstance(x, list):
        return [path__pos_pos(_x) for _x in x]

    _shape_assert(x, 2, (4, 2), "pos_pos__path")
    if torch.is_tensor(x):
        return torch.stack((x.min(-2)[0], x.max(-2)[0]), -2)
    return np.stack((x.min(-2), x.max(-2)), -2)

def pos_pos__center_offset_angle(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ convert positions box to center, half offset, angle
        in tensor shape  (..., 2, 2)
        out tensor shape  (..., 5); out_numel = in_numel*5/4
    """
    if isinstance(x, list):
        return [pos_pos__center_offset_angle(_x) for _x in x]

    return _center_offset__center_offset_angle(_pos_pos__center_offset(x))

def _pos_pos__center_offset(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ convert positions box to center, half offset
        io tensor shape  (..., 2, 2)
    """
    if isinstance(x, list):
        return [_pos_pos__center_offset(_x) for _x in x]

    _shape_assert(x, 2, (2, 2), "pos_pos__center_offset")
    if torch.is_tensor(x):
        return torch.stack((x.mean(-2), x[...,1,:] - x.mean(-2)), -2)
    return np.stack((x.mean(-2), x[...,1,:] - x.mean(-2)), -2)

def _center_offset__center_offset_angle(x: Union[Tensor, np.ndarray, list], angle: float = 0.0) -> Union[Tensor, np.ndarray, list]:
    """ add angle to center offset
        in tensor shape  (..., 2, 2)
        out tensor shape (..., 5); out_numel = in_numel*5/4
    """
    if isinstance(x, list):
        return [_center_offset__center_offset_angle(_x, angle) for _x in x]

    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    if torch.is_tensor(x):
        return torch.cat((x.view(*x.shape[:-2], 4), torch.full((*x.shape[:-2], 1), angle,
                                                               dtype=x.dtype, device=x.device)), -1)
    return np.concatenate((x.reshape(*x.shape[:-2], 4),
                           np.full((*x.shape[:-2], 1), angle, dtype=x.dtype)), -1)

def center_offset_angle__pos_pos(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ convert center, half offset, angle to positions box
        in tensor shape   (..., 5); out_numel = in_numel*5/4
        out tensor shape  (..., 2, 2)
    """
    if isinstance(x, list):
        return [center_offset_angle__pos_pos(_x) for _x in x]

    return _center_offset_angle__center_offset(_center_offset__pos_pos(x))

def _center_offset_angle__center_offset(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ remove angle from center offset
    Arg
        in tensor shape  (..., 5)
        out tensor shape  (..., 2, 2); out_numel = in_numel*4/5
    """
    if isinstance(x, list):
        return [_center_offset_angle__center_offset(_x) for _x in x]

    _shape_assert(x, 1, (5,), "center_offset__pos_pos")
    return x[...,:4].reshape(*x.shape[:-1], 2, 2)

def _center_offset__pos_pos(x: Union[Tensor, np.ndarray, list]) -> Union[Tensor, np.ndarray, list]:
    """ center, half offset box to positions
        io tensor shape  (..., 2, 2)
    """
    if isinstance(x, list):
        return [_center_offset__pos_pos(_x) for _x in x]

    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    if torch.is_tensor(x):
        return torch.stack((x[..., 0, :] - x[..., 1, :], x[..., 0, :] + x[..., 1, :]), -2)
    return np.stack((x[..., 0, :] - x[..., 1, :], x[..., 0, :] + x[..., 1, :]), -2)

# TODO: compute rotations from path
# def center_offset_angle__path(x: Tensor) -> Tensor:
#     """ remove angle from center offset
#     Arg
#         in tensor shape  (..., 5)
#         out tensor shape  (..., 2, 2); out_numel = in_numel*4/5
#     Arg
#     """
#     _shape_assert(x, 1, (5,), "center_offset__pos_pos")

#     angles = x[...,4]
#     rot = torch.stack([torch.cos(angles), -torch.sin(angles),
#                        torch.sin(angles), torch.cos(angles)], -1).view(*angles.shape, 2,2)
#     x = pos_pos__path(_center_offset__pos_pos(_center_offset_angle__center_offset(x)))
    # TODO compute rotations. T(R(T-1(x)))

###
#
# BOXES conversion with Mode wrappers
#
def mode_ij__ji(mode: str) -> str:
    """flips y-x in mode string"""
    if 'path' in mode:
        axis = ['x', 'y']
        return axis[1-axis.index(mode[0])] + mode[1:]
    _mode = list(mode)
    _mode[0], _mode[1], _mode[2], _mode[3] = _mode[1], _mode[0], _mode[3], _mode[2]
    return ''.join(_mode)

def ij__ji_mode(x: Union[Tensor, np.ndarray, list], mode: str) -> tuple:
    """ outputs ij__ji, fliped mode"""
    return ij__ji(x), mode_ij__ji(mode)

def pos_pos__pos_offset_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xyxy', inplace: bool = False) -> tuple:
    """ ouptuts pos_pos__pos_offset(x) and new mode str"""
    axis = ['x', 'y']
    offset = ['w', 'h']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], offset[i], offset[1-i]])
    return pos_pos__pos_offset(x, inplace), mode

def pos_offset__pos_pos_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xyhw', inplace: bool = False) -> tuple:
    """ ouptuts pos_offset__pos_pos(x) and new mode str"""
    axis = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], axis[i], axis[1-i]])
    return pos_offset__pos_pos(x, inplace), mode

def pos_pos__path_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xyxy') -> tuple:
    """ ouptuts pos_pos__path(x) and new mode str"""
    return pos_pos__path(x), f"{mode[0]}path"

def path__pos_pos_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xpath') -> tuple:
    """ ouptuts path__pos_pos(x) and new mode str"""
    axis = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], axis[i], axis[1-i]])
    return path__pos_pos(x), mode

def pos_pos__center_offset_angle_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xyxy') -> tuple:
    """ ouptuts pos_pos__center_offset_angle(x) and new mode str"""
    axis = ['x', 'y']
    offset = ['w', 'h']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], offset[i], offset[1-i], 'a'])
    return pos_pos__center_offset_angle(x), mode

def center_offset_angle__pos_pos_mode(x: Union[Tensor, np.ndarray, list], mode: str = 'xyhwa') -> tuple:
    """ ouptuts center_offset_angle__pos_pos(x) and new mode str"""
    axis = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.joinmode_ij__ji([axis[i], axis[1-i], axis[i], axis[1-i]])
    return center_offset_angle__pos_pos(x), mode

###
#
# 2d annotation targets affine transforms
#
def target_mode_str(mode: Union[str, config.BoxMode]) -> str:
    """ returns boxmode as string or string list
    validates mode argument is in config.BoxMode Enum
    """
    mode = mode if isinstance(mode, str) else mode.name
    _msg = f"{Col.YB} mode {mode} not defined in BoxMode {config.BoxMode.__members__}{Col.AU}"
    assert mode in config.BoxMode.__members__, _msg
    return mode
