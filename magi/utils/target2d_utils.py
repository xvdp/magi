"""@xvdp
bounding boxes and paths annotation conversions
"""
from typing import Union, Optional
import torch
import numpy as np
from koreto import Col
from .. import config
# pylint: disable=no-member

###
#
# 2d annotation targets affine transforms
#
def target_mode_str(mode: Union[str, config.BoxMode]) -> str:
    """ validates mode argument is in config.BoxMode Enum
    returns as string
    """
    mode = mode if isinstance(mode, str) else mode.name
    _msg = f"{Col.YB} mode {mode} not defined in BoxMode {config.BoxMode.__members__}{Col.AU}"
    assert mode in config.BoxMode.__members__, _msg
    return mode

def translate_target(target: Union[torch.Tensor, list, tuple],
                     translation: torch.Tensor,
                     mode: Union[str, config.BoxMode] = "xywh",
                     sign: int = 1) -> Union[list, torch.Tensor]:
    """adds or subtracts from target
    """
    _valid = (list, tuple, torch.Tensor)
    assert isinstance(target, _valid), f"expected types {_valid}, got {type(target)}"
    mode = target_mode_str(mode)

    if isinstance(target, (list, tuple)):
        mode = mode if isinstance(mode, (list, tuple)) else [mode]*len(target)
        out  = []
        for i, tgt in enumerate(target):
            out.append(translate_target(tgt, translation, mode[i], sign))
        return out

    # it would make more sense to convert to yxyx and apply all
    if mode[0] == 'x':
        translation = translation.flip(-1)
    translation = translation.mul(sign)
    if mode in ('xywh', 'yxhw'):
        translation = torch.stack((translation, torch.zeros(2)))
    elif mode in ('xywha', 'yxhwa'):
        translation = torch.cat((translation, torch.zeros(3)))

    return target.add(translation)

def affine_target(target: Union[torch.Tensor, list, tuple],
                  matrix: torch.Tensor,
                  mode: Union[str, config.BoxMode] = "xywh"):

    return target
###
#
# transformations
#
def to_homogeneous(x: torch.Tensor):
    """ expands last dimension with 1s"""
    return torch.cat((x, torch.ones((*x.shape[:-1], 1))), -1)

class Matrix2d:
    def __init__(self,
                 n: int = 1,                        # batch size
                 t: torch.Tensor = torch.zeros(2),
                 r: Union[None, float, list, tuple, torch.Tensor] = None,
                 s: Optional[torch.Tensor] = None) -> None:
        """ r 
        
        """
        self.__ = torch.broadcast_to(torch.eye(3).unsqueeze(0), (n, 3, 3))
        if t is not None:
            assert t.ndim in (1,2) and t.shape[-1] == 2, f"expected tensor shaped (2), (1,2) or ({n},2) got {t.shape} "
            t = t if t.ndim == 2 else t.unsqueeze(0)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            for i in range(n):
                self.__[i, :-1, -1] = t[min(t.len -1, i)]
        # if r is not None:
        #     if isinstance(r, (int)) or (torch.is_tensor(r) and r.ndim < n) or isinstance(r, (list, tuple)) and len(r) < n:

    @staticmethod
    def make_rotation(val):
        cos = torch.cos(torch.as_tensor(val))
        sin = torch.sin(torch.as_tensor(val))
        return torch.tensor([[cos, -1*sin, 0.], [sin, cos, 0.], [0., 0., 0.]])


    #
# Position 2d Targets
#
def _shape_assert(x: torch.Tensor, pos: int = 1, shape: tuple = (2,), msg: str = ""):
    msg = f"{msg} expected pos vector in format (...,*{shape}) got {tuple(x.shape)}"
    assert x.shape[-pos:] == shape, msg


def ij__ji(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """row major - col major
    requires last dimension to be size 2
    """
    assert x.shape[-1] in (2, 5), f"expected shapes (...,2,2) or (...,5), found {x.shape}"
    
    if isinstance(x, np.ndarray):
        return _ij__ji_np(x)

    if x.shape[-1] == 2:
        x =  x.flip(x.ndim-1)
    else:
        xa, xb = x.split((4,1), -1)
        xa = xa.view(-1,4).view(-1,2,2).flip(-1).view(-1,4).view(xa.shape)
        x = torch.cat((xa, xb), -1)
    return x

def _ij__ji_np(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 2:
        x = x[..., ::-1]
    else:
        x[..., :4] =  x[...,:4].reshape(-1, 4).reshape(-1, 2, 2)[..., ::-1].reshape(-1, 4).reshape(x[..., :4].shape)
    return x

###
#
# BOXES
#
def _clone(x: Union[torch.Tensor, np.ndarray], inplace: bool = True):
    if inplace:
        return x
    if torch.is_tensor(x):
        return x.clone().detach()
    return x.copy()

def pos_pos__pos_offset(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ convert positions box to position offset box
    io tensor shape (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_pos__pos_offset")
    x = _clone(x, inplace)
    x[..., 1, :] = x[..., 1, :].sub(x[..., 0, :])
    return x

def pos_offset__pos_pos(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ convert position offset box to positions box
    io tensor shape (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_offset__pos_pos")
    x = _clone(x, inplace)
    x[..., 1, :] = x[..., 1, :].add(x[..., 0, :])
    return x

def pos_pos__path(x: torch.Tensor) -> torch.Tensor:
    """ convert positions box to positions path
    in tensor shape  (..., 2, 2)
    out tensor shape (..., 4, 2); out_numel = in_numel*2
    """
    _shape_assert(x, 2, (2, 2), "pos_pos__path")
    paths = x.shape[:-2]
    order = torch.tensor([0, 1, 2, 1, 2, 3, 0, 3])
    return x.view(*paths, 4)[..., order].view(*paths[:2], 4, 2)

def path__pos_pos(x: torch.Tensor) -> torch.Tensor:
    """ convert positions path to positions box
    out tensor shape (..., 4, 2)
    in tensor shape  (..., 2, 2); out_numel = in_numel/2
    .. does not account for rotations if there are any
    """
    _shape_assert(x, 2, (4, 2), "pos_pos__path")
    return torch.stack((x.min(-2)[0], x.max(-2)[0]), -2)

def pos_pos__center_offset_angle(x: torch.Tensor) -> torch.Tensor:
    """ convert positions box to center, half offset, angle
        in tensor shape  (..., 2, 2)
        out tensor shape  (..., 5); out_numel = in_numel*5/4
    """
    return _center_offset__center_offset_angle(_pos_pos__center_offset(x))

def _pos_pos__center_offset(x: torch.Tensor) -> torch.Tensor:
    """ convert positions box to center, half offset
        io tensor shape  (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_pos__center_offset")
    return torch.stack((x.mean(-2), x[...,1,:] - x.mean(-2)), -2)

def _center_offset__center_offset_angle(x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
    """ add angle to center offset
        in tensor shape  (..., 2, 2)
        out tensor shape (..., 5); out_numel = in_numel*5/4
    """
    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    return torch.cat((x.view(*x.shape[:-2], 4),
                      torch.full((*x.shape[:-2], 1), angle, dtype=x.dtype, device=x.device)), -1)

def center_offset_angle__pos_pos(x: torch.Tensor) -> torch.Tensor:
    """ convert center, half offset, angle to positions box
        in tensor shape   (..., 5); out_numel = in_numel*5/4
        out tensor shape  (..., 2, 2)
    """
    return _center_offset_angle__center_offset(_center_offset__pos_pos(x))

def _center_offset_angle__center_offset(x: torch.Tensor) -> torch.Tensor:
    """ remove angle from center offset
    Arg
        in tensor shape  (..., 5)
        out tensor shape  (..., 2, 2); out_numel = in_numel*4/5
    Arg
    """
    _shape_assert(x, 1, (5,), "center_offset__pos_pos")
    return x[...,:4].view(*x.shape[:-1], 2, 2)

def _center_offset__pos_pos(x: torch.Tensor) -> torch.Tensor:
    """ center, half offset box to positions
        io tensor shape  (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    return torch.stack((x[..., 0, :] - x[..., 1, :], x[..., 0, :] + x[..., 1, :]), -2)

# TODO: compute rotations from path
# def center_offset_angle__path(x: torch.Tensor) -> torch.Tensor:
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
# BOXES with Mode wrappers
def ij__ji_mode(x: Union[torch.Tensor, np.ndarray], mode: str) -> tuple:
    """ outputs ij__ji, fliped mode"""
    x = ij__ji(x)
    _mode = list(mode)
    _mode[0], _mode[1], _mode[2], _mode[3] = _mode[1], _mode[0], _mode[3], _mode[2]
    return x, ''.join(_mode)

def pos_pos__pos_offset_mode(x: torch.Tensor, mode: str = 'xyxy', inplace: bool = False) -> tuple:
    """ ouptuts pos_pos__pos_offset(x) and new mode str"""
    axis  = ['x', 'y']
    offset = ['w', 'h']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], offset[i], offset[1-i]])
    return pos_pos__pos_offset(x, inplace), mode

def pos_offset__pos_pos_mode(x: torch.Tensor, mode: str = 'xyhw', inplace: bool = False) -> tuple:
    """ ouptuts pos_offset__pos_pos(x) and new mode str"""
    axis  = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], axis[i], axis[1-i]])
    return pos_offset__pos_pos(x, inplace), mode

def pos_pos__path_mode(x: torch.Tensor, mode: str = 'xyxy') -> tuple:
    """ ouptuts pos_pos__path(x) and new mode str"""
    return pos_pos__path(x), f"{mode[0]}path"

def path__pos_pos_mode(x: torch.Tensor, mode: str = 'xpath') -> tuple:
    """ ouptuts path__pos_pos(x) and new mode str"""
    axis  = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], axis[i], axis[1-i]])
    return path__pos_pos(x), mode

def pos_pos__center_offset_angle_mode(x: torch.Tensor, mode: str = 'xyxy') -> tuple:
    """ ouptuts pos_pos__center_offset_angle(x) and new mode str"""
    axis  = ['x', 'y']
    offset = ['w', 'h']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], offset[i], offset[1-i], 'a'])
    return pos_pos__center_offset_angle(x), mode

def center_offset_angle__pos_pos_mode(x: torch.Tensor, mode: str = 'xyhwa') -> tuple:
    """ ouptuts center_offset_angle__pos_pos(x) and new mode str"""
    axis  = ['x', 'y']
    i = axis.index(mode[0])
    mode = ''.join([axis[i], axis[1-i], axis[i], axis[1-i]])
    return center_offset_angle__pos_pos(x), mode
    