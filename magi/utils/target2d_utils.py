"""@xvdp
bounding boxes and paths annotation conversions
"""
from typing import Union, Optional
import torch
from .. import config
# pylint: disable=no-member

###
#
# 2d annotation targets affine transforms
#
def target_mode_str( mode: Union[str, config.BoxMode]):
    mode = mode if isinstance(mode, str) else mode.name
    assert mode in config.BoxMode.__members__, f"mode {mode} not defined in BoxMode {config.BoxMode.__members__}"
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

# class Matrix2d:
#     def __init__(self,
#                  n: int = 1,                        # batch size
#                  t: torch.Tensor = torch.zeros(2),
#                  r: Union[None, float, list, tuple, torch.Tensor] = None,
#                  s: Optional[torch.Tensor] = None) -> None:
#         """ r 
        
#         """
#         self.__ = torch.broadcast_to(torch.eye(3).unsqueeze(0), (n, 3, 3))
#         if t is not None:
#             assert t.ndim in (1,2) and t.shape[-1] == 2, f"expected tensor shaped (2), (1,2) or ({n},2) got {t.shape} "
#             t = t if t.ndim == 2 else t.unsqueeze(0)
#             if t.ndim == 1:
#                 t = t.unsqueeze(0)
#             for i in range(n):
#                 self.__[i, :-1, -1] = t[min(t.len -1, i)]
#         if r is not None:
#             if isinstance(r, (int)) or (torch.is_tensor(r) and r.ndim < n) or isinstance(r, (list, tuple)) and len(r) < n:

#     @staticmethod
#     def make_rotation(val):
#         cos = torch.cos(torch.as_tensor(val))
#         sin = torch.sin(torch.as_tensor(val))
#         return torch.tensor([[cos, -1*sin, 0.], [sin, cos, 0.], [0., 0., 0.]])


#
# Position 2d Targets
#
def _shape_assert(x: torch.Tensor, pos: int = 1, shape: tuple = (2,), msg: str = ""):
    msg = f"{msg} expected pos vector in format (...,*{shape}) got {tuple(x.shape)}"
    assert x.shape[-pos:] == shape, msg

def ij_ji(x: torch.Tensor) -> torch.Tensor:
    """row major - col major
    requires last dimension to be size 2
    """
    _shape_assert(x, 1, (2,), "ij_ji")
    return x.flip(x.ndim-1)
###
#
# BOXES
#
def pos_pos__pos_offset(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ convert positions box to position offset box
    io tensor shape (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_pos__pos_offset")
    if not inplace:
        x = x.clone().detach()
    x[..., 1, :] = x[..., 1, :].sub(x[..., 0, :])
    return x

def pos_offset__pos_pos(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ convert position offset box to positions box
    io tensor shape (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_offset__pos_pos")
    if not inplace:
        x = x.clone().detach()
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
    return x.view(*paths, 4)[..., order].view(*paths[:-2], 4, 2)

def path__pos_pos(x: torch.Tensor) -> torch.Tensor:
    """ convert positions path to positions box
    out tensor shape (..., 4, 2)
    in tensor shape  (..., 2, 2); out_numel = in_numel/2
    .. does not account for rotations if there are any
    """
    _shape_assert(x, 2, (4, 2), "pos_pos__path")
    return torch.stack((x.min(-2)[0], x.max(-2)[0]), -2)

def pos_pos__center_offset(x: torch.Tensor) -> torch.Tensor:
    """ convert positions box to center, half offset
        io tensor shape  (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "pos_pos__center_offset")
    return torch.stack((x.mean(-2), x[...,1,:] - x.mean(-2)), -2)

def center_offset__pos_pos(x: torch.Tensor) -> torch.Tensor:
    """ center, half offset box to positions
        io tensor shape  (..., 2, 2)
    """
    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    return torch.stack((x[..., 0, :] - x[..., 1, :], x[..., 0, :] + x[..., 1, :]), -2)

def center_offset__center_offset_angle(x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
    """ add angle to center offset
        in tensor shape  (..., 2, 2)
        out tensor shape  (..., 5); out_numel = in_numel*5/4
    """
    _shape_assert(x, 2, (2, 2), "center_offset__pos_pos")
    return torch.cat((x.view(*x.shape[:-2], 4),
                      torch.full((*x.shape[:-2], 1), angle, dtype=x.dtype, device=x.device)), -1)

def center_offset_angle__center_offset(x: torch.Tensor) -> torch.Tensor:
    """ remove angle from center offset
    Arg
        in tensor shape  (..., 5)
        out tensor shape  (..., 2, 2); out_numel = in_numel*4/5
    Arg
    """
    _shape_assert(x, 1, (5,), "center_offset__pos_pos")
    return x[...,:4].view(*x.shape[:-1], 2, 2)


def center_offset_angle__path(x: torch.Tensor) -> torch.Tensor:
    """ remove angle from center offset
    Arg
        in tensor shape  (..., 5)
        out tensor shape  (..., 2, 2); out_numel = in_numel*4/5
    Arg
    """
    _shape_assert(x, 1, (5,), "center_offset__pos_pos")

    angles = x[...,4]
    rot = torch.stack([torch.cos(angles), -torch.sin(angles),
                       torch.sin(angles), torch.cos(angles)], -1).view(*angles.shape, 2,2)
    x = pos_pos__path(center_offset__pos_pos(center_offset_angle__center_offset(x)))

    # TODO compute rotations. T(R(T-1(x)))
