"""@xvdp
bounding boxes and paths annotation conversions
"""
import torch
# pylint: disable=no-member


__all__ = ['ij_ji', 'pos_offset__pos_pos', 'path__pos_pos',
           'pos_pos__pos_offset', 'pos_pos__path', 'pos_pos__center_offset',
           'center_offset__pos_pos', 'center_offset__center_offset_angle', 'center_offset_angle__center_offset']

# TODO complete and validate center_offset_angle__path

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
