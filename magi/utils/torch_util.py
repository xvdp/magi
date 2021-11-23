"""@xvdp
logging:
    logtensor(x, msg)
    logndarray(x, msg)

torch checks:
    grad:   warn_grad_cloning()
    dtype:  torch_dtype()
    device: torch_device()
    tensor: check_contiguous(x)

tensor reshaping:
    is_broadcastable(x, other)
    get_broadcastable(x, other)
        -> match trailing dims of 'other' with singletons making them multipliable
        e.g: shapes xshape (1,3) & other.shape (1,3,45,45)_ ->  x.shape (1,3,1,1):  other*x -> OK
        -> if x single dim, extend it on 'axis' (default 1, channels)
        e.g. x = [0.3,0.3,1.], other.shape (1,3,45,45) ->  x.shape (1,3,1,1)

    broadcast_tensors(*tensors, align=-1) -> default (align=-1) expand training
        like torch.broadcast_tensors() with stack order option
        eg. tensors [1,3], [1,1,45,45], align = -1 -> [1,3,45,45]
        eg. tensors [1,3], [1,30,3,1],  align = 1  -> [1,30,3,3]

extension:
    tensor_apply(x, func, hold_axis)

"""
from collections import namedtuple
from typing import Union, Optional
import logging
import torch
import numpy as np
from koreto import Col, sround
from .. import config

_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)
_vector = (np.ndarray, torch.Tensor)

# pylint: disable=no-member

###
# logging values, shape, dtype, device
#
def logtensor(x, msg=""):
    """ log tensor properties, for debug"""
    assert isinstance(x, torch.Tensor)
    _dtype = f" {x.dtype.__repr__().split('.')[-1]}"
    _min = f" min {sround(x.min().item(), 2)}"
    _max = f" max {sround(x.max().item(), 2)}"
    _mean = f" mean {sround(x.mean().item(), 2)}"
    print(f"{msg}{tuple(x.shape)} {_dtype} {x.device.type} grad {x.requires_grad},{_mean},{_min},{_max}")

def logndarray(x, msg=""):
    """ log tensor properties, for debug"""
    assert isinstance(x, np.ndarray)
    _min = f" min {sround(x.min().item(), 2)}"
    _max = f" max {sround(x.max().item(), 2)}"
    _mean = f" mean {sround(x.mean().item(), 2)}"
    print(f"{msg}{tuple(x.shape)},{_mean},{_min},{_max}")


def warn_grad_cloning(for_display: bool, grad: bool, in_config: bool=True, verbose: bool=True) -> bool:
    """ for_display operations cannot be used with backprop
        if grad: for_display: False
        Args:
            for_display
            grad        check against, overrides for_display
            in_config   sets global config.FOR_DISPLAY
            verbose
    """
    if grad and for_display:
        if verbose:
            logging.warning(f"{Col.YB}tensors with grad, setting {Col.BB}config.FOR_DISPLAY=False{Col.YB} to prevent broken gradients !{Col.AU}")
        for_display = False

    if for_display is not None and in_config:
        config.set_for_display(for_display)
    return for_display


def slicer(shape: tuple,
           dims: Union[torch.Tensor, tuple, list],
           vals: Union[torch.Tensor, tuple, list]) -> tuple:
    """ simple slice builder returning value:value+1
    on dimensions
    Args
        shape   tuple, tensor or initial dimensions of tensor
        dims    tuple, dims over which values act
        vals    tuple len(values) == len(dimensions), 0 <= values[i] < shape[i]
    Example
        # given x: tensor of shape (4,3,20,20)
        >>> _slice = slicer((4,3,20), dims=(0,2), vals=(2,5))
        >>> _slice # -> (2:3, :, 5:6)
        >>> x[_slice].shape
        torch.Size([1, 3, 1, 20])
    """
    assert len(dims) == len(vals), f" {len(dims)} dims == {len(vals)} vals"
    assert max(dims) < len(shape), f"cannot slice dim {max(dims)} over shape {shape}"
    _len = min(max(dims) + 1, len(shape))
    fro = [0] * _len
    _to = list(shape)[:_len]
    for i, dim in enumerate(dims):
        fro[dim] = vals[i]
        _to[dim] = vals[i] + 1
    return tuple(slice(i, j) for (i, j) in zip(fro, _to))

def squeeze(x: torch.Tensor, side: int = 0, min_ndim: int = 1) -> torch.Tensor:
    """ recursive squeeze till no more squeezing is possible
    Args
        x           tensor
        side        int [0] | -1, left or right
        min_ndim    int [1] | 0, return min ndim
    """
    while x.shape[side] == 1 and x.ndim > min_ndim:
        x = x.squeeze(side)
    return x

def check_contiguous(x: torch.Tensor, verbose: bool = False, msg: str = "") -> torch.Tensor:
    """ Check if thensor is contiguous, if not, force it
        permutations result in non contiguous tensors which lead to cache misses
        Args
            tensor  (torch tensor)
            verbose (bool) default False
    """
    if x.is_contiguous():
        return x
    if verbose:
        logging.warning(msg+"tensor was not continuous, making continuous")
    return x.contiguous()

###
#
# dtype resolution
#
def is_torch_strdtype(dtype: str) -> bool:
    """ torch dtypes from torch.__dict__"""
    _torch_dtypes = ['uint8', 'int8', 'int16', 'short', 'int32', 'int', 'int64', 'long',
                     'float16', 'half', 'float32', 'float', 'float64', 'double', 'complex32',
                     'complex64', 'cfloat', 'complex128', 'cdouble', 'bool', 'qint8', 'quint8',
                     'qint32', 'bfloat16', 'quint4x2']
    return dtype in _torch_dtypes


def torch_dtype(dtype: Union[str, torch.dtype, list, tuple],
                force_default: bool = False,
                fault_tolerant: bool = True) -> Union[torch.dtype, list]:
    """ Returns torch.dtype, None, torch.det_default_dtype() or list of dtypes
    Args:
        dtype   (str, list, tuple, torch.dtype)
        force_default   (bool [False]) if True and dtype is None, sets to torch.get_default_dtype()
        fault_tolerant  (bool [True]) return invalid dtypes as None

    Examples
    >>> torch_dtype('float32') -> torch.float32
    >>> torch_dtype(['float32', 'int']) -> [torch.float32, torch.int64]
    >>> torch_dtype(None, force_default=True) -> torch.float32 ( if default_dtype: torchfloat32)
    >>> torch_dtype(None) -> None
    """
    if dtype is None and force_default:
        dtype = torch.get_default_dtype()
    elif isinstance(dtype, torch.dtype) or dtype is None:
        pass
    elif isinstance(dtype, str):
        if dtype not in torch.__dict__:
            if fault_tolerant:
                dtype = None
            else:
                raise TypeError(f"dtype {dtype} not recognized")
        else:
            dtype = torch.__dict__[dtype]
    elif isinstance(dtype, (list, tuple)):
        dtype = [torch_dtype(d) for d in dtype]
    else:
        if fault_tolerant:
            dtype = None
        else:
            raise NotImplementedError(f"dtype {dtype} not recognized")
    return dtype

def str_dtype(dtype: Union[str, torch.dtype, np.dtype]) -> str:
    """ convert torch or numpy dtype to str
    """
    if isinstance(dtype, str) and is_torch_strdtype(dtype):
        return dtype

    if isinstance(dtype, torch.dtype):
        return dtype.__repr__().split('.')[-1]

    if isinstance(dtype, np.dtype): # np dtypes
       return dtype.name
    if hasattr(dtype, '__name__'):
        return (dtype.__name__)

    return ValueError(f"dtype( {dtype} does not correspond to a valid dtype")

###
#
# device resolution
# TODO validate
#
def torch_device(device):
    """TODO clean up device and dtype handling
    this is going to bomb somewhere
    """
    return config.get_valid_device(device)

    # if isinstance(device, str):
    #     if not ':' in device and device == 'cuda':
    #         device += ':0'
    #     device = torch.device(device)
    # return device

###
#
# extend tensor methods
#
def tensor_apply_vals(x: torch.Tensor,
                      func: str="min",
                      hold_axis: Union[None, int, list] = None,
                      axis: Union[None, int, list] = None,   # unused
                      keepdims: bool = False) -> Union[torch.Tensor, tuple]:
    """ same as tensor apply, returning always tensor, no tuples """
    x = tensor_apply(x, func, hold_axis, axis, keepdims)
    if isinstance(x, tuple):
        x = x[0]
    return x

def tensor_apply(x: torch.Tensor,
                 func: str="min",
                 hold_axis: Union[None, int, list] = None,
                 axis: Union[None, int, list] = None,   # unused
                 keepdims: bool = False) -> Union[torch.Tensor, tuple]:
    """ applies tensor method over all axes except hold_axis
    Similar but opposite to the use of 'axis' or 'dim' in torch
    # TODO reverse args, use axis instead of hold_axis

    Args:
        x           (tensor)
        func        (str 'min') tensor methods with 'axis' arg
            e.g. min, max, mean, sum, prod
        hold_axis   (int | list [0])   computes func over all other axes
        keepdims    (bool [False])

    Examples:
        x = torch.linspace(-1, 2, 28).view(2, 2, 7)
        y = tensor_apply(x, "max", hold_axis=1, keepdims=True)
        y.shape #-> torch.Size([1, 2, 1])
        y       #-> tensor([[[1.2222], [2.0000]]])

        y = tensor_apply(x, "max", hold_axis=0, keepdims=True)
        y.shape #-> torch.Size([2, 1, 1])
        y       #-> tensor([[[-1.0000]], [[ 0.5556]]])
    test/test_utils.py
    """
    if hold_axis is None or (isinstance(hold_axis, (list, tuple)) and len(hold_axis) == 0):
        review = [1 for i in range(x.ndim)]
        x = getattr(x, func)()
    else:
        if isinstance(hold_axis, int):
            x, view, review, axis = _apply_single_axis(x, hold_axis)
        else:
            hold_axis = set([ax%x.ndim for ax in hold_axis])
            if len(hold_axis) == x.ndim:
                return x
            x, view, review, axis = _apply_multiple_axis(x, hold_axis)
        x = getattr(x.view(*view), func)(axis=axis)

    if keepdims: # rebuild tensor dims
        if isinstance(x, (tuple, list)):
            x = list(x)
            for i, _x in enumerate(x):
                x[i] = _x.view(*review)
            if func in ("min", "max"): #not identical but similar
                x =  namedtuple(typename=func, field_names=["values", "indices"], defaults=x)()
        else:
            x = x.view(*review)
    return x

def _apply_multiple_axis(x: torch.Tensor, hold_axis: Union[list, tuple]) -> tuple:
    """resolves tensor reshaping if multiple axis are held"""
    axis = -1
    view = (*x.shape[:len(hold_axis)], -1)

    # hold axis are in the front
    if all(ax < len(hold_axis) for ax in hold_axis):
        review = [*x.shape[:len(hold_axis)], *[1]*(x.ndim - len(hold_axis))]

    # hold axis are in the back
    elif all(ax >  x.ndim -len(hold_axis) - 1 for ax in hold_axis):
        view = (-1, *x.shape[-len(hold_axis):])
        review = [*[1]*(x.ndim - len(hold_axis)), *x.shape[-len(hold_axis):]]
        axis = 0
    # are mixed
    else:
        review = list(x.shape)
        view = [x.shape[i] for i in hold_axis] + [-1]
        x = x.permute(*hold_axis, *[ex for ex in range(x.ndim) if ex not in hold_axis]).contiguous()
        for i in range(x.ndim):
            if i not in hold_axis:
                review[i] = 1
    return x, view, review, axis

def _apply_single_axis(x: torch.Tensor, hold_axis: int) -> tuple:
    """ resolves tensor reshaping if single axis is held
    """
    hold_axis = hold_axis%x.ndim
    axis = 0 if hold_axis == x.ndim - 1 else -1
    view = (-1, x.shape[hold_axis]) if hold_axis == x.ndim - 1 else (x.shape[hold_axis], -1)
    review = (*[1]*hold_axis, x.shape[hold_axis],  *[1]*(x.ndim - 1 - hold_axis))

    # if not end dimension, permute to first dim
    if hold_axis not in (0, x.ndim-1):
        x = x.swapaxes(0, hold_axis).contiguous()
    return x, view, review, axis


###
#
# tensor (shape, dtype, device) resolution
#
def is_broadcastable(x: Union[_torchable], y: torch.Tensor) -> bool:
    """ True if x * other -> OK
    """
    if torch.is_tensor(x) and x.ndim == y.ndim and x.dtype == y.dtype and x.device == y.device:
        return all(x.shape[i] in (1, y.shape[i]) for i in range(x.ndim))
    return False

def get_broadcastable(x: Union[_torchable], other: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """ Convert 'x' to tensor with same dtype, device, ndim as tensor
    with x.shape[i] == tensor.shape[i] or axis i reduced to 1 by mean

    Args
        x       (list, tuple, int, float, ndarray torch.Tensor)
            if ndim == 1 and len() > 1 and len() == len(tensor.shape[axis]

        tensor  torch.Tensor) tensor to match
            if tensor or ndarray, axis size equal x or reduced to mean

        axis    (int [min(1 | tensor.ndim-1)]) only used broadcasting x with ndim==1
                default: 1, i.e. channels axis

    Examples
        >>> y = get_broadcastable([2,3], torch.randn([1,3,10,10]), axis=1)
        >>> y, y.shape
        (tensor([[[[2.5000]]]]), torch.Size([1, 1, 1, 1]))
        >>> y * torch.randn([1,3,10,10]) -> OK


        >>> y = get_broadcastable([2,3,4], torch.randn([1,3,10,10]), axis=1)
        >>> y, y.shape
        (tensor([[[[2.]], [[3.]], [[4.]]]]), torch.Size([1, 3, 1, 1]))
        >>> y * torch.randn([1,3,10,10]) -> OK
    """
    if is_broadcastable(x, other):
        return x

    assert isinstance(x, _torchable), f"invalid type {type(x)}"
    assert other.is_floating_point(), f"only floating_point tensors, found {other.dtype}"
    with torch.no_grad():

        axis = min(axis, other.ndim -1)
        shape = [1]*other.ndim

        if isinstance(x, (int, float)):
            shape[axis] = 1
        elif all(isinstance(i, (int,float)) for i in x):
            shape[axis] = len(x)
        else:
            x = torch.as_tensor(x).to(dtype=other.dtype)

        if isinstance(x, _vector):
            if x.ndim > other.ndim and all(n==1 for n in x.shape[other.ndim:]):
                x = x.view(*[d for d in x.shape[:other.ndim]])

            _msg = "Cannot reduce non empty trainling dimensions "
            assert x.ndim <= other.ndim, f"{Col.YB}{_msg}{x.shape}[:{other.ndim}]{Col.AU}"
            x = torch.as_tensor(x, dtype=other.dtype)
            if axis >= x.ndim:
                x = x.view(*[1]*(axis - x.ndim + 1), *x.shape)

            reduce = [i for i in range(x.ndim) if x.shape[i] not in (1, other.shape[i])]
            if reduce:
                x = x.mean(axis=reduce, keepdims=True)
            shape[:x.ndim] = x.shape

        x = torch.as_tensor(x).view(*shape).to(dtype=other.dtype)

        # reduce mean all dims not equal to 'other' or 1
        _bad_dims = [i for i in range(len(x.shape)) if x.shape[i] not in (1, other.shape[i])]
        if _bad_dims:
            x = x.mean(_bad_dims, keepdims=True)
        return x.to(device=other.device)

###
#
# extend broadcast_tensors to align data front or back
#
def to_tensor(x: Union[_torchable], dtype: Optional[torch.dtype] = None,
              device: Optional[str] = None) -> torch.Tensor:
    """Converts float, int, list, tuple ndarray, or 0d tensor to >=1d tensor
    x   int, float, np.ndarray, torch.tensor
    dtype   optional # if None and x not tensor -> default
    device  optional
    """
    if not isinstance(x, (torch.Tensor, np.ndarray)) and dtype is None:
        dtype = torch.get_default_dtype()

    _to = {k:v for k,v in {'dtype':dtype, 'device':device}.items() if v is not None}
    if isinstance(x, (int, float)) or (torch.is_tensor(x) and x.ndim == 0):
        return torch.as_tensor([x], **_to)
    return torch.as_tensor(x, **_to)

def squeeze_trailing(x: Union[_torchable], dtype: Optional[torch.dtype] = None,
                     device: Optional[str] = None, min_ndim: int = 1) -> torch.Tensor:
    """Remvoes all trailing signletons, stopping at ndim = 1
        e.g [1,3,1,1] -> [1,3]
    """
    x = to_tensor(x, dtype, device)
    while len(x.shape) and x.ndim > min_ndim and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x

def broadcast_tensors(*tensors, align: int = -1, dtype: torch.dtype = None, device: str = None) -> tuple:
    """ torch.broadcast_tensors() with stack order option
    Args
        *tensors    int, float, tuple, list, ndarray, tensor
        dtype       torch.dtype [None]
        device       torch.dtype [None]
        align       int [-1] sigletons are added to back
            if 1 they get added to front (torch.broadcast_tensors)

    Broadcasts tensors aligning to first dimension by default ( align=-1 )
        eg. shapes  [1], [2,4], [1,1,4] -> [2,4,4]
    or ( align=1 ) like torch.broadcast_tensors, aligning to last dimension, eg.
        eg. shapes [1], [2,4], [1,1,4] -> [1,2,4]

    Converts all *tensors (int,float,list,tuple,array,tensor) to tensors of same dtype and device
    Examples: test/test_tensor_util.py
    """
    _no_tensor = not any(torch.is_tensor(x) for x in tensors)
    tensors = [to_tensor(x, dtype, device) for x in tensors]

    # ensure are all in same device and dtype
    _DTYPE_default = torch.float32
    _DEVICE_default = 'cpu'

    dtype = _DTYPE_default if any(x.dtype != tensors[0].dtype for x in tensors) or _no_tensor else None
    device = _DEVICE_default if any(x.device != tensors[0].device for x in tensors) else None
    _to = {k:v for k,v in {'dtype': dtype, 'device': device}.items() if v is not None}

    # broadcast shapes, aligned -1: front, 1: back
    shapes = [list(x.shape)[::align] for x in tensors]
    shape = torch.broadcast_shapes(*shapes)[::align]
    ndim = max([x.ndim for x in tensors])

    for i, x in enumerate(tensors):
        view = list(x.shape)
        if align == -1 and len(view) < ndim:
            view += [1]*(ndim - len(view))
        tensors[i] = torch.broadcast_to(x.to(**_to).view(view), shape)
    return tuple(tensors)
