"""@xvdp
standard torch checks
"""
from typing import Union
import logging
import torch
import numpy as np
from koreto import Col, sround
from .. import config

_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)
_vector = (np.ndarray, torch.Tensor)

# pylint: disable=no-member

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


def check_contiguous(tensor, verbose=False, msg=""):
    """ Check if thensor is contiguous, if not, force it
        permutations result in non contiguous tensors which lead to cache misses
        Args
            tensor  (torch tensor)
            verbose (bool) default False
    """
    if tensor.is_contiguous():
        return tensor
    if verbose:
        print(msg+"tensor was not continuous, making continuous")
    return tensor.contiguous()

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

def ensure_broadcastable(x: Union[_torchable], tensor: torch.Tensor) -> torch.Tensor:
    """ match ndim, device, dtype of tensor"""
    if not torch.is_tensor(x) or x.ndim != tensor.ndim or x.dtype != tensor.dtype or x.device != tensor.device:
        x = reduce_to(x, tensor)
    return x


def reduce_to(x: Union[_torchable], tensor:torch.Tensor, axis: int=1) -> torch.Tensor:
    """ Convert 'x' to tensor with same dtype, device, ndim as tensor
    with x.shape[i] == tensor.shape[i] or axis i reduced to 1 by mean
    
    Args
        x       (list, tuple, int, float, ndarray torch.Tensor)
        tensor  torch.Tensor) tensor to match
            if tensor or ndarray, axis size equal x or reduced to mean

        axis    (int [min(1 | tensor.ndim-1)]) only necessary for broadcasing where x.ndim <= axis
    """
    assert isinstance(x, _torchable), f"invalid type {type(x)}"
    assert tensor.is_floating_point(), f"only floating_point tensors, found {tensor.dtype}"
    with torch.no_grad():

        axis = min(axis, tensor.ndim -1)
        shape = [1]*tensor.ndim

        if isinstance(x, _vector):
            assert x.ndim <= tensor.ndim, f"reductions not defined extra dim {x.ndim} > {tensor.ndim}"
            x = torch.as_tensor(x, dtype=tensor.dtype)
            if axis >= x.ndim:
                x = x.view(*[1]*(axis - x.ndim + 1), *x.shape)

            reduce = [i for i in range(x.ndim) if x.shape[i] not in (1, tensor.shape[i])]
            if reduce:
                x = x.mean(axis=reduce, keepdims=True)
            shape[:x.ndim] = x.shape

        elif isinstance(x, (int, float)):
            shape[axis] = 1
        else:
            assert all(isinstance(i, (int,float)) for i in x), f"only 1d lists permitted, found {x}"
            shape[axis] = len(x)

        return torch.as_tensor(x).view(*shape).to(dtype=tensor.dtype, device=tensor.device)

def is_torch_strdtype(dtype: str) -> bool:
    """ torch dtypes from torch.__dict__"""
    _torch_dtypes = ['uint8', 'int8', 'int16', 'short', 'int32', 'int', 'int64', 'long',
                     'float16', 'half', 'float32', 'float', 'float64', 'double', 'complex32',
                     'complex64', 'cfloat', 'complex128', 'cdouble', 'bool', 'qint8', 'quint8',
                     'qint32', 'bfloat16', 'quint4x2']
    return dtype in _torch_dtypes


def torch_dtype(dtype: Union[str, torch.dtype, list, tuple], force_default: bool=False, fault_tolerant: bool=True) -> Union[torch.dtype, list]:
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

def dtype_as_str(dtype: Union[str, torch.dtype, np.dtype])-> str:
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
