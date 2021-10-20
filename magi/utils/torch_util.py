"""@xvdp
standard torch checks
"""
from typing import Union
import torch
import numpy as np

# pylint: disable=no-member
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
