"""@xvdp
standard torch checks and and ducktape
"""
from typing import Union, Any
import torch

    # pylint: disable=no-member
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
