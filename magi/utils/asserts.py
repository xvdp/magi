"""@xvdp
standard asserts, wrap loops
"""
from functools import wraps
from typing import Union, Any
import numpy as np
import torch
from koreto import Col

# def assert_dims(expand_dims, allowed=(0,1), msg="") -> None:
#     _bad =  [d for d in expand_dims if d not in allowed]
#     assert not _bad, f"{Col.YB}{msg} expand_dims {expand_dims} barred, only {allowed} permitted.{Col.AU}"

def assert_equal(*items, msg=" ") -> None:
    _items_match = all([items[0] == items[i] for i in range(len(items))])
    assert _items_match, f"{Col.YB}{msg} must be equal, got {items}{Col.AU}"

def assert_in(elem: Any, allowed: Union[list, tuple, torch.Tensor, np.ndarray, dict], msg=" ") -> None:
    if not isinstance(elem, (list, tuple, torch.Tensor, np.ndarray, dict)):
        elem = [elem]
    _bad =  [d for d in elem if d not in allowed]
    assert not _bad, f"{Col.YB}{msg} {elem} barred, only {allowed} permitted.{Col.AU}"


def loop_on_list(func):
    """ wrapper to loop
    Example:
    >>> @loop_on_list
    >>> def logtensor(x, indent="", msg=""):
            print(f"{indent}{msg}{tuple(x.shape)}, {x.dtype}, {x.device}, {x.max()}, {x.min()}")
    >>> logtensor([torch.ones(12,2), torch.zeros(2, device='cuda')], msg="my tensor", indent="")
     my tensor (12, 2), torch.float32, cpu, 1.0, 1.0
     my tensor (2,), torch.float32, cuda:0, 0.0, 0.0
    """
    @wraps(func)
    def loopy(*args, **kwargs):
        if isinstance(args[0], (list, tuple)):
            args = list(args)
            arg = args.pop(0)
            if "indent" in kwargs:
                kwargs["indent"] += " "
            for a in arg:
                func(a, *args, **kwargs)
        else:
            func(*args, *kwargs)
    return loopy
