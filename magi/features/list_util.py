"""@xvdp
"""
import random
from typing import Union
import numpy as np
import torch
List = Union[tuple, list, set]
_list = (tuple, list, set)

# pylint: disable=no-member
def is_int_iterable(*args) -> bool:
    """ Returns True if args is either int or a list, tuple or set containing ints
    """
    _iterable = (list, tuple, set)
    return all([isinstance(arg, int) or (isinstance(arg, _iterable) and all([isinstance(item, int)
            for item in arg])) for arg in args])

def list_flatten(*args, sort: bool=False, unique: bool=False) -> list:
    """ Flattens iterables to a list recursively
    non iterables, and iterables not in (ndarray, Tensor, list, set, tuple) are not flattened
    Args
        *args   (list, tuple, set, ndarray, Tensor, Any)
        sort    (bool [False])
        unique  (bool [False])  #list(set(*)) with no change in order
    """
    _arrays = (torch.Tensor, np.ndarray)
    _iterable = (list, tuple, set)
    out = []

    # -> flatten
    for arg in args:
        if isinstance(arg, _arrays):
            out.extend(arg.reshape(-1).tolist())
        elif isinstance(arg, (list, set, tuple)):
            if any([isinstance(x, _iterable) for x in arg]):
                arg = list_flatten(*arg)
            out.extend(arg)
        else:
            out.append(arg)

    # -> reduce
    if unique:
        out = [out[i] for i in sorted([out.index(o) for o in list(set(out))])]

    # -> order
    if sort:
        out.sort()
    return out

def list_modulo(indices: list, mod: Union[int, float], sort: bool=False) -> list:
    """ return i%size for all elements in a float/int list
    Args

        indices (list) of int or float
        mod     (int, float)
        sort    (bool [False])
    """
    assert all([isinstance(i, (int, float)) for i in indices])
    indices =  [i%mod for i in indices]
    if sort:
        indices.sort()
    return indices

def list_subset(io_list:list, subset: Union[List, int]=None, tolerate: bool=True) -> list:
    """ Returns a curated (by item or index) or random subset of a list
    Args
        in_list (list)  list to extract subset from
        subset  (iterable, int)
            int:    out -> random set of list elements size=subset
            list:   int list
        tolerate    (bool) allow for subset not tobe entirely contained by io_list
    """
    _oper = any if tolerate else all
    if subset is None:
        return io_list

    # subset: int -> random set of elements size=subset
    if isinstance(subset, int) and 0 < subset < len(io_list):
        indices = list(range(len(io_list)))
        random.shuffle(indices)
        subset = sorted(indices[:subset])

    # subset: indices ->
    if isinstance(subset, _list) and all([isinstance(s, int) for s in subset]) and _oper([s < len(io_list) for s in subset]):
        io_list = [io_list[i] for i in subset if i < len(io_list)]

    # -> list of elemnents ( fault tolerant )
    elif isinstance(subset, _list) and _oper([s in io_list for s in subset]):
        io_list = [c for c in subset if c in io_list]

    return io_list

def list_transpose(list2d: list, stride: int=1) -> list:
    """ Flattens 2d list transposing ij->ji with size step
    sub lists do not need to be same size
    Args
        list2d
        stride  (int [1])
    Example
    list_transpose([[a,b,c,D],[d,e,f],[g,h,i,I]], step=1) -> [a,d,g,b,e,g,c,f,i,D,I]
    """
    assert isinstance(list2d, list) and all([isinstance(item, list) for item in list2d]), "expected 2d list"
    _lens = [len(sublist) for sublist in list2d]
    _len = sum(_lens)
    _i = 0
    out = []
    while _len:
        for i, sublist in enumerate(list2d):
            if _lens[i] > 0:
                out += sublist[_i:_i+stride]
                _lens[i] -= stride
        _i += stride
        _len = sum(_lens)
    return out
