"""@ xvdp

globals for namespace magi, cacheable
    DTYPE
    INPLACE
    BOXMODE
    DATAPATHS
"""
from typing import Union, Any
import os.path as osp
from enum import Enum
import pickle
import torch
# pylint: disable=no-member
# pylint: disable=not-callable



#
# global DTYPE
#
DTYPE = torch.get_default_dtype()

def set_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """ default dtype can only be floating point
    set default dtype and save to a global that could be cached if needed
    """
    global DTYPE
    if isinstance(dtype, str) and dtype in torch.__dict__:
        dtype = torch.__dict__[dtype]
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        DTYPE = dtype
        torch.set_default_dtype(DTYPE)
    else:
        DTYPE = torch.get_default_dtype()
    return DTYPE

def dype_str(dtype):
    return dtype.__repr__().split(".")[1]
#
# global INPLACE
#
INPLACE = True
def set_inplace(inplace):
    global INPLACE
    if inplace is not None:
        INPLACE = bool(inplace)
    return INPLACE
#
# global BOXMODE
#
class BoxMode(Enum):
    """ default annotation for tensorlists"""
    xywh = "xywh"
    yxhw = "yxhw"
    xywha = "xywha"
    yxhwa = "yxhwa"
    xyxy = "xyxy"
    yxyx = "yxyx"
    ypath = "ypath"
    xpath = "xpath"

BOXMODE = BoxMode.xywh
def set_boxmode(mode, msg=""):
    global BOXMODE
    if mode is not None:
        assert mode in BoxMode.__dict__, "target mode '%s' not implemented: %s"%(mode, msg)
        BOXMODE = BoxMode(mode)
    return BOXMODE

#
# global DATAPATHS
#
DATAPATHS = {}
def add_datapath(name, path):
    if osp.isdir(path):
        DATAPATHS[name] = path
def get_datapath(name):
    if name in DATAPATHS:
        return DATAPATHS[name]
    return None

# saved or loaded in config
_DATA = ["DTYPE", "INPLACE", "BOXMODE", "DATAPATHS"]

#
# store and load
#
def save_globals(fname="~/.magi.config"):
    """
    """
    fname = osp.abspath(osp.expanduser(fname))
    dic = globals()
    obj = {d:dic[d] for d in _DATA}

    with open(fname, "wb") as _fi:
        pickle.dump(obj, _fi)

    print(" saved binary config:", fname)

def load_globals(fname="~/.magi.config"):
    """ load saved defautls
    """
    fname =  osp.abspath(osp.expanduser(fname))
    if not osp.isfile(fname):
        print("'%s' not found, cannot load config, using current config:"%fname)
        return None

    dic = globals()

    with open(fname, "rb") as _fi:
        obj = pickle.load(_fi)

    for o in obj:
        if o in dic:
            if o == "DTYPE" and  obj[o] != DTYPE:
                set_dtype(DTYPE)
            dic[o] = obj[o]
        else:
            print("object '%s' not found in config"%o)
