"""@xvdp

Globals for namespace magi, cacheable to:
global MAGI_CACHE_DIR
>>> config/get_cache_dir_path()
'~/.cache/magi'

with resolution order
    os.environ['MAGI_CACHE_DIR']
    osp.join(os.environ['HOME'], '.cache', 'magi')
    osp.join(os.environ['USERPROFILE'], '.cache', 'magi')
    osp.join(tempfile.gettempdir(), '.cache', 'magi')

Subset of globals can be manually saved or loaded
    DTYPE: str      # set by transform.Open(dtype=<>, force_global=True)
        alias to get and set torch_default_dtype
    # TODO cleanup dtype and device, set and get torch default instead of global
    
    FOR_DISPLAY: bool = False # if true clones augment operations
    BOXMODE: Enum
    DEBUG: bool # TODO remove

>>> load_globals()
>>> save_globals()


Dataset Paths:
When datasets.Dataset_M is leveraged, dataset paths are cached and loaded from file:
    osp.join(MAGI_CACHE_DIR, "datasets.yml")
Written to cache on  <MyDataset>(Dataset_M)(data_root=<path>)
Loaded from cache on <MyDataset>(Dataset_M)(data_root=None)
On every load or write, missing paths are culled, paths are sorted by disk read speed.
    # TODO backup old paths 

"""
from typing import  Union
import logging
import os
import os.path as osp
from enum import Enum
from timeit import timeit
import tempfile
import torch
from koreto import ObjDict, Col
# pylint: disable=no-member
# pylint: disable=not-callable

DEBUG = False

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES = ["cpu", "cuda", *(f"cuda:{i}" for i in range(torch.cuda.device_count()))]

def device_valid(device: Union[torch.device, str]) -> bool:
    """ check if device is valid (cpu or gpu)
    Args
        device  (str, torch.device)
    does not check
    xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, mlc, xla, vulkan, meta, hpu
    """
    device = torch.device(device)
    out = False
    if device.type == "cpu":
        out = True
    elif device.type == "cuda" and torch.cuda.is_available:
        if device.index is None or device.index < torch.cuda.device_count():
            out = True

    # this does not check for XLA or other devides
    return out

def get_valid_device(device: Union[None, torch.device, str] = 'cpu') -> torch.device:
    """ returns highest available device
    e.g
    if device == 'cuda:3' but only 3 gpus available
            returns device(type='cuda', index=2)
    if cuda is not available
            returns device(type='cpu')
    Args
        device  (str, torch.device)
    does not check
    xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, mlc, xla, vulkan, meta, hpu
    """

    device = torch.device(device = device if device is not None else 'cpu')
    if device.type == "cuda":
        if not torch.cuda.is_available:
            logging.warning("'cuda' not Available, defaulting to 'cpu'")
        else:
            devices =  torch.cuda.device_count()
            if device.index is None or device.index < devices:
                return device

            logging.warning(f"cuda:{device.index}, not available, defaulting tocuda:{devices-1}")
            return torch.device("cuda", index=devices-1)

    return torch.device("cpu")

#
# global DTYPE: str
#
DTYPE = torch.get_default_dtype().__repr__().split(".")[1]

def resolve_dtype(dtype: Union[None, str, torch.dtype] = None, force_global: bool = False) -> str:
    """ default dtype can only be floating point
    set default dtype and save to a global that could be cached if needed

    >>> resolve_dtype(None or invalid) -> returns DTYPE
    >>> resolve_dtype(dtype in dtype.is_floating_point) -> returns dtype input
    >>> resolve_dtype(dtype in dtype.is_floating_point) -> sets and returns DTYPE

    """
    global DTYPE
    _torch_dtype = None

    if dtype is not None:
        if isinstance(dtype, str) and dtype in torch.__dict__:
            _torch_dtype = torch.__dict__[dtype]
    
        elif isinstance(dtype, torch.dtype):
            _torch_dtype = dtype
            dtype = dtype.__repr__().split(".")[1]
        else:
            logging.warning(f"config.set_type(dtype), invalid dtype {type(dtype)}, expect (str, torch.dtype, None)")
            dtype = None

        if _torch_dtype is not None and not _torch_dtype.is_floating_point:
            logging.warning("config.set_type(dtype), invalid dtype, expected floating_point dtypes")
            _torch_dtype = None

    if _torch_dtype is None:
        dtype = DTYPE
    elif force_global:
        torch.set_default_dtype(_torch_dtype)
        DTYPE = torch.get_default_dtype().__repr__().split(".")[1]

    return dtype

FOR_DISPLAY = False
def set_for_display(for_display: bool = True) -> bool:
    global FOR_DISPLAY
    if for_display is not None:
        FOR_DISPLAY = bool(for_display)
    return FOR_DISPLAY
#
# global BOXMODE: Enum
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


###
#
# Main cache dir
#
MAGI_CACHE_DIR = None

def set_cache_dir(path: str) -> None:
    global MAGI_CACHE_DIR
    MAGI_CACHE_DIR = path

def get_cache_dir_path(*paths: str) -> str:
    """ resolves and makes cache path if none found
    """
    global MAGI_CACHE_DIR
    if MAGI_CACHE_DIR is None:
        if 'MAGI_CACHE_DIR' in os.environ:
            MAGI_CACHE_DIR = os.environ['MAGI_CACHE_DIR']
        elif 'HOME' in os.environ:
            MAGI_CACHE_DIR = osp.join(os.environ['HOME'], '.cache', 'magi')
        elif 'USERPROFILE' in os.environ:
            MAGI_CACHE_DIR = osp.join(os.environ['USERPROFILE'], '.cache', 'magi')
        else: #/tmp
            MAGI_CACHE_DIR = osp.join(tempfile.gettempdir(), '.cache', 'magi')

    out = osp.join(MAGI_CACHE_DIR, *paths)
    os.makedirs(out, exist_ok=True)
    return out

#
# store and load
#
def save_globals(*paths: str) -> str:
    """
    """
    name = osp.join(get_cache_dir_path(*paths), "magi_config.yml")
    dic = globals()
    obj = ObjDict({d:dic[d] for d in ["DEBUG", "DTYPE", "FOR_DISPLAY", "BOXMODE"]})
    obj.BOXMODE = BOXMODE.name
    obj.to_yaml(name)
    return name

def load_globals(*paths: str) -> None:
    """ load saved defautls
    """
    name = osp.join(get_cache_dir_path(*paths), "magi_config.yml")
    if not osp.isfile(name):
        print("'%s' not found, cannot load config, using current config:"%name)
        return None

    dic = globals()
    _load = ObjDict()
    _load.from_yaml(name)
    if 'BOXMODE' in _load:
        _load.BOXMODE = BoxMode(_load.BOXMODE )
    dic.update(**_load)
    print(f"Loaded globals{_load}")

#
# dataset registry
#
def write_dataset_cache(datasets: ObjDict, *paths: str):
    cachename = osp.join(get_cache_dir_path(*paths), "datasets.yml")
    datasets.to_yaml(cachename)
    print(f"    dataset cache written to {Col.BB}'{cachename}'{Col.AU}")

def write_dataset_path(name: str, path: str, *paths: str):
    datasets = load_dataset_cache(*paths)
    if name not in datasets:
        datasets[name] = []

    if path not in datasets[name]:
        datasets[name].append(path)
        print(f"{Col.B}Write {name} data_root {Col.GB}'{path}'{Col.AU}")
        write_dataset_cache(datasets, *paths)

def load_dataset_cache(*paths: str) -> ObjDict:
    """ load and clean up paths.
    """
    _dirty = False
    datasets = ObjDict()
    cachename = osp.join(get_cache_dir_path(*paths), "datasets.yml")
    if osp.isfile(cachename):
        datasets.from_yaml(cachename)
        for _, name in enumerate(datasets):
            _len = len(datasets[name])
            datasets[name] = [pth for pth in datasets[name] if osp.isdir(pth)]
            _dirty = len(datasets[name]) != _len
            if len(datasets[name]) == 0:
                del datasets[name]
    if _dirty:
        write_dataset_cache(datasets, *paths)
    return datasets

def load_dataset_path(name: str, *paths: str) -> str:
    """ get cached path, if more than one, return minimum
    """
    path = None
    datasets = load_dataset_cache(*paths)
    if name in datasets:
        if len(datasets[name]) == 1:  # if more than 1, test access speed
            path = datasets[name][0]
        else:
            _time = [timeit(setup='import os', number=100, stmt=f"os.listdir('{p}')")
                     for p in datasets[name]]
            path = datasets[name][_time.index(min(_time))]
    return path
