"""@xvdp

Image opening utilities
"""
from typing import Union, Optional
from inspect import currentframe, getframeinfo
import warnings
import logging
from io import BytesIO
import os
import os.path as osp
import hashlib
from urllib.parse import urlparse
import requests
import numpy as np
import torch

import accimage # fastest for jpg->torch
from PIL import Image
import cv2

from koreto import Col
from koreto.utils import ObjDict

from . import check_contiguous
from .. import config

# torchvision transforms are Modules
_TT = torch.nn.modules.module.Module
_Vector = Union[torch.Tensor, np.ndarray]


# pylint: disable=no-member
# pylint: disable=not-callable
def _image_backends(backend: Optional[str] = None,
                    img: Optional[str] = None,
                    dtype: str = "float32",
                    channels: int = 3) -> list:
    """ resolve image handler backend
    """
    backends = ["accimage", "PIL", "opencv"]

    if backend is not None and  backend != backends[0]  and backend in backends:
        backends.remove(backend)
        backends.insert(0, backend)

    _backends = backends.copy()
    if img is not None and osp.splitext(img)[-1].lower() not in (".jpg", "jpeg") or dtype not in ("float16", "float32", "float64") or channels != 3:
        _backends.remove("accimage")
    return _backends

def open_acc(img: str,
             dtype: str = "float32",
             out_type: str = "numpy",
             channels: Optional[int] = None,
             transforms: Optional[_TT] = None) -> Optional[_Vector]:
    """ opens image using accimage
    faster for jpg which is stored same as torch tensors, CHW
    Args
        img         (str) image path
        dtype       (str ["float32"])
        out_type    (str ["numpy]) | "torch"
        channels    (int [None: same as stored]), | 1,3,4
        transforms  (torchvision.transforms [None])
    """
    try:
        # loads image as float32 CHW
        pic = accimage.Image(img)
        if transforms is not None:
            pic = transforms(pic)

        out = np.zeros([pic.channels, pic.height, pic.width], dtype="float32")
        pic.copyto(out)

        if dtype != "float32":
            out = to_np_dtype(dtype, out_type, out_type=out_type)
        if out_type == "numpy":
            out = out.transpose(1, 2, 0)
            return np_fix_channels(out, channels, fillvalue=None)
        return _to_torch(out)
    except:
        logging.debug("accimage could not resolve image")

    return None

def open_pil(img: str,
             dtype: str = "float32",
             out_type: str = "numpy",
             channels: Optional[int] = None,
             transforms: Optional[_TT] = None) -> Optional[_Vector]:
    """ opens image using PIL, returns np.ndarray or torch.tensor
    Args
        img         (str) image path
        dtype       (str ["float32"])
        out_type    (str ["numpy]) | "torch"
        channels    (int [None: same as stored]), | 1,3,4
        transforms  (torchvision.transforms [None])
    """
    # loads image as uint8 HWC
    try:
        out = Image.open(img)
        if transforms is not None:
            out = transforms(out)

        out = np.array(out, copy=False)
        out = to_np_dtype(out, dtype, out_type=out_type)
        out = np_fix_channels(out, channels, fillvalue=None)

        if out_type != "numpy":
            out = _to_torch(out, permute=(2, 0, 1))

        return out
    except:
        logging.debug("PIL could not resolve image")

    return None

def open_cv(img: str,
             dtype: str = "float32",
             out_type: str = "numpy",
             channels: Optional[int] = None,
             transforms: Optional[_TT] = None) -> Optional[_Vector]:
    """ opens image using opencv, returns np.ndarray or torch.tensor
    Args
        img         (str) image path
        dtype       (str ["float32"])
        out_type    (str ["numpy]) | "torch"
        channels    (int [None: same as stored]), | 1,3,4
        transforms  (torchvision.transforms [None])
    """
    try:
        out = cv2.cvtColor(cv2.imread(img), cv2.IMREAD_UNCHANGED)
        if transforms is not None:
            out = transforms(out)

        out = to_np_dtype(out, dtype, out_type=out_type)
        out = np_fix_channels(out, channels, fillvalue=None)

        if out_type != "numpy":
            out = _to_torch(out, permute=(2, 0, 1))
        return out
    except:
        logging.debug("opencv could not resolve image")

    return None

def _to_torch(data: np.ndarray, permute: Union[None, list, tuple] = None) -> torch.Tensor:
    """"""
    data = torch.from_numpy(data)
    if permute is not None:
        data = data.permute(*permute).contiguous()
    return data

def _assert_dtype(dtype: str, out_type: str) -> None:
    _valid_dypes = {'numpy':["uint8", "float16", "float32", "float64"],
                    'torch':["float16", "float32", "float64"]}[out_type]
    assert dtype in _valid_dypes, f"{Col.RB} only {_valid_dypes} supported, found: {dtype}"

def to_np_dtype(out: np.ndarray, dtype: str, out_type: str = "torch") -> np.ndarray:
    """ type conversion
    """
    _assert_dtype(dtype, out_type)
    _in_dtype = out.dtype.name
    assert _in_dtype == "uint8" or "float" in _in_dtype, f"input dtype invalid {_in_dtype}"

    if dtype == _in_dtype:
        return out

    if "float" in dtype:
        out = out.astype(dtype)

    # from uint to float
    if _in_dtype == 'uint8':
        out /= 255

    # from float to -> uint, ints
    if dtype == "uint8":
        out = (out*255).astype(dtype)
    elif dtype == "int64":
        out = (out*(2**63-1)).astype(dtype)
    elif dtype == "int32":
        out = (out*(2**31-1)).astype(dtype)
    return out

def get_cache_name(url: str, cache_dir: Optional[str] = None) -> str:
    """ caches names url images to local file.
    """
    if cache_dir is None:
        cache_dir = config.get_cache_dir_path('images')
        os.makedirs(cache_dir, exist_ok=True)

    fname = hashlib.md5(url.encode("utf-8")).hexdigest()
    _ext = osp.splitext(url.split("?")[0])[-1]
    if _ext.lower() in (".jpg", ".jpeg", ".png"):
        fname += _ext
    return osp.join(cache_dir, fname)

def cache_image(img: Image, url: str, cache_name: Optional[str] = None) -> Optional[str]:
    """ Saves image, and url address dict to cache folder
    Args
        img         (PIL.Image)
        url         (str)
        cache_name  (str [None])
    """
    if cache_name is None:
        return None

    cache_dict = ObjDict()
    cache_file = osp.join(osp.dirname(cache_name), 'image_cache.json')
    if osp.isfile(cache_file):
        cache_dict.from_json(cache_file)
    cache_dict.update({osp.basename(cache_name):url})
    cache_dict.to_json(cache_file)

    img.save(cache_name)
    return cache_name

def open_url(url: str,
             cache_name: Optional[str] = None,
             dtype: str = "float32",
             out_type: str = "torch",
             channels: Optional[int] = None,
             transforms: Optional[_TT] = None) -> Optional[_Vector]:
    """ Opens url image and caches to 'cache_name' if not None
    Args
        url         (str) valid url
        cache_name  (str [None]) # if present and valid save
        dtype       (str ['float32']) in uint8, float16, float32, float64
        out_type    (str ['torch']) | 'numpy'
        channels    (int [None]) default open with stored channels, 1,3,4 - fix channels
        transforms  (torchvision transform [None]) apply torchvision transform
    """
    response = requests.get(url)
    if response.status_code == 200:
        try:
            out = Image.open(BytesIO(response.content))
            cache_image(out, url, cache_name)
            if transforms is not None:
                out = transforms(out)

            out = np.array(out, copy=False)
            out = to_np_dtype(out, dtype, out_type=out_type)
            out = np_fix_channels(out, channels, fillvalue=None)
            if out_type != "numpy":
                out = _to_torch(out, permute=(2, 0, 1))
            return out
        except:
            logging.debug("Failed to load image from  <{}>".format(url))

    logging.debug("Code {} Failed to load <{}>".format(response.status_code, url))

    return None

def open_img(img: str,
             out_type: str = "numpy",
             dtype: str = "float32",
             grad: Optional[bool] = None,
             device: Union[None, str, torch.device] = None,
             backend: Optional[str] = None,
             transforms: Optional[_TT] = None,
             channels: Optional[int] = None,
             verbose: bool = True) -> Optional[_Vector]:
    """ image open
        accimage:   faster for jpg RGB
        PIL
        cv2
    TODO replace print with logging
    """
    _backends = _image_backends(backend=None, img=img, dtype=dtype, channels=channels)
    _assert_dtype(dtype, out_type)
    out = None

    #  is url
    if not osp.isfile(img) and urlparse(img).scheme:
        cached_name = get_cache_name(img, cache_dir=None)
        if not osp.isfile(cached_name):
            out = open_url(img, cached_name, dtype=dtype, out_type=out_type, channels=channels, transforms=transforms)
        else:
            img = cached_name

    #   is image
    while out is None and _backends:
        backend = _backends.pop(0)
        # cycle through available _BACKENDS
        if backend == "accimage":
            out = open_acc(img, dtype=dtype, out_type=out_type, channels=channels, transforms=transforms)
        elif backend == "PIL":
            out = open_pil(img, dtype=dtype, out_type=out_type, channels=channels, transforms=transforms)
        elif backend == "opencv":
            out = open_cv(img, dtype=dtype, out_type=out_type, channels=channels, transforms=transforms)
    assert out is not None, f"could not open img {img} with available _BACKENDS {_backends}"

    if out_type == "torch":
        out.unsqueeze_(0)
        out = check_contiguous(out, verbose)

        if device is not None and out.device != torch.device(device):
            out = out.to(device=device)
        if grad is not None and out.requires_grad != grad:
            out.requires_grad = grad

    return out

def torch_fix_channels(x: torch.Tensor, channels: int, dim: int = 1) -> torch.Tensor:
    """ extend or contract dimensions to match channels
    """
    _shape = x.shape
    _ch = _shape[1]
    if _ch == channels:
        return x
    elif channels == 1:
        return grayscale(x)
    else:
        out = torch.cat([x]+[torch.unbind(x, dim)[0].unsqueeze(dim)]*(channels-_ch), dim=dim)
    return out

def np_fix_channels(img: np.ndarray, channels: int, fillvalue: Union[None, int,float] = None
) -> np.ndarray:
    """
    validate image has spec'd channels, clip channels or add channels
    all arrays returned as a shape length 3 array
    Args
        img         (ndarray) shape length 2 or 3
        channels    (int) expected number of channels
        fillvalue   (float [None]) if None repeats the first channel, else fills with value
    """
    assert len(img.shape) in [2, 3], "image has shape "+str(img.shape)

    logging.debug(f"fix channels, requested {channels}, image shape {img.shape}")

    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    if channels is None:
        return img
    _msg = "only rgb, rgba and greyscale conversions handled, requested channels "+str(channels)
    assert channels in [1, 3, 4], _msg

    if channels == img.shape[2]: # no action needed
        return img

    # 3 or 4 to 1, grayscale
    if channels == 1:
        return grayscale(img)

    # 1 to 3 or 4, repeat or fill with value
    if img.shape[2] == 1:
        if fillvalue is None:
            return np.repeat(img, channels, axis=2)
        return np.concatenate([img]+[fillvalue * np.ones(img.shape, dtype=img.dtype)]*(channels - 1), axis=2)

    # 4 to 3: remove alpha channel
    if channels == 3 and img.shape[2] == 4:
        return img[:, :, :3]

    # 3 to 4: add alpha channel
    if channels == 4 and img.shape[2] == 3:
        return np.concatenate((img, np.zeros(img[:, :, :1].shape, dtype=img.dtype)), axis=2)


def grayscale(img: _Vector) ->  _Vector:
    """ convert 3 or 4 channel image to grayscale
    using RGB* [0.2989, 0.5870, 0.1140]
    """
    if isinstance(img, torch.Tensor):
        return _grayscale_torch(img)
    else:
        return _grayscale_np(img)

def _grayscale_assert(img: _Vector, channel_index: int = 2, dims: int = 3) -> bool:
    _msg = f"{Col.RB}expecting rgb or rgba, not {str(img.shape)}{Col.AU}"
    assert len(img.shape) == dims and img.shape[channel_index] in [3, 4], _msg
    return True

# @overload
def _grayscale_np(img: np.ndarray) -> np.ndarray:
    """
        convert 3 channel image to greyscale
        4th channel is ignored
    """
    if _grayscale_assert(img, channel_index=2, dims=3):
        _img_dtype = img.dtype.name
        if _img_dtype =="uint8":
            img = img.astype('float32')/255.
        tog = np.array([0.2989, 0.5870, 0.1140], dtype=img.dtype)
        img = np.clip((img[:, :, :3].reshape(-1, 3)*tog).sum(axis=-1), 0., 1.).reshape(*img.shape[:2], 1)
        if _img_dtype =="uint8":
            img = np.round(img*255).astype('uint8')
        return img

# @overload 
def _grayscale_torch(img: torch.Tensor) -> torch.Tensor:
    """
        convert 3 or 4 channel image to greyscale
    """
    if _grayscale_assert(img, channel_index=1, dims=4):
        n, c, h, w = img.shape

        tog = torch.tensor([[[0.2989], [0.5870], [0.1140]]], dtype=img.dtype, device=img.device)
        return torch.clamp((img[:, :3, ...].reshape(n, 3, -1)*tog).sum(axis=1, keepdims=True), 0,1).reshape(n, 1, h, w)

def save_img(data: np.ndarray,
             name: str,
             ext: str = ".png",
             folder: str = ".",
             conflict: int = 1,
             bpp: int = 8) -> bool:
    """
    save image out with  PIL (or cv2 if bpp==16)
    """
    name = _resolve_path(name=name, ext=ext, folder=folder, conflict=conflict)
    backend = "opencv" if bpp == 16 else "PIL"

    for _, _d in enumerate(data[0]):
        if backend == "opencv":
            cv2.imwrite(name, _d)
        else:
            if len(_d.shape) == 3 and _d.shape[2] == 1:
                _d = _d[:2]
            _im = Image.fromarray(_d)
            _im.save(name)
        name = _increment_name(name)

        logging.info("save with backend: %s"%backend, _d.shape, _d.dtype, _d.min(), _d.max())

    return True

def _resolve_path(name: str, ext: str = ".png", folder: str = ".", conflict: int = 1) -> str:
    """
    """
    folder = osp.abspath(osp.expanduser(folder))
    os.makedirs(folder, exist_ok=True)
    _exts_pil = [".bmp", ".dib", ".gib", ".jfif", ".jpeg", ".jpg", ".pcx", ".png", ".pbm",
                 ".pgm", ".ppm", ".pnm", ".tga", ".tga", ".tiff", ".tif"]
    _exts_cv2 = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm",
                 ".ppm", ".sr", ".ras", ".tiff", ".tif"]
    _name, _ext = osp.split(name)
    if _ext not in _exts_cv2 and _ext not in _exts_pil:
        name = _name + ext

    name = osp.join(folder, name)
    if osp.isfile(name):
        if not conflict:
            msg = "file_util, save_img(), file <%s> exists, nothing saved"%name
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
            name = None
        if conflict < -1:
            msg = "file <%s> exists, overwriting"%name
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
        if conflict > 0:
            name = _increment_name(name)
    return name

def _increment_name(name: str) -> str:
    while osp.isfile(name):
        _name, _ext = osp.splitext(name)
        i = -1
        if not _name[i:].isnumeric():
            name = _name+"_1"+_ext
        else:
            while _name[i:].isnumeric():
                i -= 1
            i += 1
            numeric = _name[i:]
            number = int(_name[i:])+1
            name = _name[:i]+"%%0%dd"%len(numeric)%int(number)+_ext
    return name
