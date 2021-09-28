"""@xvdp

File opening utilities
"""
from typing import Any, Union
from inspect import currentframe, getframeinfo
import warnings
import io
import os
import os.path as osp
from urllib.parse import urlparse
import requests
import numpy as np
import torch
import torchvision.transforms as tt


# todo fix image backend globals
IMG_BACKEND = None
_accimage = False
_opencv = False
_pil = False
try:
    import accimage # fastest
    _accimage = True
    IMG_BACKEND = "accimage"
except:
    print("accimage NOT found")
    pass
try:
    import cv2
    _opencv = True
    if IMG_BACKEND is None:
        IMG_BACKEND = "opencv"
except:
    print("opencv NOT found, some functionality may fail")
    pass
try:
    from PIL import Image
    _pil = True
    if IMG_BACKEND is None:
        IMG_BACKEND = "PIL"
except:
    print("PIL NOT found")

if IMG_BACKEND is None:
    print("One of 'accimage', 'PIL' or 'opencv' required")

# pylint: disable=no-member
# pylint: disable=not-callable

def open_acc(img: str, dtype: str="float32", out_type: str="numpy", channels: int=None,
             transforms: tt=None) -> np.ndarray:
    """ opens image using accunage"""

    assert dtype in ("uint8", "float32"), "accimage only ouputs float32 or uint8"
    pic = accimage.Image(img)
    if transforms is not None:
        pic = transforms(pic)

    if out_type == "numpy":
        out = np.zeros([pic.channels, pic.height, pic.width], dtype=np.__dict__[dtype])
        pic.copyto(out)
        out = out.transpose(1, 2, 0)
        out = np_fix_channels(out, channels, fillvalue=None)

    elif out_type == "torch":
        out = np.zeros([pic.channels, pic.height, pic.width], dtype=np.__dict__[dtype])
        pic.copyto(out)
        out = torch.from_numpy(out)
    return out

def open_pil(img: str, dtype: str="float32", channels:int =None, transforms: tt=None) -> np.ndarray:
    """ opens image using PIL"""
    out = Image.open(img)
    if transforms is not None:
        out = transforms(out)

    out = np.array(out, copy=False) #.convert("RGB")
    out = totype(out, dtype)
    out = np_fix_channels(out, channels, fillvalue=None)
    return out

def open_cv(img: str, dtype: str="float32", channels: int=None, transforms:tt=None) -> np.ndarray:
    """ opens image using opencv"""
    out = cv2.cvtColor(cv2.imread(img), cv2.IMREAD_UNCHANGED)# cv2.COLOR_BGR2RGB)
    if transforms is not None:
        out = transforms(out)
    out = totype(out, dtype)
    out = np_fix_channels(out, channels, fillvalue=None)
    return out

def totype(out: np.ndarray, dtype:str) -> np.ndarray:
    """ type conversion
    """
    _dtype = out.dtype.name
    if dtype in _dtype:
        return out
    if "float" in dtype or "double" in dtype:
        out = out.astype(np.__dict__[dtype])
    if _dtype == 'uint8':
        out /= 255

    if dtype == "int64":
        out = (out*(2**63-1)).astype(np.__dict__[dtype])
    elif dtype == "int32":
        out = (out*(2**31-1)).astype(np.__dict__[dtype])
    return out


def open_url(img: str, cache_img: str, dtype: str="float32", channels:int=None,
             transforms:tt=None) -> np.nadarray:
    """ opens jpges or pngs from url
        caches it to path
    """
    _image_types = ("image/jpeg", "image/png")

    # open url and save cache
    out = None
    _res = requests.get(img)
    if _res.ok and _res.headers.get('content-type') in _image_types:

        try:
            out = Image.open(io.BytesIO(_res.content))
            out.save(cache_img)
            _cacheinfo = osp.join(osp.split(cache_img)[0], "cache_info.csv")
            with open(_cacheinfo, "a") as _fi:
                _fi.write("%s,%s\n"%(cache_img, img))
            print("Opened Image <%s> from url, saved to <%s>"%(img, cache_img))
        except:
            print("could not open url in pil <%s>"%img)

    if transforms is not None:
        out = transforms(out)

    out = np.array(out, copy=False) #.convert("RGB")
    out = totype(out, dtype)
    out = np_fix_channels(out, channels, fillvalue=None)
    return out


def _cache_name(path:str, cache:str="~/.cache/images") -> str:
    cache = osp.expanduser(cache)
    os.makedirs(cache, exist_ok=True)
    _name, _ext = osp.splitext("_".join(path[1:].split("/"))[:128])

    _exts = ".jpeg", ".jpg", ".png"
    if _ext.lower() not in _exts:
        for _e in _exts:
            if _e in _ext.lower():
                _ext = _e
        if _ext not in _exts:
            _ext = ".png"

    return osp.join(cache, _name + _ext)

def open_img(img: str, out_type: str="numpy", dtype: str="float32", grad: bool=None,
             device: str=None, backend: str=None, transforms: tt=None, channels: int=None,
             verbose: bool=False) -> Any:
    """ general image open"""
    global IMG_BACKEND
    _dtypes = ("double", "float64", "float", "float32", "float16", "int64", "int32", "uint8")
    assert dtype in _dtypes, "The only supported types are: %s"%(str(_dtypes))
    if backend is None:
        backend = IMG_BACKEND
    if dtype not in ("uint8", "float32") and backend == "accimage":
        if _opencv:
            backend = "opencv"
        elif _pil:
            backend = "PIL"
        else:
            assert False, "opencv or PIL necessary to import dtype '%s'"%dtype


    if not osp.isfile(img):
        _url = urlparse(img)
        assert _url.scheme, "file is neither found in <%s>, nor is it a url"%img
        cache_img = _cache_name(_url.path)
        if osp.isfile(cache_img):
            img = cache_img
        else:
            try:
                out = open_url(img, cache_img, dtype=dtype, channels=channels,
                               transforms=transforms)
                backend = "url"
            except:
                print("cannot open url '%s'"%img)
                return None

    if backend == "accimage":
        try:
            out = open_acc(img, dtype=dtype, out_type=out_type, channels=channels,
                           transforms=transforms)
        except:
            if verbose:
                print("cannot open '%s' with accimage"%img)
            backend = "PIL"

    if backend == "PIL":
        try:
            out = open_pil(img, dtype=dtype, channels=channels, transforms=transforms)
        except:
            if verbose:
                print("cannot open '%s' with PIL"%img)
            backend = "opencv"

    if backend == "opencv":
        try:
            out = open_cv(img, dtype=dtype, channels=channels, transforms=transforms)
        except:
            print("cannot open '%s' with openCV"%img)
            return None

    if backend in ("opencv", "PIL", "url") and out_type == "torch":
        out = torch.from_numpy(out).permute(2, 0, 1)
        if not out.is_contiguous():
            out.contiguous()
    if verbose:
        print("   opened with '%s'"%(backend))

    if out_type == "torch":
        if device is not None and out.device != torch.device(device):
            out = out.to(device=device)
        if grad is not None:
            out.requires_grad = grad

        if verbose:
            _shape = "img_shape'%s'"%str(torch.tensor(out.shape).tolist())
            _dtype = "dtype '%s'"%str(out.dtype)
            _device = "device '%s'"%out.device
            _grad = "grad '%s'"%out.requires_grad
            print("   %s, %s, %s, %s"%(_shape, _dtype, _device, _grad))

    return out

def torch_fix_channels(tensor: torch.Tensor, channels: int, dim: int=1) -> torch.Tensor:
    """ extend or contract dimensions to match channels
    """
    _shape = tensor.shape
    _channels = _shape[1]
    if _channels == channels:
        return tensor
    out = torch.zeros(torch.cat([tensor]+[torch.unbind(tensor, dim)[0].unsqueeze(dim)]*(channels - _channels)), dim=dim)
    return out

def np_fix_channels(img: np.ndarray, channels: int, fillvalue: Union[int,float]=None) -> np.ndarray:
    """
    validate image has spec'd channels, clip channels or add channels
    all arrays returned as a shape length 3 array
    Args
        img         (ndarray) shape length 2 or 3
        channels    (int) expected number of channels
        fillvalue   (float [None]) if None repeats the first channel, else fills with value
    """
    assert len(img.shape) in [2, 3], "image has shape "+str(img.shape)

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
        return np.concatenate([img]+[fillvalue * np.ones(img.shape)]*(channels - 1), axis=2)

    # 4 to 3: remove alpha channel
    if channels == 3 and img.shape[2] == 4:
        return img[:, :, :3]

    # 3 to 4: add alpha channel
    if channels == 4 and img.shape[2] == 3:
        return np.concatenate((img, np.zeros(img[:, :, :1].shape)), axis=2)

def grayscale(img: np.ndarray) -> np.ndarray:
    """
        convert 3 or 4 channel image to greyscale
    """
    _msg = "expecting rgb or rgba, not "+str(img.shape)
    assert len(img.shape) == 3  and img.shape[2] in [3, 4], _msg
    tog = np.array([0.2989, 0.5870, 0.1140])
    # flattens, multiplies, sums
    return np.ceil((img[:, :, :3].reshape(-1, 3)*tog).sum(axis=1)).astype(img.dtype).reshape(*img.shape[:2], 1)

def save_img(folder: str, name: str, ext: str, data: np.ndarray, bpp: int, conflict: int, backend: str) -> bool:
    """
    """
    if folder is None:
        folder = '.'
    folder = osp.abspath(folder)
    if not osp.isdir(folder):
        os.mkdir(folder)
    name = osp.join(folder, name) + ext
    if osp.isfile(name):
        if not conflict:
            msg = "file_util, save_img(), file <%s> exists, nothing saved"%name
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
            return None
        if conflict < -1:
            msg = "file <%s> exists, overwriting"%name
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
        if conflict > 0:
            name = _increment_name(name)

    _backend = IMG_BACKEND
    if backend is not None:
        _backend = backend
    if bpp == 16:
        assert _opencv, "only cv2 backend can currently save out 16bpp files, not found"
        _backend = "opencv"

    for _, _d in enumerate(data[0]):
        if _backend == "opencv":
            cv2.imwrite(name, _d)
        else:
            print("save with backend: %s"%_backend, _d.shape, _d.dtype, _d.min(), _d.max())
            if _d.shape[2] == 1:
                _d = _d*np.ones([1, 1, 3], dtype=_d.dtype)
            _im = Image.fromarray(_d)
            _im.save(name)
        name = _increment_name(name)

    #print("TODO: !!! save target layers out")
    return True

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


def get_files(folder: Union[str, list, tuple]=".", ext: Union[str, list, tuple]=None, recursive: bool=False) -> list:
    """ conditional file getter
    Args
        folder      (str|list ['.'])  folder | list of folders
        ext         (str|list [None]) file extensions, default, any
        recursive   (bool[False])
    """
    folder = [folder] if isinstance (folder, str) else folder
    folder = [osp.abspath(osp.expanduser(f)) for f in folder]
    ext = [ext] if isinstance (ext, str) else ext
    cond = lambda x, ext: True if ext is None else osp.splitext(x)[-1].lower() in ext

    out = []
    for fold in folder:
        if not recursive:
            out += [f.path for f in os.scandir(fold) if f.is_file() and cond(f.name, ext)]
        else:
            for root, _, files in os.walk(fold):
                out += [osp.join(root, name) for name in files if cond(name, ext)]
    return sorted(out)

def verify_image(name: str, verbose: bool=False) -> bool:
    """
    True:  48 us
    False: 68 us
    jpg and png headers with open('rb') is ~3x to 6x faster, but this is practical and tested
    """
    try:
        im = Image.open(name)
        return True
    except:
        if verbose:
            print(f" Not an Image: {name}")
        return False

def verify_images(images: list, verbose: bool=False) -> list:
    return [im for im in images if verify_image(im, verbose)]

def get_images(folder: Union[str, list, tuple]=".", recursive: bool=False,
               verify: bool=True, verbose: bool=True) -> list:
    """ conditional image file getter
    Args
        folder      (str|list ['.'])  folder | list of folders
        recursive   (bool [False])
        verify      (bool [True]), leverages PIL to read the header of each file
                        may be a bit slower
                        loading verify_image2() should be faster but is not fully tested
        verbose     (default [True])
    """
    _images = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    out = get_files(folder=folder, ext=_images, recursive=recursive)

    if verify:
        out = verify_images(out, verbose)

    if verbose:
        print(f"get_images()-> {len(out)} found")
    return out

def rndlist(inputs: Union[list, tuple, np.ndarray, torch.Tensor], num: int=1) -> Any:
    """ returns random subset from list
    Args
        inputs   (iterable)
        num      (int [1]) number of elements returned
    """
    choice = np.random.randint(0, len(inputs), num)
    if isinstance(inputs, (np.ndarray, torch.Tensor)):
        return inputs[choice]
    return [inputs[c] for c in choice]
