"""
(c) xvdp
"""
import os
import os.path as osp
from inspect import currentframe, getframeinfo
import warnings
import collections
import copy
import numbers
import math
from urllib.parse import urlparse
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from .func_target import *
from .func_random import *
from .func_cv2 import rotatecv2d
from .func_color import *

from .tensor_util import check_contiguous, check_tensor, unfold_tensors, refold_tensors, assert_tensor_list_eq
from .file_util import open_img, save_img
from .func_util import *
from .fft_utils import FFT
from .show_util import showim, showhist, showims
from .np_util import np_shift_target, np_validate_dtype, np_grid, assert_np_list_eq, get_bpp
from .laplace import compute_pad, get_pyrdata, build_coords
from .. import config
from ..util import mem
from ..nn import GaussNoise

Iterable = collections.abc.Iterable

try:
    import accimage
    #assert "copy" in accimage.Image.__dict__, "install accimage from https://github.com/xvdp/accimage"
except ImportError:
    accimage = None

try:
    import cv2
except ImportError:
    cv2 = None

# pylint: disable=no-member
# pylint: disable=not-callable

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def open_file(file_name, dtype, device, grad=False, out_type="torch", channels=None, transforms=None, verbose=False):
    """transforms.Open()
    Opens a filename as torch tensor
    TODO: expand to open audio, volume files, 4, 1 channel images, high dynamic range
    TODO create label file
    TODO ensure grad requirement is propagated
    TODO cleanup numpy out_type
    # file_util: cannot open accimage with dtype == uint8

    Args:
        file_name   (string): valid existing filename
        dtype       (string): torch data format (Default "float32")
        device      (string): "cuda" or "cpu"; opens casts tensor on creation to device
        grad        (bool):    requires_grad

    """
    ## kind of crapish iffs
    if device == "cpu" and dtype == "float16":
        dtype = "float32"
        print(f"cpu does not support half, opening as float32")
    ## TODO fix accimage for uint8
    if out_type == "numpy":
        dtype = "uint8"
    ## TODO what if the stuff is multimodal. damnit

    if isinstance(file_name, (list, tuple)):
        batchlist = []
        size = None
        _covert_to_tensor = out_type == "torch"
        for i, _file in enumerate(file_name):
            tensor = open_file(_file, dtype=dtype, device=device, grad=grad, out_type=out_type,
                               channels=channels, transforms=transforms, verbose=verbose)
            if config.DEBUG:
                _size = check_tensor(tensor)
                mem.report(config.MEM, "open_file, sz: %dMB"%(_size/2**20))
                config.MEMORY += check_tensor(_size)

            if i == 0:
                size = tensor.shape
            _covert_to_tensor = (False, True)[size == tensor.shape]
            batchlist.append(tensor)

        if _covert_to_tensor:
            tensor = torch.cat(batchlist, dim=0)
            return check_contiguous(tensor, verbose)
        if out_type == "numpy":
            return np.vstack(batchlist)
        return batchlist

    assert osp.isfile(file_name) or urlparse(file_name).scheme, "filename not found"

    if verbose:
        #print("config verbose", config.VERBOSE)
        if out_type == "numpy":
            print("Open(): F.open_file(): '%s' as numpy"%(file_name))
        print("Open(): F.open_file(): '%s' using device '%s'"%(file_name, device))

    tensor = open_img(file_name, out_type=out_type, dtype=dtype, grad=grad, device=device, 
                      backend=None, channels=channels, transforms=transforms, verbose=verbose)
    if tensor is None:
        return None

    if config.DEBUG:
        _size = check_tensor(tensor)
        # if config.MEM is None:
        #     config.MEM = CudaMem(config.DEVICE)
        # config.MEM.report("open_file, sz: %d"%(_size/2**20))
        _msg = "open_file, sz: %dMB"%(_size/2**20)
        mem.report(memory=config.MEM, msg=_msg)
        config.MEMORY += _size

    if out_type == "torch":
        tensor.unsqueeze_(0)
        tensor = check_contiguous(tensor, verbose)

    return tensor


def assert_types_equal(type_a, type_b):
    """ patch, TODO use only lists, not tuples
    """
    if not (type_a in (list, tuple) and type_b in (list, tuple)):
        assert type_a == type_b, "cannot merge 2 data types, %s and %s"%(type_a, type_b)

def merge(data_list):
    """ Merges [tensor,...,] or [(tensor, target, label),...,]
    Returns [tensor, [target,...], [label,...]]

    Args
        data_list [tensor,...,] or [(tensor, target, label),...,]
    """
    _assert_msg = "merge requires list of (tensor, target, label)"
    assert isinstance(data_list, (list, tuple)), _assert_msg

    _merge_type = None # type of main merge element
    _sub_types = [] # if merge type is list or tuple, check subtypes

    for i, data in enumerate(data_list):
        if i == 0:
            _merge_type = type(data)
        else:
            assert_types_equal(_merge_type, type(data))
            # cannot merge 2 data types, %s and %s"%(_merge_type, type(d))

        if isinstance(data, (list, tuple)):
            if i == 0:
                _sub_types = [type(_d) for _d in data]
            else:
                _msg = "cannot merge unequal len subtype lists, %d, %d"%(len(_sub_types), len(data))
                assert len(_sub_types) == len(data), _msg
                _st = [type(_d) for _d in data]
                #print(_st)
                _msg = "list members must all have same type, got %s %s"%(str(_sub_types), str(_st))
                assert all(_sub_types[j] == _st[j] for j in range(len(_sub_types))), _msg

    if _merge_type == torch.Tensor:
        _shape, _dtype, _device = assert_tensor_list_eq(data_list, tolerant=False)
        return torch.cat(data_list, dim=0)

    elif _merge_type == np.ndarray:
        assert_np_list_eq(data_list)
        return np.stack(data_list, axis=0)

    elif _merge_type in (list, tuple):
        out = [[] for i in range(len(data_list[0]))]

        for i, data in enumerate(data_list):
            for j in range(len(data_list[0])):
                out[j].append(data[j])

        for i, _ in enumerate(out):
            if torch.is_tensor(out[i][0]):
                out[i] = torch.cat(out[i], dim=0)
            elif isinstance(out[i][0], np.ndarray):
                out[i] = np.stack(out[i], axis=0)
            elif isinstance(out[i][0], (tuple, list)):
                _out = []
                for _o in out[i]:
                    _out += list(_o)
                out[i] = _out
            elif isinstance(out[i][0], type(None)):
                out[i] = None

    return out

def addbbox(data, bboxdata, index=None, inplace=False):
    """tranforms.AddBox()
    Needs target/label centric full rewrite
            Passing Net requirements. eg. this net outputs n targets.
        Store target info as db, pass int labels and uuid connection to db

        Args,
            data        (batch_tensor or tuple of (batch_tensor, target_tensor_list, labels))
            bboxdata    (tensor or list of boxtensors),
                if boxdata is tensor, batch_tensor size needs to be 1
                    or index < n and data tuple with valid sized target_tensor_list
            index       (int, default None)

    """

    _tensor_list_idx = False
    if isinstance(data, (tuple, list)):
        data, types = unfold_tensors(data, msg="addbbox(), input", inplace=inplace)
        _tensor_list_idx = [i for i in range(len(types)) if types[i] == "tensor_list"]
    else:
        data = [data]

    _n = len(data[0])

    if isinstance(bboxdata, torch.Tensor):
        _msg = "bad bbox shape %s for tensor shape %s "%(tuple(bboxdata.shape), tuple(data.shape))
        assert _n == 1, _msg
        bboxdata = [bboxdata]

    elif isinstance(bboxdata, (list, tuple)):
        if index is not None:
            if index == len(data):
                _tensor_list_idx = False
            elif index > len(data):
                assert isinstance(index, int) and len(data) > index, "index out of bounds, %d < %d"%(index, len(data))
            elif index not in _tensor_list_idx:
                assert False, "cant add target at position {}, occupied by item of type {}".format(index, types[index])
            
        assert _n == len(bboxdata), "bad bbox len %d for batch size %d"%(len(bboxdata), _n)
    else:
        print("bounding box data type not understood", type(bboxdata))
        raise NotImplementedError

    if not _tensor_list_idx:
        data += list(bboxdata)
    else:
        _tt = data[_tensor_list_idx]

        if index is not None:
            assert _tt[index][0].shape == bboxdata[0][0].shape
            _tt[index] = torch.cat((_tt[index], bboxdata[0]))
        else:
            for i, _ in enumerate(bboxdata):
                _reqshape = _tt[i][0].shape
                _gotshape = bboxdata[i][0].shape
                assert _reqshape == _gotshape, "req:%s, got %s"%(str(tuple(_reqshape)),
                                                                 str(tuple(_gotshape)))
                _tt[i] = torch.cat((_tt[i], bboxdata[i]))

    return data


def to_tensor(pic, dtype=None, device=None, verbose=False): #simiar to torvision.transforms.functional
    """transofmrs.ToTensor()
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            img = img.float().div(255)

    elif accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        img = torch.from_numpy(nppic)

    else:
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            img = img.float().div(255)

    img = check_contiguous(img.unsqueeze_(0), verbose)
    dtype = None if dtype is None else torch.__dict__[dtype]
    img = img.to(device=device, dtype=dtype)
    return img

def fork(data, number):
    """ transforms.Fork()
        concatenates data to n size batch
    """
    data, _ = unfold_tensors(data, msg="fork() input:, ", inplace=True)

    if config.DEBUG:
        _old_size = check_tensor(data[0])

    assert isinstance(number, int), "number needs to be integer"

    for i, datum in enumerate(data):
        if torch.is_tensor(datum):
            data[i] = torch.cat([data[i] for j in range(number)], dim=0)
        elif isinstance(datum, (list, tuple)):
            data[i] = data[i]*number

    if config.DEBUG:
        config.TRANSIENT += _old_size
        config.MEMORY += check_tensor(data[0]) - _old_size

    return data

def to_numpy(data, ncols=None, pad=1, bpp=0, clean=False, inplace=False, allow_dims=(1, 3),
             dtype=None, mode=config.BoxMode.xywh):
    """ transforms.ToNumpy()
    Converts a ``torch.Tensor`` to ``numpy.array``
    if tensor is 2D (NCHW) return numpy format NHWC
        if ncols is not None, builds an image grid format HWC
    1 and 3D tensors are returned as numpy of same shape
    if torch image is not float converts it to float

    """
    data, types = unfold_tensors(data, msg="to_numpy() ", inplace=True)
    tensor = data[0]

    if config.VERBOSE:
        print("ToNumpy(): dims", tensor.ndimension(), "ncols", ncols)

    if inplace and tensor.is_cuda:
        msg = "to_numpy(): image in cuda tensors cannot share memory with numpy"
        frameinfo = getframeinfo(currentframe())
        warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
        inplace = False

    # legacy show only first stet of targets
    ndtarget = []
    for i in range(1, len(data)):
        if types[i] == "tensor_list":
            for target_tensor in data[i]:
                if target_tensor is not None and len(target_tensor) > 0:
                    assert torch.is_tensor(target_tensor), "tensor required, found (%s)"%type(target_tensor)
                    ndtarget.append(_to_numpy(target_tensor, inplace))
            break
    labels = []
    if len(data) > 1 and types[-1] == "Tensor":
        labels = data[-1].tolist()

    # convert tensor
    if tensor.ndimension() == 4: #only tensors with shape NCHW unfolded for view
        ndarray = _to_numpy(tensor, inplace, dtype=dtype).transpose(0, 2, 3, 1)
        if ncols is not None and ncols > 0:
            ndarray, ndtarget = np_grid(ndarray, target=ndtarget, ncols=ncols, pad=pad,
                                        mode=mode, allow_dims=allow_dims)
            if ndarray.ndim == 3 and ndarray.shape[2] == 1:
                ndarray = ndarray.reshape(ndarray.shape[:2])
    elif tensor.ndimension() == 3:
        ndarray = _to_numpy(tensor, inplace, dtype=dtype).transpose(1, 2, 0)
    else:
        ndarray = _to_numpy(tensor, inplace, dtype=dtype)

    # print("to_numpy()\n\t", ndarray.shape)

    if bpp:
        assert bpp in (8, 16), "only 8 and 16 bpp conversions handled"
        if ndarray.min() < 0 or ndarray.max() > 1:
            msg = "to_numpy(): image in %.3f - %.3f, hard clamp"%(ndarray.min(), ndarray.max())
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
            ndarray = np.clip(ndarray, 0., 1.)

        img_bpp = get_bpp(ndarray)
        if img_bpp > bpp:
            msg = "to_numpy(): has precision %dbpp, requested  %dbpp, compressing"%(img_bpp, bpp)
            frameinfo = getframeinfo(currentframe())
            warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)

        if bpp == 8:
            ndarray = (ndarray*(2**8 -1)).astype(np.uint8)
        else:
            ndarray = (ndarray*(2**16 -1)).astype(np.uint8)

    if (clean or config.FORCE_CLEANUP) and tensor.is_cuda:
        assert (not inplace), "to_numpy(), cannot delete a tensor that is held in place"
        del tensor
        torch.cuda.empty_cache()

    return [ndarray, ndtarget, labels]

def _to_numpy(tensor, inplace, dtype=None):
    """ handle copy vs reference of cpu data numpy pytorch cpu
    # converts to float; why?
    """
    npdtypes = ("float", "double", "uint8", "float32", "float64", "int", "int64", "int32")
    dtypein = tensor.dtype.__repr__().split('.')[1]
    dtypein = "float32" if dtypein == "float16" else dtypein
    dtype = dtypein if dtype is None else dtype
    assert dtype in npdtypes, "only %s dtypes accepted as numpy images"%str(npdtypes)

    if not inplace and not tensor.is_cuda:
        ndarray = tensor.data.detach().to("cpu").numpy().copy()
    else:
        ndarray = tensor.data.detach().to("cpu").numpy()

    # if ndarray.dtype != np.uint8:
    if dtype == "uint":
        ndarray *= 255
    ndarray = ndarray.astype(dtype)
    return ndarray

def reorder_labels(labels):
    # TODO redo labels
    if labels is None:
        return labels
    targets = []
    for label in labels:
        targets += label['targets']
    return targets

def is_tensorlist(data):
    """ checks if data is list or tuple of torch tensors
    """
    if not isinstance(data, (list, tuple)):
        return False
    _istensor = all([isinstance(data[i], torch.Tensor) for i in range(len(data))])
    if _istensor:
        return True
    _islist = all([isinstance(data[i], (tuple, list)) for i in range(len(data))])
    if not _islist:
        return False
    return all([isinstance(data[i][0], torch.Tensor) for i in range(len(data))])

def are_images(data):
    """
    """
    _istensor = lambda x: torch.is_tensor(x) and x.dtype == torch.float
    for item in data:
        if not _istensor(item) and not _istensor(item[0]):
            return False
    return True

def is_tensor_image_list(data):
    """ is list of tensors
    """
    # are_images = lambda x: all([x[0].ndim == y.ndim for y in x]) and all([y.dtype == torch.float32 for y in x])
    return is_tensorlist(data) and are_images(data)

def is_tensor_batch(data):
    """ is list batch, first element is image, rest arent
    """
    _list = isinstance(data, (list, tuple))
    _imagefirst = data[0].dtype == torch.float32
    _notimage_others = not all([data[0].ndim == y.ndim for y in data])
    _are_equal_len = all([(y is None or len(data[0]) == len(y)) for y in data])
    return _list and _imagefirst and _notimage_others and _are_equal_len

def show(data, ncols, pad, show_targets, annot, width, height, path, as_box, max_imgs,
         unfold_channels, **kwargs):
    """ functional for transforms.Show()
        Args
            data    (ndarray img)   # image can be 4,3 or 2d
                    ([ndarray img, [ndarray annot], ndarray tgts])
                    (tensor img)

                    ([tensor img, tensor_list annot, tensor tgts])           < one image with annotations
                    ([tensor, ..., tensor] imgs)                             < list of images without anotations
                    ([[tensor img, tensor_list annotd, tensor tgts], ...,[img anns, tgts]])

    TODO: redo show:
    1. if list, tuple
        - look at dims in list elements
            if NCHW and N > 1
                unfold cat
            if CHW or 1CWH
                subplot
        - if bounding boxes: show

    2. if NCWH or CWH

    """
    if config.VERBOSE:
        print("show(): ncols", ncols)

    if "mode" in kwargs:
        config.set_boxmode(kwargs["mode"])

    allow_dims = [1, 3, 4]
    mode = config.BOXMODE if "mode" not in kwargs else kwargs["mode"]

    if isinstance(data, np.ndarray):
        if config.VERBOSE:
            print("show(): ndarray")
        showim(data, targets=None, labels=None, show_targets=show_targets, annot=annot, width=width,
               height=height, path=path, as_box=as_box, **kwargs)
        return

    if isinstance(data[0], np.ndarray):
        if config.VERBOSE:
            print("show(): ndarray_list")
        targets = None if len(data) == 1 else data[1]
        labels = None if len(data) < 3 else data[2]
        showim(data[0], targets=targets, labels=labels, show_targets=show_targets, annot=annot,
               width=width, height=height, path=path, as_box=as_box, **kwargs)
        return

    if "bbox" in kwargs and isinstance(data, (list, tuple)) and len(data) > 1:
        if data[1] is not None and len(data[1]) > 0:
            box = targets_bbox(data[1])
            data = addbbox(data, box)

    # list containing single batch of tensors
    if isinstance(data, (tuple, list)) and len(data)==1:
        data = data[0]
    
    # list of single image tensors 1,C,H,W or C,W,H
    if is_tensor_image_list(data):
        if config.VERBOSE:
            print("show(): tensor_image_list")
        images = []
        if ncols is None:
            ncols = min(len(data), 8)
        images = []
        for _data in data:
            image, targets, labels = to_numpy(_data, ncols, pad=pad, allow_dims=allow_dims,
                                              mode=mode)
            images.append(image)

        showims(images, targets=None, labels=None, show_targets=show_targets, annot=annot,
                width=width, height=height, path=path, as_box=as_box,
                unfold_channels=unfold_channels, ncols=ncols, **kwargs)

    # if list of lists containing equal image tensors
    elif is_tensor_batch(data):
        if config.VERBOSE:
            print("show(): tensor_batch")
        if ncols is None:
            ncols = min(len(data[0]), 8)
        images, targets, labels = data_to_numpy(data=data, pad=pad, max_imgs=max_imgs, ncols=ncols,
                                                width=width, inplace=False, **kwargs)

        showim(images, targets, labels, show_targets=show_targets, annot=annot, width=width,
               height=height, path=path, as_box=as_box, unfold_channels=unfold_channels,
               ncols=ncols, **kwargs)

    elif torch.is_tensor(data) and data.dtype in (torch.float32, torch.uint8):
        if config.VERBOSE:
            print("show(): tensor")
        if ncols is None:
            ncols = min(len(data), 8)
        images, targets, labels = data_to_numpy(data=data, pad=pad, max_imgs=max_imgs, ncols=ncols,
                                                width=width, inplace=False, **kwargs)

        showim(images, targets, labels, show_targets=show_targets, annot=annot, width=width,
               height=height, path=path, as_box=as_box, unfold_channels=unfold_channels, **kwargs)


def data_to_numpy(data, pad, max_imgs, ncols, width=10, inplace=True, **kwargs):
    """ convert tensor, and tensor list to numpy and np list
    """
    data, types = unfold_tensors(data, inplace=inplace)

    if max_imgs > 0:
        data[0] = data[0].clone().detach().cpu()[:max_imgs]
    else:
        max_imgs = None

    for i in range(len(data)):
        if types[i] in ("tensor", "Tensor"):
            data[i] = data[i].clone().detach().cpu()[:max_imgs]
        elif types[i] in ("tensor_list", "list", "tuple"):
            data[i] = data[i][:max_imgs]

    if ncols is None:
        ncols = min(data[0].shape[0], 8)

    allow_dims = [1, 3, 4]
    # print("data_to_numpy()\n\t", data[0].shape)

    mode = config.BOXMODE if "mode" not in kwargs else kwargs["mode"]
    image, targets, labels = to_numpy(data, ncols, pad, allow_dims=allow_dims, mode=mode)

    if "hist" in kwargs:
        showhist(image, width)

    if data[0].is_cuda:
        del data[0]
        torch.cuda.empty_cache()

    return[image, targets, labels]

def save(data, folder, name, ext, bpp, conflict, backend):
    """ saves image out
    """
    _imgexts = [".jpg", ".png", ".tif", ".webp"]
    assert ext in _imgexts, "only %s extensions supported"%(_imgexts)
    assert bpp in (8, 16), "only 8 & 16 bpp supported"
    if backend is not None:
        backend = backend.lower()
        assert backend[0] in ('a', 'c', 'p'), "backends accepted: 'PIL, accimage, cv2"
    if bpp == 16:
        assert ext in (".tif", ".png"), "only 16bit tif and png supported"

    if ext in _imgexts:
        if isinstance(data, np.ndarray):
            if len(data.shape) < 4:
                data = np.expand_dims(data, 0)
            return save_img(folder, name, ext, [data], bpp, conflict, backend)
        
        return save_img(folder, name, ext, to_numpy(data, bpp=bpp), bpp, conflict, backend)

    print("unsupported format")
    return False


def resize(data, size, interpolation="linear", square=False, align_corners=False, verbose=False):
    r"""Resize the input torch image

    Args:
        img (Torch): Image, 1, 2 or 3d data to be resized
            accepted formats NCW, NCHW or NCWHD

        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`

        interpolation (str, optional): Desired interpolation.
            Default is ``linear``, options ``nearest, linear, area``

    Returns:
        torch.tensor: Resized image.
        target_tensor_list
        label_list
    """
    if config.VERBOSE:
        print("Resize():")

    data, types = unfold_tensors(data, msg="resize() input", inplace=True)
    tensor = data[0]

    dims = tensor.ndimension() - 2
    _tensor_size = list(tensor.size()[2:])

    # validate inputs
    #size = check_size(size)
    interpolation = check_interp(interpolation, ["nearest", "linear", "area", "cubic"])

    if interpolation in ("linear", "cubic"):
        if dims == 2:
            interpolation = "bi"+interpolation
        elif dims == 3:
            interpolation = "tri"+interpolation

    if size is None or square:
        _size = gen_size(tensor, size, as_tensor=False)
    elif isinstance(size, int):
        minsize = min(_tensor_size)
        _size = tuple((np.array(_tensor_size)*size/minsize).astype(int))
    elif isinstance(size, (list, tuple, np.ndarray)):
        _size = tuple(size)
    elif torch.is_tensor(size):
        _size = tuple(size.tolist())
    else:
        assert False, "size type not recognized <%s>"%size

    # do nothing if file is correct size
    if _size == _tensor_size:
        return data

    _min = tensor.min().item()
    _max = tensor.max().item()

    tensor = nn.functional.interpolate(tensor, size=_size, mode=interpolation,
                                       align_corners=align_corners).clamp(tensor.min(),
                                                                          tensor.max())
    data[0] = check_contiguous(tensor, verbose, msg="resize()interpolate()")

    for i in range(1, len(data)):
        if types[i] == "tensor_list":
            for j, _t in enumerate(_size):
                _size[j] /= _tensor_size[j]
            data[i] = scale_targets(data[i], _size)


    return data

def _zoom(data, scale=None, interpolation="linear", pad_color=0.5, inplace=False):
    """ _zoom() = resize and centercrop

        TODO:
            add bernouilli
            add distribution
            inplace
            expose to transforms
    """
    data, types = unfold_tensors(data, msg="_zoom() input", inplace=inplace)
    tensor = data[0]

    _size = torch.tensor(tensor.size()[2:]).to(dtype=torch.float) # device=tensor.device

    if scale is None: # TODO random scales
        scale = _size.div(_size.min()).pow(2).sum().sqrt().item()

    if scale == 1 or scale == 0:
        return data

    scale = abs(scale)

    if scale < 1: # scale down: 
        _new_size = (_size*scale).long()
        _s = ((_size-_new_size)/2).long()
        _e = _new_size + _s
        _tensor = torch.full_like(data[0], pad_color)

        data = resize(data, size=_new_size, interpolation=interpolation)
        _tensor[:, :, _s[0]:_e[0], _s[1]:_e[1]] = data[0]
        data[0] = _tensor

        for i in range(1, len(data)):
            if types[i] == "tensor_list":
                for j, target_tensor in enumerate(data[i]):
                    data[i][j] = shift_target(target_tensor, _s)
    else: # scale up: crop and scale up
        _crop_start = gen_center(1, tensor) -_size.view(1, -1, 1)/(2* scale)
        data = resize(crop(data, _crop_start.long(), (_size/scale).long()), size=_size,
                      interpolation=interpolation, square=True)
    return data

def rescale(data, scale, p, distribution, interpolation="linear", mean=None):
    r"""Scale the input torch image
    if multidim required: implementation
    http://localhost:8889/notebooks/pytorch_sketch/numpy%20random.ipynb

    Args:
        tensor (Torch): Image, 1, 2 or 3d data to be scaled
            accepted formats NCW, NCHW or NCWHD

        scale (sequence or positive real number): Desired scale factor.
            If size is sequence, output size will be scaled by each entry.
            Sequence length must equal Image Tensor shape length - 2.
            If size is an int or float, all dimensions will be scaled in unison.
        interpolation (str, optional): Desired interpolation.
            Default is ``linear``, options ``nearest, linear, area``

    Returns:
        torch.tensor: Resized image.
    """
    if config.VERBOSE:
        print("Scale():")

    if mean is None:
        if isinstance(scale, Iterable):
            mean = 0
            mean = sum([s/2.0 for s in scale])/len(scale)
        else:
            mean = scale/2.0

    interpolation = check_interp(interpolation, ["nearest", "linear", "area", "cubic"])
    scale = check_scale(scale)

    data, types = unfold_tensors(data, msg="rescale() input", inplace=True)
    tensor = data[0]

    dims = tensor.ndimension() - 2

    if p > 0:
        print("functional().rescale() p>0, unsed")
        # distribution = check_distribution(distribution)
        # prob = bernoulli(p, len(tensor), dtype=tensor.dtype, device=tensor.device, grad=False)
        #print(prob, scale)

    #     if distribution == config.RndMode.normal:
    #         bound_normal_proc(x, a=scale, b=mean, stds=3, size=1000000)
    for i in range(1, len(data)):
        if types[i] == "tensor_list":
            data[i] = scale_targets(data[i], scale)

    align_corners = None
    if interpolation in ("linear", "cubic"):
        align_corners = False
        if dims == 2:
            interpolation = "bi"+interpolation
        elif dims == 3:
            interpolation = "tri"+interpolation

    _min = tensor.min().item()
    _max = tensor.max().item()
    tensor = nn.functional.interpolate(tensor, scale_factor=scale, mode=interpolation,
                                       align_corners=align_corners).clamp(_min, _max)

    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="rescale() output:, ")

    return data

##
# affine transforms
#
#   .rotate
#   .flip
def rotate(data, angle, p, distribution, independent, center_choice, center_sharpness,
           interpolation, units, scalefit, inplace=False, debug=0):
    """ Rotates image by angle
    Args:
        data            (tuple of (tensor,target_tensors, label))

        angle           (float, tuple of floats), rotation angle, or angle parameters in units
                        if p > 0
                            if angle is tuple, range between angle[0], angle[1]
                            if angle is float range between -angle, angle
                            if distribution is normal, angle = 3rd SD
        p:              (float) 0-1 beroulli probability that randomness will occur
        distribution    (enum config.RnDMode) normal / uniform
        center_choice   (int, default 1) bitmask
                            1: image center
                            2: target items boudning box center
                                3: mean (1,2) ...
        center_sharpness(int: default 1)
                            0: exact
                            1: normal dist, 5  sigma
                            2: normal dist, 3, stds
                            3: normal dist, 1, std

        interpolation   (str) default 'linear'|'nearest'|'cubic'|'lanczos'
        scalefit        (float [1.0]) in 0, 1 
        units           (str) 'pi'|'radians'|'degrees'

        TODO Fix rotation seed for multiprocessing
        TODO fix single tensor list rotation
    """
    if config.VERBOSE:
        print("Rotate():")

    interpolation = check_interp(interpolation, ["nearest", "linear", "cubic", "lanczos"])
    distribution = check_distribution(distribution)
    # angle = number_in_range(angle, msg="angle")
    units = check_units(units)
    scalefit = min(1.0, max(scalefit, 0.0))

    if angle == 0:
        return data

    data, types = unfold_tensors(data, msg="rotate() input", inplace=inplace)
    tensor = data[0]

    # TODO this is kind of weird p==0 IS IT SAME BEHAVIOR IN ALL?
    if p > 0:  # probability that image will be rotated
        num = len(tensor) if independent else 1
        angles = get_random_rotation(num, angle, p, distribution)
    else:
        angles = torch.tensor([angle], dtype=torch.float32)

    dims = tensor.ndimension() - 2
    if dims == 1:
        print("cannot rotate a 1D tensor")
    assert dims == 2, " not implementeed in more or less than 2 dimensions"

    _idx = [i for i in range(len(data)) if types[i] == "tensor_list"]
    if _idx:
        target_tensor = data[_idx[0]]
        if len(_idx) > 1:
            print("only first tensor list is rotated, refactor code functional._rotate2d()")
    else:
        target_tensor = None

    data[0], target_tensor = _rotate2d(tensor, target_tensor, angles, center_choice,
                                       center_sharpness, interpolation, units, scalefit, debug)
    data[0] = check_contiguous(data[0], verbose=config.DEBUG, msg="rotate() output:, ")
    if target_tensor is not None:
        data[_idx[0]] = target_tensor

    return data

def _scalerot2d(center, angles, height, width, scalefit=1):
    """ return scale factor to avoid padding to show
    Args
        center      tensor (N,1,2) # TODO instead of permuting center ??  could build pos to match.. duh
            or no!!! center has to be (N,1,[h,w]) everywhere
        angles      tensor (N)
        height      int
        width       int
        scalefit    float (0-[1])
    """
    _n = len(center)

    _pos = torch.tensor([[0, 0], [height, 0], [0, width], [height, width]]) * torch.ones((_n, 4, 2))
    _mat = _get_rotation2(angles, num=len(center)) # == torch.inverse(_mat)
    _newpos = (_pos - center) @ _mat + center
    _overflow = (_newpos - center)/ center

    scales = _overflow.view(len(_overflow),- 1).max(dim=1)[0]
    scales.mul_(scalefit).add_(1 - scalefit)
    return scales

    # this is wrong, do scale per angle
    # return 1*(1 - scalefit) +scalefit*max(_overflow.max(), _overflow.min()*-1)

def _rotate2d(tensor, target_tensor_list, angles, center_choice, center_sharpness, interpolation, units, scalefit, debug=0):
    """ rotates a tensor and its target tensor
        target tensor in shape (dims, items)
    """
    _rotated = False
    scale = None

    _n, _c, _h, _w = tensor.shape

    center = gen_center(center_choice, tensor, target_tensor_list) # (N,2,1)
    # print(center)

    if interpolation == "lanczos" or str(tensor.device) == "cpu" and debug != 1:
        if cv2 is None:
            print("cv2 not available, reverting to linear interpolation")
            interpolation = "linear"
        else:
            scale = rotatecv2d(tensor, _convert_rotation_units(angles, units, 'd'),
                               interpolation, center, scalefit)

            _rotated = True
            if target_tensor_list is None:
                return tensor, target_tensor_list

    angles = _convert_rotation_units(angles, units, 'r')
    matrix = _get_rotation2(angles, num=_n)

    if scale is None:
        scale = _scalerot2d(center.permute(0, 2, 1).contiguous(), -1 * angles, _h, _w, scalefit)

    scalemat = scale.view(len(scale), 1, 1) * (torch.eye(2).expand_as(matrix))

    # rotates tensors with torch
    if not _rotated:
        positions = _get_positions_matrix2(tensor) # (N,2,2)
        # (N,2,2)  ((N,2,2) - (N,2,1))
        # bmm (b×n×p) = (b×n×m) (b×m×p)

        # inverse scale as one is scaling positions, not data: zoom pos in = zoom img out
        pos_rotated = torch.bmm((matrix @ torch.inverse(scalemat)), (positions - center)) + center
        __DEBUGCHECK(angles, matrix, center, positions, pos_rotated)
        tensor = _interpolate(tensor, positions, pos_rotated, interpolation)
        tensor = check_contiguous(tensor)
        __DEBUGCHECK(pos_rotated)

    # rotates targets
    target_tensor_list = rotate_targets(target_tensor_list, center=center,
                                        matrix=(matrix @ scalemat), angles=angles, size=(_h, _w))
    return tensor, target_tensor_list

def __DEBUGCHECK(*args):
    if config.DEBUG:
        for arg in args:
            config.TRANSIENT += check_tensor(arg)

def gen_center(center_choice, tensor, target_tensor_list=None, distribution=None, shift=None):
    """Returns center, torch.Tensor [N,2,1]
    Args
        center_choice:  (int: 1) -TODO  change this to enum
                1:  image center
                2:  targets center - if existing - else, look at self information?
                    3 mean
        tensor:         (torch tensor NCHW)
        target_tensor   (list of tensors)
        distribution    (config.BoxMode default: .normal) |.uniform
        shift           (tensor (2,1))

    if center outside of tensor, consider it invalid
    """
    assert len(tensor.shape) == 4, "tensor shape in (NCHW), found shape len %d"%len(tensor.shape)

    center_choice = max(min(center_choice, 3), 1)
    if target_tensor_list is None or len(target_tensor_list) == 0:
        center_choice = 1

    if center_choice == 1: # image center
        center = _get_image_center2(tensor)

    elif center_choice == 2: # center of targets
        center = targets_center(target_tensor_list)

    elif center_choice == 3: # mean between image and targets centers
        img_center = _get_image_center2(tensor)
        tgt_center = targets_center(target_tensor_list)
        # if len(angle) > 1:
        #     tgt_center = tgt_center.mean(0).unsqueeze_(0)
        center = torch.cat([img_center, tgt_center], dim=0).mean(0).unsqueeze_(0)

    if shift is not None:
        assert shift.shape == (2, 1), "shift requires shape (2,1) found %s"%str(tuple(shift.shape))
        distribution = check_distribution(distribution)

        if distribution == config.RndMode.uniform:
            center += uniform(-1, 1, center.shape, center.dtype, center.device, grad=False)*shift
        else:
            _delta = normal(0, 1, center.shape, center.dtype, center.device, grad=False)*shift
            center += normal(0, 1, center.shape, center.dtype, center.device, grad=False)*shift
    # clamp center to image boundary.
    center = torch.min(torch.max(center, torch.zeros_like(center)),
                       torch.tensor(tensor.size()[2:], dtype=center.dtype, device=center.device,
                                    requires_grad=center.requires_grad).view(1, 2, 1))
    return center

def _get_image_center2(tensor):
    """ returns center [N,2,2]
    """
    _n, _c, _h, _w = tensor.shape
    center = torch.tensor([[_h/2], [_w/2]], dtype=tensor.dtype, device=tensor.device,
                          requires_grad=False)
    center = torch.stack([center for i in range(_n)])
    return center

def _get_rotation2(angle, num=None):
    """ returns rotation matrix [N,2,2]
        angle   (torch.tensor) [N]
    """
    # expand to full affine matrix
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # if debug == 3:
    #     matrix = torch.stack([torch.stack([cos, -sin, sin, cos])])
    #     matrix = matrix.view(4, cos.numel()).permute(1, 0).view(cos.numel(), 2, 2).contiguous()
    # else:
    matrix = torch.stack([torch.stack([cos, -sin], dim=1), torch.stack([sin, cos], dim=1)], dim=2)

    if num is not None and num > len(angle):
        matrix = torch.cat([matrix for i in range(num)], dim=0)
    return matrix

def _get_positions_matrix2(tensor):
    """ returns a tensor shape, [N,2,2]
    """
    _n, _c, _h, _w = tensor.shape
    _y = torch.arange(0.0, _h, dtype=tensor.dtype, device=tensor.device, requires_grad=False)
    _x = torch.arange(0.0, _w, dtype=tensor.dtype, device=tensor.device, requires_grad=False)

    grid = torch.meshgrid((_y, _x))
    grid[0].unsqueeze_(0)
    grid[1].unsqueeze_(0)

    positions = torch.cat(grid, 0).view(2, -1)
    positions = torch.stack([positions for i in range(_n)])
    positions = check_contiguous(positions)
    return positions

def _interpolate(tensor, positions, pos_rotated, interpolation="linear"):
    """
    maybe superseed by
    torch.einsum('bn,anm,bm->ba', l, A, r)
    or torch.nn.functional.bilinear
    """
        # torch.Size([2, 589824]) torch.Size([2, 589824])
        # torch.Size([1, 2, 589824]) torch.Size([1, 2, 589824])

    #out img
    _n, _c, _h, _w = tensor.shape
    out_tensor = torch.zeros([_n, _c, _h, _w], dtype=tensor.dtype, device=tensor.device)

    # temp img: pad maxh & maxy
    # filter rotated indices < 0 > hmax,wmax

    # if config.DEBUG:

    for i, pos_rot in enumerate(pos_rotated):
        _ge = torch.ge(pos_rot, torch.tensor([[0], [0]], dtype=pos_rotated.dtype,
                                             device=tensor.device)).prod(0)
        _lt = torch.lt(pos_rot, torch.tensor([[_h], [_w]], dtype=pos_rotated.dtype,
                                             device=tensor.device)).prod(0)
        non_zero = (_ge*_lt).nonzero().squeeze()
        assert non_zero.is_contiguous(), "indices should be contiguous"
        _img = torch.zeros([1, _c, _h+1, _w+1], dtype=tensor.dtype, device=tensor.device)
        _img[:, :, :_h, :_w] = tensor[i:i+1]
        _img[:, :, _h, :_w] = _img[:, :, _h-1, :_w]
        _img[:, :, :_h, _w] = _img[:, :, :_h, _w-1]

        if interpolation == "linear":
            # alpha blends
            alphas = pos_rot%1
            bot = pos_rot.floor().to(dtype=torch.int64, device=tensor.device)
            top = pos_rot.ceil().to(dtype=torch.int64, device=tensor.device)
            # print('alphas', alphas.shape)
            # print('bot', bot.shape)
            # print('top', top.shape)

            out_tensor[i:i+1] = _linear(alphas, non_zero, top, bot, out_tensor[i:i+1], _img,
                                        positions[i].to(dtype=torch.int64))

        elif interpolation == "nearest":
            out_tensor[i:i+1] = _nearest(non_zero, pos_rot.round().to(dtype=torch.int64),
                                         out_tensor[i:i+1], _img,
                                         positions[i].to(dtype=torch.int64))

    out_tensor = check_contiguous(out_tensor, verbose=config.DEBUG, msg="_interpolate output:, ")

    return out_tensor

def _linear(alphas, non_zero, top, bot, out_tensor, _img, positions):
    # 00*(1-dh)*(1-dw) + 01*(1-dh)*dw + 10*dh*(1-dw) + 11*dh*dw
    _a0 = alphas[0, non_zero]
    _a1 = alphas[1, non_zero]
    _ia0 = 1 - alphas[0, non_zero]
    _ia1 = 1 - alphas[1, non_zero]
    _00 = _img[:, :, bot[0, non_zero], bot[1, non_zero]] * _ia0 * _ia1
    _01 = _img[:, :, bot[0, non_zero], top[1, non_zero]] * _ia0 * _a1
    _10 = _img[:, :, top[0, non_zero], bot[1, non_zero]] * _a0 * _ia1
    _11 = _img[:, :, top[0, non_zero], top[1, non_zero]] * _a0 * _a1
    out_tensor[:, :, positions[0, non_zero], positions[1, non_zero]] = _00 + _01 + _10 + _11

    return out_tensor

def _nearest(non_zero, pos_rotated, out_tensor, _img, positions):
    _nz = non_zero
    _p = positions
    _pr = pos_rotated
    out_tensor[:, :, _p[0, _nz], _p[1, _nz]] = _img[:, :, _pr[0, _nz], _pr[1, _nz]]
    return out_tensor

def _convert_rotation_units(angle, inunit, outunit):
    """ maps between degrees, pis and radians
        angle   (torch.tensor) [N]
        inunit,
        outnit  (str) in ["radians", "degrees", "pi"]
    """
    if outunit == inunit:
        return angle
    if outunit == 'r':
        if inunit == 'd':
            return math.pi * (angle%360)/180 # deg to rad
        return angle * math.pi # pi to rad

    if outunit == 'd':
        if inunit == 'r':
            return  angle * 180 / math.pi # rad to deg
        return angle * 180 # rad to pi

    if outunit == 'p':
        if inunit == 'r':
            return angle / math.pi # pi to rad
        return angle / 180 # pi to deg


def flip(data, px, py, independent, inplace=False):
    """ Flips image in x and or y given probability
    # TODO inplace
    """
    data, types = unfold_tensors(data, msg="flip() input", inplace=inplace)
    tensor = data[0]
    _idx = [i for i in range(len(data)) if types[i] == "tensor_list"]

    _n, _c, _h, _w = tensor.size()
    size = _n if independent else 1
    device = tensor.device

    _p = bernoulli((px, py), (size, 2), torch.uint8, device, grad=False)

    dims = []
    for i, _ in enumerate(_p):
        _d = [3] if _p[i, 0] else []
        _d = _d + [2] if _p[i, 1] else _d
        dims.append(_d)
    if not dims:
        return data

    #print(f"flips {dims}")
    if size == 1: # batch size == 1 or all flips are identical
        if dims[0]:
            tensor[:] = tensor.flip(dims=dims[0])
    else:
        for i in range(size):
            if dims[i]:
                tensor[i:i+1] = tensor[i:i+1].flip(dims=dims[i])
    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="flip output:, ")

    for i in range(1, len(data)):
        if types[i] == "tensor_list":
            data[i] = flip_targets(data[i], _p, _h, _w)
    return data

##
# laplacian pyramids
#
#   .laplace_pyramid()
#   .gauss_up()
#   .gauss_dn()
def laplace_pyramid(data, steps, pad_mode, min_side, augments=None, pshuffle=0.0, **kwargs):
    """Applies a gaussian convolution, normalizes to maintain image statististics
        2D

    Args:
        img   (Tensor): format NCL, NCHW, NCHWD
        steps    (int): number of steps, 0-n (Default: 0, limt reached with min_side
        pad_mode (int): 0, trims original image to evenly divisible size
                        1, pads (reflective to keep statistics) to evenly divisible size
        min_side (int): 2-m, clamps steps if pyramid step <= min side

    Returns:
        expanded image, out(h,w) = 2*in(h,w)
    """
    stride = 2

    if config.VERBOSE:
        print("LaplacePyramid():")

    data, types = unfold_tensors(data, msg="laplace_pyramid input:, ", inplace=True)
    tensor = data[0]

    assert min_side > 1, "smallest minimum side = 2, got %d"%min_side
    if not isinstance(steps, int) or steps < 0:
        raise TypeError('Got inappropriate steps arg: {}'.format(steps))

    dims = tensor.ndimension() - 2
    assert dims == 2, "laplace pyramid for dims 1 and 3 not implemented"

    # operate all channels for entire batch, independently per channel
    channels = tensor.size()[1] * tensor.size()[0]

    # Down Convolution:
    gauss_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, stride=stride,
                           groups=channels, bias=False)
    gauss_kernel = _make_gaussian_kernel(tensor.dtype, channels, dims, tensor.device)
    gauss_conv.weight.data = gauss_kernel

    # Up Convolution = Unpool(stride) -> Gaussian Convolve Stride = 1: blur
    up_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, stride=1, groups=channels,
                        bias=False)
    up_conv.weight.data = _make_gaussian_kernel(tensor.dtype, channels, dims, tensor.device)*4

    # Unpool
    unpool = nn.MaxUnpool2d(stride, stride=stride)

    # Compute padding and crop (pad_mode=1) or pad (pad_mode=0) to size divisible by 2
    # Adjust input image size
    _n, _c, _h, _w = tensor.size()
    _h0, _h1, _w0, _w1, steps = compute_pad((_h, _w), steps=steps, min_side=min_side,
                                            pad_mode=pad_mode)
    if pad_mode == 0: #crop
        _hsz = _h + _h1
        _wsz = _w + _w1
        tensor = tensor[:, :, -_h0:_hsz, -_w0:_wsz]
    else: #pad
        tensor = nn.ReflectionPad2d((_w0, _w1, _h0, _h1))(tensor)

    if config.VERBOSE:
        print("original size", _n, _c, _h, _w)
        print("opad", _h0, _h1, _w0, _w1, steps)
        print(tensor.requires_grad, tensor.device)

    # Compute pyramid address coordinates
    _n, _c, _h, _w = tensor.size()
    coords = build_coords(_h, _w, steps=steps, as_torch=True, device=tensor.device)
    # coords[0[[x,y],...,steps],1[,2[],3[],4[]]
    # P0 from 0, to 2
    # P1 from 1, to 3
    # P2 from 2, to 4
    # 0.........0a...0b.   &c
    # .         .    1b .
    # .    P0  1a....2a.3b.4b
    # .         .    .     .
    # 1.........2....3a....4a
    # .         .          .
    # .    P1   .    P2    .
    # .         .          .
    # ..........3..........4
    if config.DEBUG:
        _size = check_tensor(tensor)/2**20
        _msg = "Lpyr Padding _img: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)

    # set augments
    _normed = None
    _augments = None
    if augments is not None:
        # print(type(augments), augments)
        _augments = [a for a in augments]

        # mean center up and down pyramids only not laplacian
        for aug in _augments:
            if aug.__class__.__name__ in ("Normalize", "MeanCenter"):
                _normed = _augments.pop(_augments.index(aug))
                # print(_normed)
                break

        # shuffle augments
        if _augments:
            try:
                if pshuffle > 0:
                    pshuffle = min(1, pshuffle) #?
                    if pshuffle < 1: # cointoss
                        pshuffle = bernoulli(pshuffle, 1, dtype=torch.uint8, device="cpu",
                                             grad=False)[0]
                    if pshuffle:
                        shuffle(_augments)
            except:
                print("augments:", type(_augments))
                for aug in _augments:
                    print(aug.__class__.__name__)
        else:
            _augments = None
        # print(augments)

    # Create out vector / 2x, y size - half vertical size of final out vector
    out = torch.cat((tensor, torch.ones_like(tensor)/2.), dim=3)

    if config.VERBOSE:
        print((torch.ones_like(tensor)/2).requires_grad)
        print(out.requires_grad, out.device)

    if config.DEBUG:
        _size = check_tensor(out)/2**20
        _msg = "Lpyr out: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)
        print("     out.is_contiguous()", out.is_contiguous())

    # Pyramids
    # 1. Compute Pyramid down sequentially
    for i in range(steps): # pyramid down
        tensor = _gauss_down(tensor, gauss_conv)
        _pos = coords[0, i+1]
        _to = coords[2, i+1]
        out[:, :, _pos[0]:_to[0], _pos[1]:_to[1]] = tensor

    # 2. Compute Pyramid Up in one step
    _up = _gauss_up(out[:, :, :coords[1, 1, 0], _w:], up_conv, unpool)

    if config.DEBUG:
        _size = check_tensor(_up)/2**20
        _msg = "Lpyr up: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)

    # 3. Laplacian: down pyramid - up pyramid
    if _normed:
        _lap = (out - _up)
    else: # normalize laplacian to visible range if pyramid not normalized
        _lap = _norm_to_range((out - _up), 0.0, 1.0, excess_only=False,
                              independent=False, per_channel=False)

    if config.DEBUG:
        _size = check_tensor(_lap)/2**20
        _msg = "Lpyr diff: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)
        print("     lap.is_contiguous()", _lap.is_contiguous())

    if config.VERBOSE:
        print("padded size", _n, _c, _h, _w)
        print("coords", coords)

    # Reorder pyramid
    # upper half of image
    # 4. add center pyramid (P1) to out tensor
    for i in range(1, len(coords[0])):
        # pyramid created by gaussian upscales, if no augments exists
        if _augments is None:
            out[:, :, coords[1, i, 0]:coords[3, i, 0],
                coords[1, i, 1]:coords[3, i, 1]] = _up[:, :, coords[0, i, 0]:coords[2, i, 0],
                                                       coords[0, i, 1]:coords[2, i, 1]]
        # cycle thru agumentations and place augmentations of the sharp image
        # force inplace=False to ensure that each augmentation is independent
        else:
            j = i%len(_augments)
            try:
                _out = out[:, :, coords[0, i, 0]:coords[2, i, 0],
                           coords[0, i, 1]:coords[2, i, 1]]#.clone().detach()
                out[:, :, coords[1, i, 0]:coords[3, i, 0],
                    coords[1, i, 1]:coords[3, i, 1]] = _augments[j](_out)[0]
            except:
                print("Failed on Augment:", _augments[j].__class__.__name__)
                print(_augments[j])
                print(" into", out[:, :, coords[1, i, 0]:coords[3, i, 0],
                                   coords[1, i, 1]:coords[3, i, 1]].shape)
                print(" used", out[:, :, coords[0, i, 0]:coords[2, i, 0],
                                   coords[0, i, 1]:coords[2, i, 1]].shape)
                raise

    # 5. mean center upper half
    if _normed:
        out = _normed(out)[0]

    # 6. add laplacian data to right pyramid (P2)
    for i in range(1, len(coords[0])):
        # vertical pyramid, laplacian
        out[:, :, coords[2, i, 0]:coords[4, i, 0],
            coords[2, i, 1]:coords[4, i, 1]] = _lap[:, :, coords[0, i, 0]:coords[2, i, 0],
                                                    coords[0, i, 1]:coords[2, i, 1]]

    # 7. Vertical stack
    # out is the upper half
    # mean center augmented lower left quadrant (P1)
    # bottom of laplacian is lower right quadrant (P2)
    # since we no longer need
    if _augments is not None:
        # original image
        tensor = out[:, :, coords[0, 0, 0]:coords[2, 0, 0], coords[0, 0, 1]:coords[2, 0, 1]]
        if _normed is not None: #_up[:, :, :, :w]
            out = torch.cat((out, torch.cat((_normed(_augments[0](tensor))[0],
                                             _lap[:, :, :, :_w]), dim=3)), dim=2)
        else:
            out = torch.cat((out, torch.cat((_augments[0](tensor)[0],
                                             _lap[:, :, :, :_w]), dim=3)), dim=2)
    else:
        out = torch.cat((out, torch.cat((_up[:, :, :, :_w], _lap[:, :, :, :_w]), dim=3)), dim=2)

    # 8. Mask laplacian grid to prevent activations caused by convolutions overlapping grid
    # TODO replace with computed mask in model.
    if _normed:
        for i, _c in enumerate(coords[1]):
            out[:, :, _c[0]-1:_c[0]+1, _c[1]:] = 0
        for i, _c in enumerate(coords[3]):
            out[:, :, : _c[0], _c[1]-2:_c[1]+2] = 0

    if config.DEBUG:
        _size = check_tensor(out)/2**20
        _msg = "Lpyr out: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)

    # 9.if targets exist, propagate
    _offset = torch.tensor([_h0, _w0], dtype=out.dtype, device=out.device, requires_grad=False)

    for i in range(1, len(data)):
        if types[i] == "tensor_list":
            data[i] = gauss_pyramid_expanded_targets(data[i], coords, _offset)

    # 10 cleanup
    data[0] = check_contiguous(out, verbose=config.DEBUG, msg="laplace_pyramid output:, ")
    del _up
    del _lap
    del tensor
    del coords, _offset
    if config.FORCE_CLEANUP and data[0] .is_cuda:
        torch.cuda.empty_cache()
        if config.DEBUG:
            _msg = "delete cache"
            mem.report(memory=config.MEM, msg=_msg)

    if config.DEBUG:
        _size = check_tensor(data[0] )/2**20
        _msg = "Lpyr out: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)

    return data

def gauss_down(data):
    """Applies a gaussian convolution, normalizes to maintain image statististics

    Args:
        img (Tensor): format NCL, NCHW, NCHWD

        TODO dims should be stored in State
            separate 1,2,3d conv
    """
    if config.VERBOSE:
        print("GaussDn():")
    data, types = unfold_tensors(data, msg="gaussdn input:, ", inplace=True)
    tensor = data[0]

    # if target_tensor is not None:
    #     print("target tensor not yet implemented for gaussian pyramid")

    dims = tensor.ndimension() - 2
    assert dims == 2, "gauss pyramid for dims 1 and 3 not implemented"

    # operate all channels for entire batch, independently per channel
    channels = tensor.size()[1] * tensor.size()[0]
    gauss_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, stride=2, groups=channels,
                           bias=False)
    gauss_conv.weight.data = _make_gaussian_kernel(tensor.dtype, channels, dims, tensor.device)

    data[0] = _gauss_down(tensor, gauss_conv)
    return data

def _gauss_down(tensor, gauss_conv):
    return _norm_to_range(gauss_conv(tensor), tensor.min(), tensor.max(), excess_only=False,
                          independent=False, per_channel=False)

def gauss_up(data):
    """Expands image then applies gaussian convolution*4 , normalizes to maintainimage statististics

    Args:
        img (Tensor): format NCL, NCHW, NCHWD

        TODO dims should be stored in State
            separate 1,2,3d conv
    """
    if config.VERBOSE:
        print("GaussUp():")

    data, types = unfold_tensors(data, msg="gaussup input:, ", inplace=True)
    tensor = data[0]

    dims = tensor.ndimension() - 2
    assert dims == 2, "gauss pyramid for dims 1 and 3 not implemented"

    # if target_tensor is not None:
    #     print("target tensor not yet implemented for gaussian pyramid")

    # operate all channels for entire batch, independently per channel
    channels = tensor.size()[1] * tensor.size()[0]

    # Up Convolution = Unpool(stride=2) -> Gaussian Convolve Stride = 1: blur
    gauss_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, stride=1, groups=channels,
                           bias=False)
    gauss_conv.weight.data = _make_gaussian_kernel(tensor.dtype, channels, dims, tensor.device) * 4

    # Unpool
    unpool = nn.MaxUnpool2d(2, stride=2)
    data[0] = _gauss_up(tensor, gauss_conv, unpool)
    return data

def _gauss_up(tensor, gauss_conv, unpool):
    stride = 2
    # build index grid for unpool operation
    _n, _c, _h, _w = tensor.shape
    grid = ([([1,]+[0]*(stride-1))*_w] + [([0,]+[0]*(stride-1))*_w]*(stride-1))*_h
    mask = torch.tensor(grid, dtype=torch.int64, device=tensor.device, requires_grad=False).view(-1)
    indices = mask.nonzero().reshape(_n, 1, _h, _w) + torch.zeros([_n, _c, _h, _w],
                                                                  dtype=torch.int64,
                                                                  device=tensor.device,
                                                                  requires_grad=False)
    return _norm_to_range(gauss_conv(unpool(tensor, indices)), tensor.min(), tensor.max(),
                          excess_only=False, independent=False, per_channel=False)

def _make_gauss_weights(dtype, dims, device, kernel_size=5):
    """ makes size 5 kernel of 1, 2 or 3d weights
        cannonical gaussian pyramid, approximated kernel size 5x5
    """
    assert dims in (1, 2, 3), "dimension not undestood, only 1, 2 or 3d data suported"
    if kernel_size == 5:
        _gk = torch.tensor([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=dtype, device=device)
    else:
        _gk = _general_gauss_weights(kernel_size=kernel_size, dtype=dtype, device=device)

    if dims == 1:
        return _gk # 1d gaussian

    gk2 = torch.ger(_gk, _gk)
    if dims == 2:
        return gk2 # 2d gaussian

    # 3d gaussian
    return gk2.unsqueeze(2)*(_gk.unsqueeze(0).unsqueeze(0))

def _general_gauss_weights(kernel_size, dtype, device):
    """ Generalization of the canonical [0.0625, 0.25, 0.375, 0.25, 0.0625] to any kernel size
        sum = 1
        at ksize = 5 # [0.05532584 0.24428895 0.40077052 0.24428895 0.05532584]
    """
    _size = (kernel_size-1)/2
    _sd = 0.5025 *_size
    _m = 0.0
    _v = _sd**2
    _x = torch.arange(-_size, _size+1, dtype=dtype).to(device=device)
    _g = torch.exp(-1.0*(_x-_m)**2.0/(2*_v)).div(torch.sqrt(torch.tensor(2.0*math.pi*_v,
                                                                         dtype=dtype,
                                                                         device=device)))
    return _g.div(_g.sum())

def _make_gaussian_kernel(dtype, channels, dims, device):
    """ makes gaussian weights for a convolution operation"""
    gauss_w = _make_gauss_weights(dtype, dims, device)
    return torch.unsqueeze(torch.stack([gauss_w for i in range(channels)]), 1)


def get_image_scales(size, target_size=512, num_scales=4):
    """
        size        (tuple) input image size (h,w)
        target_size (int [512])
        num_scales  (int [4])
    """
    scale = np.max(np.array(size)/target_size)
    min_scale = np.array(size)/scale
    #scale = np.ceil(scale).astype(int)

    return np.round(np.array([min_scale*(i+1) for i in range(num_scales)])).astype(int)



def scale_stack(data, sizes=None, target_size=512, interpolation="cubic", num_scales=4,
                verbose=False, mode="xyhw"):
    r"""Stack scaled tensors:
    output: list(tensor,...) tensor format NCWH

    Args:
        data (list,tuple, (tensor,)) tensor: Image, 1, 2 or 3d data to be resized
            accepted formats CHW, NCHW

        sizes       (ndarray [None]) stack of sizes

        target_size (int [512]), length of maximum dimension

        interpolation (str, optional): Desired interpolation.
            Default is ``cubic``, options ``nearest, linear, area, cubic``

    Returns:
        torch.tensor: Resized image.
        target_tensor_list
        label_list
    """
    if verbose:
        print("ScaleStack()")

    data, types = unfold_tensors(data, msg="scale stack", inplace=True)
    tensor = data[0]

    if tensor.ndimension() == 3:
        tensor = tensor.view(1, *tensor.shape)
    _msg = "tensors expected in NCHW or CHW, got shape: %s"%(str(tuple(tensor.shape)))
    assert tensor.ndimension() == 4, _msg

    #size = check_size(size)
    _valid_interp = ["nearest", "linear", "area", "cubic"]
    _msg = "interpolation not in %s, found %s"%(str(_valid_interp), interpolation)
    assert interpolation in _valid_interp, _msg

    align_corners = None
    if interpolation in ("linear", "cubic"):
        align_corners = False
        interpolation = "bi"+interpolation

    if sizes is None:
        sizes = get_image_scales(tensor.shape[2:], target_size=target_size, num_scales=num_scales)
    if isinstance(sizes, (list, tuple)):
        sizes = np.array(sizes).reshape(-1, 2).astype(int)
    elif isinstance(sizes, torch.Tensor):
        sizes = sizes.cpu().clone().detach().numpy().reshape(-1, 2).astype(int)

    assert isinstance(sizes, np.ndarray), "expected ndarray got <%s>"%str(type(sizes))
    _msg = "expected ndarray shape(-1,2) got %s"%str(tuple(sizes.shape))
    assert len(sizes.shape) == 2 and sizes.shape[1] == 2, _msg

    stacks = {i:[] for i in range(len(data))}
    for i, _size in enumerate(sizes):
        _img = nn.functional.interpolate(tensor, size=tuple(_size), mode=interpolation,
                                         align_corners=align_corners).clamp(tensor.min(),
                                                                            tensor.max())
        if not i:
            stacks[0].append(_img)

            for j in range(1, len(data)):
                if type[j] == "tensor_list":
                    stacks[j].append(scale_targets(data[j], _size, mode=mode))
                else:
                    stacks[j].append(data[j])
        else:
            positions = np.array([(y, x) for y in range(i+1) for x in range(i+1)]).astype(np.int)
            for _, pos in enumerate(positions):
                _y, _x = sizes[0] * pos
                _h, _w = sizes[0] * (pos + 1)
                stacks[0].append(_img[..., _y:_h, _x:_w])
                for j in range(1, len(data)):
                    if type[j] == "tensor_list":
                        stacks[k].append(crop_targets(sizes[0]*pos, sizes[0]*(pos + 1),
                                                      data[j], cutoff=0.5, mode=mode))
                    else:
                        stacks[j].append(data[j])

    for i in stacks:
        if torch.is_tensor(stacks[i][0]):
            stacks[i] = torch.cat(stacks[i], dim=0)
            stacks[i] = check_contiguous(stacks[i], verbose, msg="tensor_stack()interpolate()")

    return list(stacks.values())

def tensor_slice(tensors, crop_start, crop_size):
    """ crop tensor
    """
    dtype = tensors.dtype
    device = tensors.device

    _n, _c, _h, _w = tensors.shape
    # _ch, _cw = view(crop_size, -1).tolist()

    # out = torch.ones((_n, _c, _ch, _cw), dtype=dtype, device=device, requires_grad=grad)/2.0

    _max_size = torch.tensor((_h, _w), device=crop_start.device, dtype=crop_start.dtype,
                             requires_grad=False)

    crop_start = torch.max(crop_start, torch.zeros_like(crop_start))

    _to = crop_start.add(crop_size)
    _dif = torch.max(_to - _max_size, torch.zeros_like(_to))
    crop_start.sub_(_dif)
    _to.sub_(_dif)
    out = tensors[:, :, crop_start[0]:_to[0], crop_start[1]:_to[1]]

    # cropped_targets = None
    # if targets is not None:
    #     cropped_targets = []
    #     for target_tensor in targets:
    #         cropped_target_tensor, _ = crop_targets(crop_start, crop_size,
    #                                                 target_tensor.clone(), cutoff=0.5)
    #         cropped_targets.append(cropped_target_tensor)

    return out#, cropped_targets

def tensor_list_slice(targets, crop_start, crop_size, max_size):
    crop_start = torch.max(crop_start, torch.zeros_like(crop_start))
    _to = crop_start.add(crop_size)
    _dif = torch.max(_to - max_size, torch.zeros_like(_to))
    crop_start.sub_(_dif)
    out = []
    for target_tensor in targets:
        cropped_target_tensor, _ = crop_targets(crop_start, crop_size,
                                                    target_tensor.clone(), cutoff=0.5)
        out.append(cropped_target_tensor)
    return out
"""
FIX  when there are no targets
tensor_slice(tensor[i:i+1],target_tensor_list[i:i+1],_crop_start, crop_size)
"""
def transpose(vector):
    """ transpose ndarray or tensor
    """
    if torch.is_tensor(vector):
        return vector.t()
    return vector.T

def view(vector, shape):
    """ reshape ndarray or tensor
    """
    if torch.is_tensor(vector):
        return vector.view(shape)
    return vector.reshape(shape)

def crop(data, crop_start, crop_size):
    """
    Args
        data        (tuple) tensor, targets, labels
        crop_start  (tensor) shape (N,2,M) N: batch items, M: crop bifucations to new batch elems
        crop_size   (tensor) shape (2)
    """
    data, types = unfold_tensors(data, msg="crop input:, ", inplace=True)
    tensor = data[0]

    _msg = "tensor shape, %s needs crop_start (%d,2,M) found %s"%(str(tuple(tensor.shape)),
                                                                  len(tensor),
                                                                  str(tuple(crop_start.shape)))
    assert len(crop_start.shape) == 3, _msg
    assert crop_start.shape[1] == 2, _msg
    assert crop_start.shape[0] == len(tensor), _msg


    _msg = "crop size needs to be shape (2), found %s"%str(tuple(crop_size))
    assert len(crop_size.shape) == 1 and len(crop_size) == 2, _msg

    out = [[] for i in range(len(data))]

    _batchcrop = True   # all crops for an entire batch are equal
    for i in range(len(crop_start) - 1):
        if torch.ne(crop_start[i], crop_start[i+1]).any():
            _batchcrop = False
            break

    _n, _c, _h, _w = tensor.shape
    max_size = torch.tensor((_h, _w), device=crop_start.device, dtype=crop_start.dtype,
                            requires_grad=False)

    if _batchcrop:
        crop_start = crop_start[0]
        for i, _crop_start in enumerate(transpose(crop_start)):

            _crop_tensor = tensor_slice(tensor, _crop_start, crop_size)
            out[0].append(_crop_tensor)

            for j in range(1, len(data)):
                if types[j] == "tensor_list":
                    out[j].append(tensor_list_slice(data[j], crop_start, crop_size, max_size))
                else:
                    out[j].append(data[j])

            if config.LOG_VALS:
                print(f"crop from: {crop_start.tolist()} to: {(crop_start + crop_size).tolist()}")
    else:
        for i, _sub_crop in enumerate(crop_start):    # batch elements
            for j, _crop_start in enumerate(_sub_crop.t()):
                _tlist = None if not (isinstance(target_tensor_list,
                                                 list)) else target_tensor_list[i:i+1]
                if config.DEBUG:
                    print(f"  nobatch :{_crop_start},{crop_size}, tensor: {tensor.shape}")
                assert j == 0, "fix subcrop case, crop %s"%str(crop_start.shape)

                _crop_tensor = tensor_slice(tensor[i:i+1], _crop_start, crop_size)
                out[0].append(_crop_tensor)

                for k in range(1, len(data)):
                    if types[k] == "tensor_list":
                        out[k].append(tensor_list_slice(data[k][i:i+1], crop_start, crop_size, max_size))
                    else:
                        out[k].append(data[j][i:i+1])


                if config.LOG_VALS:
                    print(f"crop from: {_crop_start.tolist()} to: {(_crop_start + crop_size).tolist()}")

    for i in range(len(out)):
        if torch.is_tensor(out[i][0]):
            out[i] = torch.cat(out[i])
            out[i] = check_contiguous(out[i], verbose=config.DEBUG, msg="crop output:, ")

    return out

def gen_size(tensor, size=None, contract=0.0, as_tensor=True):
    """generate size tensor based on image tensor
    Args
        tensor      (torch.Tensor)
        size        (int, tuple of ints), default None
        contract    (float 0-0.5), percentage contraction if size is None
        as_tensor   (bool)
    Returns
        tensor shape (dims)
    """
    if config.LOG_VALS:
        print("gen_size()", tensor.shape, "size", size)

    size = size if size is not None else int(min(tensor.size()[2:])*(1 - contract))
    size = int_iterable_in_range(size, low=4, size=tensor.ndimension()-2)

    _msg = "size must be list, tuple, ndarray or tensor, found %s"%type(size)
    assert isinstance(size, (list, tuple, np.ndarray, torch.Tensor)), _msg
    if as_tensor:
        if isinstance(size, (list, tuple)):
            size = torch.tensor(size, dtype=tensor.dtype, device=tensor.device, requires_grad=False)
        elif isinstance(size, np.ndarray):
            size = torch.from_numpy(size).to(dtype=tensor.dtype, device=tensor.device)
            size.requires_grad = False
    else:
        if isinstance(size, np.ndarray):
            size = size.tolist()
        elif torch.is_tensor(size):
            size = size.cpu().clone().detach().tolist()

    if config.LOG_VALS:
        print("gen_size(), size resolved to", str(size))

    return size

def squeeze_crop(data, size, ratio, interpolation):
    """
    functional for transforms.SqueezeCrop()
    sizes:
        arg size: int of final side length
        im_size: current size of the image
        crop_size: intermediate size of crop
    """
    data, types = unfold_tensors(data, msg="crop input:, ", inplace=True)
    tensor = data[0]

    device = tensor.device

    t_size = torch.tensor(tensor.shape[2:], device=device).long()
    _diff = t_size[0] - t_size[1]
    start = torch.clamp(torch.tensor([_diff, -_diff]), 0,
                        t_size.max()).mul(ratio/2.).long()

    crop_size = t_size.sub(start)
    start = start.view(2, 1).mul(torch.ones((len(tensor), 2, 1),
                                            dtype=start.dtype, device=start.device))

    data = crop(data, start, crop_size)
    return resize(data, size, interpolation=interpolation, square=True)


def _get_resize_crop_params(height, width, scale, ratio, p):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        height, width of tensor
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = height * width
    if p:
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return torch.tensor([i, j], dtype=torch.int64), torch.tensor([h, w], dtype=torch.int64)

            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(ratio):
                w = width
                h = int(round(w / min(ratio)))
            elif in_ratio > max(ratio):
                h = height
                w = int(round(h * max(ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2
    else:
        w = h = min(height, width)
        i = (height - h) // 2
        j = (width - w) // 2

    return torch.tensor([i, j], dtype=torch.int64), torch.tensor([h, w], dtype=torch.int64)

def resize_crop(data, size, scale, ratio, interpolation, p):
    """  functional for ResizeCrop - tensor rewrite of torchvision RandomResizeCrop
        similar to center_crop
    """
    data, types = unfold_tensors(data, msg="crop input:, ", inplace=True)
    tensor = data[0]
    _n, _c, _h, _w = tensor.shape

    crop_start, crop_size = _get_resize_crop_params(_h, _w, scale, ratio, p)
    data = crop(data, crop_start.view(1, -1, 1), crop_size)

    data = resize(data, size=torch.tensor(size, dtype=torch.torch.int64),
                  interpolation=interpolation)
    return data



    # _datas = []
    # for i, _s in enumerate(tensor):
    #     print()
    #     crop_start, crop_size = _get_resize_crop_params(_h, _w, scale, ratio, p)
    
    #     _data = [tensor[i:i+1]]
    #     for j in range(1, len(data)):
    #     #     if types[j] == "tensor_list":
    #     #         _data.append([data[j][i:i+1]]) # or data.append(data[j][i])
    #     #     elif types[j] == "Tensor":
    #     #         _data.append([data[j][i:i+1]])

    #     # for j in range(1, len(data)):
    #         if types[j] == "tensor_list":
    #             _data.append(data[j][i])
    #             # _target_tensor = data[j][i]
    #         elif types[j] in ("tensor", "Tensor"):
    #             _data.append(data[j][i:i+1])
    #         else:
    #             _data.append(data[j])

    #     _data = crop(_data, crop_start.view(1, -1, 1), crop_size)
    #     _data = resize(_data, size=torch.tensor(size, dtype=torch.torch.int64),
    #                    interpolation=interpolation)
    # #     _datas.append(_data)
    # # if len(_datas) > 1:
    # #     _data, types = refold_tensors(_datas, types)
    # return _data
    
    #, types
        #_datas.append(_data)

    # print("RC OUT")
    # for i, _d in enumerate(_datas):
    #     _sz = ""
    #     if torch.is_tensor(_d):
    #         _sz = _d.shape
    #     elif isinstance(_d, (tuple, list)):
    #         _sz = len(_d)
    #     print(type(_d), types[i], _sz)

    # return refold_tensors(_data, types)

    # out = [torch.cat([_data[0] for _data in _datas])]

    # out[0] = check_contiguous(out[0], verbose=config.DEBUG, msg="center_crop output:, ")
    # for j in range(1, len(data)):
    #     if types[j] in ("tensor", "Tensor"):
    #         out += [torch.cat(_datas[j])]
    #     elif types[j] == "tensor_list":
    #         _tensorlist = []
    #         for i in range(len(_datas[j])):
    #             _tensorlist += _datas[j][i]
    #         out += _tensorlist
    #     else:
    #         out += data[j]
    # return out

def center_crop(data, size, center_choice, shift, distribution, zoom, zdistribution, zskew):
    """
    functional for transforms.Center_Crop
    """
    if config.VERBOSE:
        print("CenterCrop:")

    data, types = unfold_tensors(data, msg="crop input:, ", inplace=True)
    tensor = data[0]

    distribution = check_distribution(distribution)
    zdistribution = check_distribution(zdistribution)

    _max_size = None

    if size is None and zoom > 0:
        frameinfo = getframeinfo(currentframe())
        msg = "Center_Crop(zoom=) has no effect unless size is set to target size"
        warnings.showwarning(msg, UserWarning, frameinfo.filename, frameinfo.lineno)
        zoom = 0

    # if zoom > 0:
    #     mean = 1 - zoom * (1 - zskew)
    #     if not isinstance(size, int):
    #         mean = [mean for i in range(len(size))]

    #     _max_size = gen_size(tensor).unsqueeze_(-1)
    #     _size = gen_size(tensor, size, contract=0.1)

    #     try:
    #         if zdistribution == config.RndMode.normal:
    #             _dist = bound_normal(mean, a=1-zoom, b=1, stds=3, samples=len(tensor),
    #                                  dtype=tensor.dtype, device=tensor.device, grad=False)
    #         else:
    #             _dist = uniform(a=1-zoom, b=1, samples=len(tensor), dtype=tensor.dtype,
    #                             device=tensor.device, grad=False)
    #     except:
    #         print(f"mean {mean}, a(1-zoom){1-zoom}, b {1}")
    #         raise ValueError

    #     #crop, then rescale
    #     _sizes = _dist.mul(_max_size).add((1 - _dist).mul(_size.unsqueeze_(-1)))
    #     _shift = _sizes.clone().detach().mul(shift)

    #     if config.LOG_VALS:
    #         for i, _d in enumerate(_dist):
    #             print(f"center_crop zoom: {_d} size: {_sizes[i]} shift: {_shift[i]}")

    #     _datas = []

    #     for i, _s in enumerate(tensor):
    #         _data = [tensor[i:i+1]]
    #         _target_tensor = None
    #         for j in range(1, len(data)):
    #             if types[j] == "tensor_list":
    #                 _data.append(data[j][i])
    #                 _target_tensor = data[j][i]
    #             elif types[j] in ("tensor", "Tensor"):
    #                 _data.append(data[j][i:i+1])
    #             else:
    #                 _data.append(data[j])

    #         _center = gen_center(center_choice, tensor[i:i+1], _target_tensor, distribution,
    #                              _shift[:, i:i+1])
    #         crop_start = _center - _sizes[:, i:i+1].view(1, -1, 1)/2
    #         # if config.DEBUG:
    #         #     print(f"center:{_center}, sizes:{_sizes} -> {_sizes[:, i:i+1].view(1, -1, 1)/2}")

    #         _data = crop(_data, crop_start.long(), _sizes[:, i].long())
    #         _data = resize(_data, size=_size.view(-1).long(), interpolation="linear")
    #         _datas.append(_data)

    #     return refold_tensors(_datas, types)

    # size of resulting image
    # if size is None: size = minside - minside*contract
    size = gen_size(tensor, size, contract=0.1)

    # shift, amount of max random shift from center for sd=1
    _shift = size.clone().detach().mul(shift).view(-1, 1)

    _tensor_list_idx = [i for i in range(len(types)) if types[i] == "tensor_list"]
    _target_tensor = None if not _tensor_list_idx else data[_tensor_list_idx[0]]
    center = gen_center(center_choice, tensor, _target_tensor, distribution, _shift)

    crop_start = (center - size.view(1, -1, 1)/2).long()
    size = size.long()

    data = crop(data, crop_start, size)

    #if need to scale up
    if tensor.shape[2:] < tuple(size.tolist()):
        data = resize(data, size)

    if config.VERBOSE:
        print(" :F.center_crop(): %s to %s"%(crop_start, size))

    return data

def two_crop(data, inplace=True):
    """transforms.TwoCrop()
    """
    if config.VERBOSE:
        print("TwoCrop():")

    data, types = unfold_tensors(data, msg="crop input:, ", inplace=inplace)
    tensor = data[0]

    assert tensor.ndimension() == 4, "only NCHW tensors supported"
    _n, _c, _h, _w = tensor.size()
    if _h == _w:
        return data

    size = gen_size(tensor, None).long()

    crop_start = torch.tensor([[0, 0], [_h-_w, 0] if _h > _w else [0, _w-_h]], device=tensor.device,
                              dtype=torch.int64, requires_grad=False).t().unsqueeze(0)
    return crop(data, crop_start, size)

def normalize(data, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        tensor  (Tensor or sequence where [0] is Tensor): Tensor image of size [N,C,H,W]
        mean    (sequence): Sequence of means for each channel
        std (sequence): Sequence of standard deviations for each channely
        inplace (bool[False])

    Returns:
        list [norm tensor, data[1], data[2]]
    """
    data, types = unfold_tensors(data, msg="normalize input:,", inplace=inplace)
    data[0] = _normalize(data[0], mean=mean, std=std, inplace=True)
    return data


def _normalize(tensor, mean, std, inplace=False):
    """normalize oprating on tensors"""
    if not inplace:
        tensor = tensor.clone().detach()

    # mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    # std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    # tensor.sub_(mean[:, None, None]).div_(std[:, None, None])

    dims = [1 for i in range(tensor.ndim - 2)]
    mean = _norm_mean(tensor, mean)
    std = _norm_std(tensor, std)
    try:
        tensor.sub_(mean.view(*mean.shape, *dims)).div_(std.view(*std.shape, *dims))
    except:
        print("\n normalize fails\n  ", tensor.shape, mean.shape, dims, std.shape)
        raise NotImplementedError
    return tensor

def _norm_mean(tensor, mean):
    """ if mean is a sequence, returns it as tensor
        if mean is float tensor.mean() - mean
    """
    if isinstance(mean, (int, float)):
        return tensor.view(*tensor.shape[:2], -1).mean(-1).sub(mean)
    else:
        return torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)

def _norm_std(tensor, std):
    """ if std is a sequence, returns it as tensor
        if std is float tensor.std()/std
    """
    if isinstance(std, (int, float)):
        assert std > 0, "std cannot be zero"
        std = tensor.view(*tensor.shape[:2], -1).std(-1).div(std)
        std[std < 0.001] = 0.001
        return std
    else:
        return torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)


def unnormalize(data, mean, std, inplace=False, clip=False):
    """UnNormalize a tensor image with mean and standard deviation.
    Args:
        tensor  (Tensor or sequence where [0] is Tensor): Tensor image of size [N,C,H,W]
        mean    (sequence): Sequence of means for each channel
        std     (sequence): Sequence of standard deviations for each channely
        inplace (bool[False])
    Returns:
        list 
    """
    data, types = unfold_tensors(data, msg="unnormalize input:,", inplace=inplace)
    data[0] = _unnormalize(data[0], mean, std, inplace=True, clip=clip)
    return data

def _unnormalize(tensor, mean, std, inplace=False, clip=False):
    """unnormalize oprating on tensors"""
    if not inplace:
        tensor = tensor.clone().detach()

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    if clip:
        tensor.clamp_(0.0, 1.0)
    return tensor

def norm_to_range(data, minimum, maximum, excess_only, independent, per_channel, inplace):
    """map tensor linearly to a range
    Args:
        data (Tensor): Tensor of size (C, H, W) to be normalized to range.
    Returns:
        Tensor: linearly normalized to range
    """
    data, types = unfold_tensors(data, msg="unnormalize input:,", inplace=inplace)
    tensor = data[0]

    while tensor.ndim < 4:
        tensor = tensor.view(1, *tensor.shape)
    if tensor.dtype not in (torch.float64, torch.float32, torch.float16):
        tensor = tensor.to(dtype=torch.float32)

    tensor = _norm_to_range(tensor, minimum, maximum, excess_only, independent, per_channel)
    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="_norm_to_range() output ")

    return data

def _norm_to_range(tensor, minimum, maximum, excess_only, independent, per_channel=True):
    """norm to range oprating on tensors"""

    if independent:
        for i, _ in enumerate(tensor):
            tensor[i:i+1] = _norm_to_range(tensor[i:i+1], minimum=minimum, maximum=maximum,
                                           excess_only=excess_only, independent=False,
                                           per_channel=per_channel)

    else:
        # could extend channel min broadcast to batch size
        if per_channel:
            _shape = [1 for i in range(tensor.dim())]
            _shape[1] = tensor.shape[1]
            _min = torch.tensor([tensor[:, i, ...].min() for i in range(_shape[1])],
                                dtype=tensor.dtype, device=tensor.device).view(_shape)
            _max = torch.tensor([tensor[:, i, ...].max() for i in range(_shape[1])],
                                dtype=tensor.dtype, device=tensor.device).view(_shape)
        else:
            _min = tensor.min()
            _max = tensor.max()

        if not excess_only or tensor.min().item() < minimum or tensor.max().item() > maximum:
            _denom = _max.sub(_min).add(minimum)
            _deneq0 = (_denom == 0).to(dtype=_denom.dtype)
            _denom.add_(_deneq0) # NaN prevent if denom == 0: denom = 1

            tensor.sub_(_min).mul_(maximum - minimum).div_(_denom)
    return tensor
##
#
#   Appearance Transforms
#
def rgb2lab(data, illuminant="D65", observer="2", p=1, independent=0, inplace=False):
    """ functional for transform.RGBToLab()
    """
    data, types = unfold_tensors(data, msg="rgb2lab()  input:,", inplace=inplace)
    tensor = data[0]

    if p < 1:
        _len = len(tensor) if independent else 1
        prob = bernoulli(p, _len, dtype=torch.uint8, device="cpu", grad=False)
    else:
        prob = [1]

    R = RGBTo(tensor, inplace=True)
    if tensor.is_cuda:
        tensor[:] = R.xyz2lab_cuda()
    else:
        tensor[:] = R.xyz2lab_cpu()
    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="rgb2lab() output:, ")

    return data

def lab2rgb(data, illuminant="D65", observer="2", p=1, independent=1, inplace=False):
    """ functional for transform.LabToRGB()
    """
    data, types = unfold_tensors(data, msg="lab2rgb()  input:,", inplace=inplace)
    tensor = data[0]

    tensor[:] = xyz2rgb(lab2xyz(tensor, illum=illuminant, observer=observer))
    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="lab2rgb() output:, ")
    return data

def saturate(data, sat_a, sat_b, p=1, distribution=None, independent=1, inplace=False):
    """ functional for transform.Saturate()
    # TODO this cannot work in place, fix
    """
    data, types = unfold_tensors(data, msg="saturate()  input:,", inplace=inplace)
    tensor = data[0]

    distribution = check_distribution(distribution)
    if sat_b is not None:
        _size = len(tensor)
        if not independent:
            _size = 1
        _sat = get_random(tensor, sat_a, sat_b, p, distribution, independent, grad=False)
    else:
        _sat = torch.tensor([sat_a for i in range(len(tensor))], dtype=tensor.dtype,
                            device=tensor.device, requires_grad=False)

    tensor = to_saturation(tensor, _sat, inplace=True)

    del _sat
    if config.FORCE_CLEANUP and tensor.is_cuda:
        torch.cuda.empty_cache()

    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="saturate() output:, ")
    return data

def color(data, p, target, distribution, independent, inplace, luma=False, low=0.2, high=0.95):
    """functional form transform.ColorShift()
    # TODO this cannot work in place, fix
    """
    data, types = unfold_tensors(data, msg="saturate()  input:,", inplace=inplace)
    tensor = data[0]

    if p == 0: # match single target
        tensor = _chroma_match(tensor, target, inplace=True, luma=luma, low=low, high=low)
    else:

        prob = bernoulli(p, len(tensor), dtype=tensor.dtype, device=tensor.device, grad=False)
        _lab = get_random_color(tensor, distribution, independent=independent, grad=False)
        target = lab2xyz(_lab)

        tensor = chroma_match(tensor, target, prob, inplace=True, luma=luma, low=low, high=low)
        del prob
        del target
        if config.FORCE_CLEANUP and tensor.is_cuda:
            torch.cuda.empty_cache()

    if config.DEBUG:
        _size = check_tensor(tensor)/2**20
        _msg = "shift_color: %dMB"%(_size)
        mem.report(memory=config.MEM, msg=_msg)

    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="color() output:, ")
    return data

def chroma_match(tensors, target, prob, inplace, luma=False, low=0.2, high=0.95):
    """ given a target list, a probability list match color of tensor batch,
    #TODO check inplace
    """
    if inplace:
        out = tensors
    else:
        out = tensors.clone().detach()
        if config.DEBUG:
            _msg = "clone"
            mem.report(memory=config.MEM, msg=_msg)

    if not luma:
        target /= target[:, 1:2]

    for i, tensor in enumerate(out):
        if prob[i]:
            gray = to_grayscale(out[i:i+1], inplace=False)
            xyz = rgb2xyz(out[i:i+1], inplace=True)
            mask = get_whites_mask(gray, low=low, high=high)
            chroma = get_whites_mean(xyz, mask)
            del mask
            if config.FORCE_CLEANUP and tensors.is_cuda:
                torch.cuda.empty_cache()

            if not luma:
                chroma /= chroma[1]

            #_luma = xyz.mean([0, 1, 2])[1]
            xyz = chroma_adapt(xyz, chroma, target[i])
            out[i:i+1] = xyz2rgb(xyz)

    return out

def _chroma_match(tensors, target, inplace, luma=False, low=0.2, high=0.95):
    """given single target XYZ color, match color of tensors
    TODO merge with function above
    TODO clean up after itself
    TODO inplace / offplace
    TODO simpler version, no white balance
    """
    if inplace:
        out = tensors
    else:
        out = tensors.clone().detach()
        if config.DEBUG:
            _msg = "clone"
            mem.report(memory=config.MEM, msg=_msg)

    if not luma:
        target /= target[1]

    for i, tensor in enumerate(tensors):
        gray = to_grayscale(out[i:i+1], inplace=False)
        xyz = rgb2xyz(out[i:i+1], inplace=True)
        mask = get_whites_mask(gray, low=low, high=high)
        chroma = get_whites_mean(xyz, mask)
        del mask
        if config.FORCE_CLEANUP and tensor.is_cuda:
            torch.cuda.empty_cache()

        if not luma:
            chroma /= chroma[1]
        xyz = chroma_adapt(xyz, chroma, target)
        out[i:i+1] = xyz2rgb(xyz)

    del gray
    del xyz
    del chroma
    if config.FORCE_CLEANUP and out.is_cuda:
        torch.cuda.empty_cache()

    _msg = "_chroma_match empty_cache"
    mem.report(memory=config.MEM, msg=_msg)

    return out

def get_chroma(target, low=0.2, high=0.95):
    """ Returns CIEXYZ tensor of shape [3]
    Args:
        target
            (None)          returns ImageNet mean
            (torch.Tensor)  requires [NCHW] tensor, with 3 channels
            (str)           Standard illuminant ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75',
                                'E', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                                'F11', 'F12']
            (np.ndarray, list, tuple, tensor)    requires array shape 3
    """
    if target is None:
        chroma = torch.from_numpy(Imagenet_XYZ.mean).to(dtype=torch.__dict__[config.DTYPE],
                                                        device=config.DEVICE)

    elif isinstance(target, (tuple, list, np.ndarray, torch.Tensor)) and len(target) == 3:
        assert len(target) == 3, "expected '(x,y,z)' format"
        chroma = torch.tensor(target).to(dtype=torch.__dict__[config.DTYPE], device=config.DEVICE)

    elif isinstance(target, torch.Tensor):
        _shape = target.shape
        assert len(_shape) == 4 and _shape[1] == 3, "Requires 3 color channel NCHW tensor"
        chroma = _chroma_from_tensor(target, low=low, high=high)

    elif isinstance(target, str):
        chroma = get_illuminant(illuminant=target, observer=2)
        chroma = torch.from_numpy(chroma).to(dtype=torch.__dict__[config.DTYPE],
                                             device=config.DEVICE)
        luma = torch.from_numpy(Imagenet_XYZ.mean).to(dtype=torch.__dict__[config.DTYPE],
                                                      device=config.DEVICE)[1].item()
        chroma /= luma

    else:
        print("chroma not recognized, implement in functional get_chroma()")
        raise NotImplementedError

    if config.DEBUG:
        _msg = "chroma match"
        mem.report(memory=config.MEM, msg=_msg)

    return chroma

def _chroma_from_tensor(tensor, low, high):
    """ returns unnormalized chroma, xyz
        TODO if tensor N > 1 fix and verify getting chroma array
        TODO get mean, then compute rgb2xyz
    """
    gray = to_grayscale(tensor, inplace=False)
    xyz = rgb2xyz(tensor)
    luma = xyz.mean([0, 1, 2])[1]
    mask = get_whites_mask(gray, low=low, high=high)
    chroma = get_whites_mean(xyz, mask)

    if config.DEBUG:
        print(f"_chroma_from_tensor() whites xyz {chroma}")
    chroma = luma*chroma/chroma[1]
    if config.DEBUG:
        print(f"_chroma_from_tensor() adjusted chroma xyz {chroma}")

    return chroma
###
#
# noise
#
def noise(data, blend, scale, p, mode, gray, independent, clamp, inplace):
    """ transforms.Noise()
    """
    data, types = unfold_tensors(data, msg="noise()  input:,", inplace=inplace)
    tensor = data[0]

    size = len(tensor) if independent else 1
    prob = bernoulli(p, size, dtype=torch.uint8, device="cpu", grad=False)
    if isinstance(scale, tuple):
        if len(scale) == 1:
            scale = scale(1, scale[0])
        scales = mutlinoulli(scale[0], scale[1]+1, len(prob), dtype=torch.int, device="cpu",
                             grad=False)
    else:
        scales = torch.ones(len(prob), dtype=torch.int, device="cpu", requires_grad=False) * scale

    #print(f"scales:{scales}, prob:{prob}, ")
    if not independent:
        if prob.item():
            tensor[:] = GaussNoise(blend=blend, mode=mode, scale=scales[0].item(), gray=gray,
                                   clamp=None, force=True)(tensor)
            if config.LOG_VALS:
                print("noise: blend %.2f, mode %d, scale %d, clamp %s"%(blend, mode, scale, clamp))
    else:
        for i, p in enumerate(prob):
            if p:
                tensor[i:i+1] = GaussNoise(blend=blend, mode=mode, scale=scales[i].item(),
                                           gray=gray, clamp=None, force=True)(tensor[i:i+1])

                if config.LOG_VALS:
                    print("noise: blend %.2f, mode %d, scale %d, clamp %s"%(blend, mode,
                                                                            scales[i].item(),
                                                                            clamp))

    if clamp and (tensor.min() < 0 or tensor.max() > 1):
        tensor = _clamp(tensor, soft=True, independent=True)

    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="noise() output:, ")
    return data

def high_response(data, p, independent, shuffle, inplace):
    """transforms.HighResponse()
    to_grayscale, invert, folded
    """
    data, types = unfold_tensors(data, msg="high_response()  input:,", inplace=inplace)
    tensor = data[0]
    n, c, h, w = tensor.shape
    assert c == 3, "implemented only for 3 channel images, got '%d'"%c


    size = n if independent else 1
    prob = bernoulli(p, size, dtype=torch.uint8, device="cpu", grad=False)
    if not independent:
        tensor[:] = _high_response(tensor, shuffle)
    else:
        for i, p in enumerate(prob):
            if p:
                tensor[i:i+1] = _high_response(tensor[i:i+1], shuffle)
    data[0] = tensor
    return data

def _high_response(tensor, shuffle=True):
    """ inplace 3 channel grayscale
        Args:
            shuffle ([True]) shuffle order of out channels
    """
    to_grayscale(tensor, inplace=True)
    _ch = np.arange(3)
    if shuffle:
        np.random.shuffle(_ch)

    tensor[:, _ch[1], :, :] = 1 - tensor[:, _ch[1], :, :]
    tensor[:, _ch[2], :, :] = 1 - torch.abs(1 - 2* tensor[:, _ch[2], :, :])
    tensor = check_contiguous(tensor, verbose=config.DEBUG, msg="_high_response() output:, ")
    return tensor

def gamma(data, g, a, b, p, distribution, independent, inplace, verbose=False):
    """transforms.Gamma()
    """
    data, types = unfold_tensors(data, msg="high_response()  input:,", inplace=inplace)
    tensor = data[0]
    n, c, h, w = tensor.shape

    if not p:
        _gamma(tensor, g_from=g, g_to=a, verbose=verbose)

    else:
        size = n if independent else 1

        prob = bernoulli(p, size, grad=False)

        _tg = [a, g]
        if b is not None:
            _tg.append(b)
        _tg = sorted(_tg)

        if distribution == "uniform":
            gammas = torch.clamp(uniform(_tg[0], _tg[-1], samples=size, grad=False,
                                         dtype=tensor.dtype, device=tensor.device),
                                 _tg[0], _tg[-1])
        elif distribution == "normal":
            gammas = torch.clamp(normal(mean=(_tg[0] + _tg[-1])/2, std=3, samples=size,
                                        dtype=tensor.dtype, device=tensor.device, grad=False),
                                 _tg[0], _tg[-1])
        else: # randint
            gammas = bernoulli(0.5, size, dtype=tensor.dtype,
                               device=tensor.device)*(_tg[-1] - _tg[0]) + _tg[0]

        if not independent and prob[0]:
            _gamma(tensor, g_from=g, g_to=gammas[0], verbose=verbose)
        else:
            _gamma(tensor, g_from=g, g_to=gammas.view(-1, 1, 1, 1), verbose=verbose)

    data[0] = tensor
    return data

def _gamma(tensor, g_from=2.2, g_to=1.0, verbose=False):
    if verbose:
        print("Gamma(%.2f -> %.2f)"%(g_from, g_to))
    tensor.pow_(g_from/g_to)

##
#
# fmix
#
def mix(data, p, inplace):
    """ TODO mix loss as well
    mixaugment paper
    """
    data, types = unfold_tensors(data, msg="mix()  input:,", inplace=inplace)
    tensor = data[0]
    n, c, h, w = tensor.shape

    prob = bernoulli(p, n, dtype=torch.uint8, device="cpu", grad=False)
    mix_with = np.roll(np.arange(n), 1)

    for i, _p in enumerate(prob):
        if _p:
            _i = mix_with[i]
            _other = tensor[_i:_i+1] if _i != i else _zoom(rotate(noise(tensor[_i:_i+1], 0.15, (1, 4),
                                                                    1, 0, True, 0, None, False),
                                                              0.75, 1, None, 0, 1, 1, "linear",
                                                              "pi"), scale=2)[0]
            # print(_other.dtype, _other.device)
            tensor[i:i+1].lerp_(_other, 0.5)

    data[0] = check_contiguous(tensor, verbose=config.DEBUG, msg="mix() output:, ")
    return data
###
#
#
#
def softclamp(data, soft=True, independent=True, inplace=True):
    """ clamp between 1 and 0
        Args:
            list   ((torch.Tensor), [], [])
            soft    ([True]) tanh or torch hard clamp
            independent: per elemenet of batch
    """
    data, types = unfold_tensors(data, msg="softclamp()  input:,", inplace=inplace)
    data[0] = _clamp(data[0], soft, independent)
    return data

def _clamp(tensor, soft=True, independent=True):
    """ clamp between 1 and 0
        Args:
            tensor  (torch.Tensor)
            soft    ([True]) tanh or torch hard clamp
            independent: per elemenet of batch
    """
    if not soft:
        return _hardclamp(tensor, independent)
    return _softclamp(tensor, independent)

def _hardclamp(tensor, independent=True):
    """ torch clamp between 1 and 0
        Args:
            tensor  (torch.Tensor)
            independent: per elemenet of batch
    """
    if independent:
        for i, o in enumerate(tensor):
            tensor[i:i+1] = torch.clamp(tensor[i:i+1], 0, 1)
    else:
        tensor[:] = torch.clamp(tensor[:], 0, 1)
    return tensor

def _softclamp(tensor, independent=True):
    """ piecewise tanh clamp between 1 and 0
        Args:
            tensor  (torch.Tensor)
            independent: per elemenet of batch
    """
    if independent:
        for i, o in enumerate(tensor):
            tensor[i:i+1] = tanh_clamp(tensor[i:i+1], inplace=True)
    else:
        tensor[:] = tanh_clamp(tensor[:], inplace=True)
    return tensor


def mask(data, low, high, inplace):
    """
    """
    data, types = unfold_tensors(data, msg="mask()  input:,", inplace=inplace)
    data[0][:] = get_whites_mask(data[0], low=low, high=high)
    return data

def glow(data, p, threshold, blend, scale, clamp, inplace):
    """
    """
    data, types = unfold_tensors(data, msg="glow()  input:,", inplace=inplace)

    if config.LOG_VALS:
        print("glow: scale %.3f threshold %.3f, blend %.2f"%(scale, threshold, blend))

    #_mask = _blur(get_whites_mask(out, low=0.0, high=threshold), p=1, x=0.02, y=0.02)
    _mask = _blur(get_whites_mask(data[0], low=0.0, high=threshold),
                  p=1, x=0.05, y=0.05, dx=0.02, dy=0.01)

    data[0] = data[0] * (1 - _mask) + _mask*torch.lerp(data[0], _blur(data[0]*scale, p=1, x=0.5, y=0.2), blend)

    if clamp:
        data[0] = _clamp(data[0])

    return data
##
#
# fourier agumentations
#
def ftclamp(data, p, both=True, inplace=True):
    """ clamp of lower range of both or imaginary part of fourier
        Args:
            data    (list(Tensor, tensorlist, labelist))
            p       (float 0-1)     bernoulli probability of clamp
            both    (bool[True])    False, clamp real, True, clamp both real and imaginary
    """
    data, types = unfold_tensors(data, msg="glow()  input:,", inplace=inplace)

    n, c, h, w = data[0].shape

    data[0] = check_contiguous(data[0], verbose=config.DEBUG, msg="ftclamp() input:, ")

    prob = bernoulli(p, n, dtype=torch.uint8, device="cpu", grad=False)
    for i, p in enumerate(prob):
        if p:
            onesided = True
            shape = data[0][i:i+1].shape
            fft = torch.rfft(out[i:i+1], 2, onesided=onesided)
            if both:
                fft = torch.clamp(fft, min=0.0)
            else:
                fft[..., 1] = torch.clamp(fft[..., 1], min=0.0)
            data[0][i:i+1] = torch.clamp(torch.irfft(fft, 2, onesided=onesided,
                                                 signal_sizes=shape[2:]), min=0.0, max=1.0)

    data[0] = check_contiguous(data[0], verbose=config.DEBUG, msg="ftclamp() output:, ")

    return data


def _blur(tensor, p, x, y, dx=0, dy=0, angle=0, da=0, independent=1, distribution="uniform",
          visualize=False):

    _kernel = None
    _fourier = None
    if visualize:
        _kernel = tensor.clone().detach()
        _fourier = tensor.clone().detach()

    n, c, h, w = tensor.shape

    alphas = [x, y, dx, dy]
    for a in alphas:
        assert 1.0 >= a >= 0, "x, y, rangex, rangey blur amounts must be between 0 and 1"

    # convert from image percentage
    dx = w * max(min(dx, 1.0), 0.0)
    dy = w * max(min(dy, 1.0), 0.0)
    sx = w * max(min(x, 1.0), 0.001)
    sy = h * max(min(y, 1.0), 0.001)

    size = n if independent else 1
    prob = bernoulli(p, size, dtype=torch.uint8, device="cpu", grad=False)

    params = [sx, sy, angle]
    ranges = [dx, dy, da]
    for i, param in enumerate(params):
        if not ranges[i]:
            params[i] = torch.tensor([param for _p in range(size)], dtype=torch.float, device="cpu",
                                     requires_grad=False)
        elif distribution == "uniform":
            params[i] = uniform(param - ranges[i], param + ranges[i], size,
                                dtype=torch.float, device="cpu", grad=False)
        elif distribution == "normal":
            params[i] = uniform(param, ranges[i], size,
                                dtype=torch.float, device="cpu", grad=False)

    params[0] = torch.clamp(params[0], min=0.1, max=w)
    params[1] = torch.clamp(params[1], min=0.1, max=h)
    params = torch.stack(params).t()

    if config.LOG_VALS:
        print("blur: x %.3f, y %.3f, dx %.3f, dy %.3f, a %.3f, %.3f da"%(x, y, dx, dy, angle, da))
        print("--")
        for p in params:
            print("x %.3f, y %.3f, angle %.2frad"%((p[0]), (p[1]), p[2]))

    if independent:
        for i, p in enumerate(prob):
            if p:
                _f = _fourier if _fourier is None else _fourier[i:i+1]
                _k = _kernel if _kernel is None else _kernel[i:i+1]
                _ftblur(tensor[i:i+1], params[i], c, h, w, _k, _f) # tensor[i:i+1] =
    elif prob:
        _ftblur(tensor, params[i], c, h, w, _kernel, _fourier) # tensor[:] =

    if visualize:
        _t = []
        if _kernel is not None:
            _t.append(_kernel)
        if _fourier is not None:
            _t.append(_fourier)
        if _t:
            _t.append(tensor)
            tensor = torch.cat(_t, dim=0)

    return tensor

def _ftblur(tensor, params, c, h, w, _kernel=None, _fourier=None):

    tensor = check_contiguous(tensor, verbose=config.DEBUG, msg="ftblur() input:, ")
    sx = params[0]
    sy = params[1]
    angle = params[2]

    onesided = False
    # shape = tensor[i:i+1].shape # if onesided ==True
    fft = ffshift(torch.rfft(tensor, 2, onesided=onesided), True)
    psf = gauss_grid(c, w, h, sx, sy, angle, is_complex=True).to(device=tensor.device)

    # if config.DEBUG:
    #     print("blur() psf minmax", psf.min(), psf.max())
    #     print("blur() fft minmax", fft.min(), fft.max())

    if _kernel is not None:# visualize kernel
        _kernel[:] = torch.unbind(psf, -1)[0].unsqueeze(0)

    fft.mul_(psf)

    # if config.DEBUG:
    #     print("blur() fft mul minmax", fft.min(), fft.max())

    if _fourier is not None: # visualize fourer
        _fourier[:] = torch.unbind(fft, -1)[0].unsqueeze(0).clone().detach()
        _fourier[:] = torch.log(torch.abs(_fourier)+ 1)
        _fourier[:] /= _fourier.max()

    fft = ifftshift(fft, True)
    tensor[:] = torch.irfft(fft, 2, onesided=onesided)#, signal_sizes=shape)

    tensor = check_contiguous(tensor, verbose=config.DEBUG, msg="ftblur() output:, ")

    del fft
    del psf
    if config.FORCE_CLEANUP and tensor.is_cuda:
        torch.cuda.empty_cache()
    #return tensor

def blur(data, p, x, y, dx, dy, angle, da, distribution, independent, clamp, inplace,
         visualize=False):
    """ blur image using a PSF on fourier space
    change, anisotropy and rotation
    Clean up useless vram occupation
    """
    data, types = unfold_tensors(data, msg="blur()  input:,", inplace=inplace)

    data[0] = _blur(data[0], p, x, y, dx, dy, angle, da, independent, distribution, visualize)
    if clamp:
        data[0] = _clamp(data[0])
    data[0] = check_contiguous(data[0], verbose=config.DEBUG, msg="blur() output:, ")
    return data

def ffshift(x, inplace=True, inv=0):
    n, c, h, w, j = x.shape
    shifts = ((h)//2, (w)//2) if inv else ((h+1)//2, (w+1)//2)
    if not inplace:
        return torch.roll(x, shifts=shifts, dims=(2, 3))
    x[:] = torch.roll(x, shifts=shifts, dims=(2, 3))
    x = check_contiguous(x, verbose=True, msg="ffshift():, ")
    return x

def ifftshift(x, inplace=True):
    return ffshift(x, inplace=inplace, inv=1)

def get_cov(ang, sx, sy):
    """returns covariance matrix for gaussian distribution / requires normalization"""
    a = math.cos(ang)**2/(2*sx**2) + math.sin(ang)**2/(2*sy**2)
    b = -1*math.sin(2*ang)/(4*sx**2) + math.sin(2*ang)/(4*sy**2)
    c = math.sin(ang)**2/(2*sx**2) + math.cos(ang)**2/(2*sy**2)
    return torch.tensor([[a, b], [b, c]], dtype=torch.float32, device="cpu", requires_grad=False)

def tnorm(grid, center, cov, dims=2):
    """ n dim norm
    """
    _x = grid - center
    _ex = torch.exp(-0.5 * torch.einsum('...i, ij, ...j->...', _x, torch.inverse(cov), _x))
    return _ex / (torch.sqrt((2*math.pi)**dims * torch.det(cov)))

def gauss_grid(c, h, w, sx, sy, ang, is_complex=False):
    """
    """
    grid = torch.stack(torch.meshgrid(torch.linspace(0., 1., w), torch.linspace(0., 1., h)), dim=-1)
    mean = torch.full((2,), 0.5, dtype=torch.float32, device="cpu", requires_grad=False)
    cov = get_cov(ang, sx, sy)
    psf = torch.stack([tnorm(grid, mean, cov, dims=2)for i in range(c)], dim=0).unsqueeze(0)
    psf.sub_(psf.min()).div_(psf.max())
    if not is_complex:
        return psf
    return torch.stack([psf for i in range(2)], dim=-1)


####
#
# halftones
#
#
def halftone(data, scale, p, shift, inplace):
    data, types = unfold_tensors(data, msg="halftone()  input:,", inplace=inplace)
    raise NotImplementedError
    return data
