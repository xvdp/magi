"""(c) xvdp

Agument Library with syntax derived to torchvision.transforms (TT)
requires: python >= 3.6, torch >= 1.1

Differences:
* operates on torch tensors in NCHW format
    - TODO WIP: only some functions have been adapted NCD and NCHWD
* targets hanled by tensor transforms as, pos-pos, pos-delta, center-delta-angle:
    - ypath, xpath, yxyx, xyxy, yxwh, xyhw, yxwha, xyhwa
    TODO Sparse IO tensor for targets
* all transforms can be randomized with bernoulli prob and normal and uniform dist over ranges
    - TODO WIP ensure coordination with multiprocess multigpu pytorch random seed handler
* transforms are typed with a class attribute
    __type__ = "IO"             # handle inputs and outputs
    __type__ = "Compose"        # concat transforms
    __type__ = "TensorCompose"  # increase batch size
    __type__ = "Affine"         # Change pixel location and/or image size
    __type__ = "Appearance"     # Change pixel value
* all on init args can be updated locally on call

Functions
  "IO"
    Open()      # opens image file leveraging fastest available interface/ NEW function
      TODO .h5py, .tiff, .psd,. hdr, .exr, .leveldb, .pydicom, .las, .ptx, .fbx
    ToNumpy()   # converts tensor to numpy
    ToTensor()  # identical behaviour to TT, added probability term
    Show()      # shows tensor and targets
    Save()      # save image/s
      TODO video

  "Compose"
    Compose()   # similar behaviour to TT
    Choose()    # Binary Choice of Transforms of Transform Compose

  "TensorCompose"
    Fork()     # returns a batch of tensors from one or a batch of tensors
    Merge()
    TwoCrop()  # splits rectangular image into 2 square images #NEW
      TODO NCrop()

  "Affine"
    Flip()      # flips image
    Rotate()    # interpolations, nearest, linear, cubic, lanczos
    Resize()    # tensor formats, NCL, NCWH, NCWHD - interpolations: nearest, linear, area
    Scale()     # tensor format NCL, NCWH, NCWHD - interpolations: nearest, linear, area
    ResizeCrop()    # similar to RandomResize Crop
    SqueezeCrop()   # crops and squeezes NCL, NCHW or NCHWD tensors to square
    CenterCrop()    # crops CL, NCHW or NCHWD tensors; size=None, crops to minimum dimension
    ScaleStack()    # scales image to closest
    LaplacePyramid()# Laplacian Pyramid,
    GaussUp()       # unpool for pyramid
    GaussDn()       # pool for pyramid
      TODO Affine()
      TODO Perspective()
      TODO Distort() / Undistort
    Zoom()      # Scale up, crop to same size

  "Appearance"
    Saturate()      # replaces Grayscale - can desaturate to gray, invert or increase saturation
    RGBToLab()      # RGB to Lab
    LabToRGB()      # RGB to Lab
    ColorShift()    # replaces ColorJitter - rotates color in Lab space over ImageNet's distribution
    Noise()         # adds gauss noise
    HighResponse()  # grayscale, inverse grayscale, midtones
    Mix()           # mixAugment Confounder
     TODO Other  Confounders # square, blurred, linear, from distribution
    Gamma()         #
    Blur()          # blur kernel width stdx, stdy, angle
    Mask()          #
    Glow()          # lightens image with glow around lighter aresa
    Clamp()         # tanh softclamp or standard torch clamp
    FTClamp()       # clamps lower values of FFT

    Normalize()     # identical behaviour to TTs
    MeanCenter()    # alias to Normalize
    UnNormalize()   # invert of transform.Normalize()
    MeanUnCenter()  # alias to UnNormalize
    NormToRange()   # handles tensor of formats, NCL, NCWH, NCWHD

     TODO Halftone()

TODO General:
# "ToPILImage", "Pad", "Lambda", "RandomApply", "RandomOrder", "RandomResizedCrop",
# "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation", "AffineTransformation", 
# "RandomAffine"
> MEMORY STATE
> NORMALIZATION STATE
> GRADIENT MANAGEMENT
> LABELS: MINIMAL + DB
    uuid - for weight sharing and identity?
    mask and cull labels

> REGISTER OPERATORS & FILE HANGLERS
. register new augment operators.
. register data type: 1D, 2D, 3D, Mixed
. register new file openers dic: {ext:opener}
> UNIT TESTS
"""
import collections
import warnings
import logging
import numbers
import numpy as np
import torch
from . import functional as F
from .func_util import update_kwargs, check_size
from .func_random import bernoulli
from .tensor_util import unfold_data, inspect_data_sample, unfold_tensors, refold_tensors
from .. import config
from ..util.mem import CudaMem as cudamem

# pylint: disable=no-member
iterable = collections.abc.Iterable
__all__ = ["Config",
           "Compose", "Fork", "Merge", "Choose",
           "Open", "ToTensor", "ToNumpy", "Show", "Save",
           "Resize", "Scale", "Zoom", "Rotate", "Flip",
           "GaussDown", "GaussUp", "LaplacePyramid", "ScaleStack",
           "SqueezeCrop", "CenterCrop", "TwoCrop", "ResizeCrop",
           "Saturate", "RGBToLab", "LabToRGB", "ColorShift", "Noise", "HighResponse", "Gamma",
           "Clamp", "Mix", "FTClamp", "Blur", "Glow", "Mask",
           "Normalize", "UnNormalize", "MeanCenter", "MeanUnCenter", "NormToRange"]

def _make_repr(cls, exclude_keys=None):
    """ general make __repr__
    Args
        cls             (self)
        exclude_keys    (list, tuple [None])
    """
    rep = cls.__class__.__name__+"("
    for i, (key, value) in enumerate(cls.__dict__.items()):
        if exclude_keys is not None and key in exclude_keys:
            continue
        value = value if not isinstance(value, str)  else f"'{value}'"
        sep = "" if not i else ", "
        rep += f"{sep}{key}={value}"
    return rep + ")"

def Config(device=None, dtype=None, boxmode=None, rndmode=None, grad=False, datapaths=None, debug=None, verbose=None):
    """Sets global parameters
        Takes defaults from ..config.py
        device  (str) in "(cuda", "cuda:<index>", "cpu"); default, cuda if available
            cuda is much faster for many operations ops can quickly fill up VRAM
        dtype   (str) in ("half", "float", "double", "float16", "float32", "float64");
            float32 is most common desirable for balance of speed and capability
            float16 useful for TPUs, torch 1.0 does not handle float16 in cpu well
            float64 useful for higher precision computation, slower in FP64 crippled cards
        boxmode (str) in ("xywh", "yxhw", "xywha", "yxhwa", "xyxy", "yxyx", "ypath", "xpath")
            x...,       many datasets as well as matplotlib.Rectangle take xywh order
            y...,       images are height dominant, torch [NCHW], opencv, matplotlib [HWC]
            xywh,yxhw   top left corner, size
                shape [N,2,2]
            xyxy,yxyx   top left corner, bottom right corner
                shape [N,2,2]
            xpath,ypath path for bbox, segmentations can be represented as paths
                shape [N,pathsize,2]
            xywha,yxwha center, halfsize, angle
                shape: [N,5]

            TODO fix, xxyy, yyxx, failures detectd
            xxyy, yyxx  alternate path representation
                shape [N,2,pathSize]

        rndmode (str) in ("normal","uniform") determines kind of random distribution in transforms
    """
    __type__ = "Config"
    config.set_device(device)
    config.set_dtype(dtype)
    config.set_boxmode(boxmode)
    config.set_rndmode(rndmode)
    config.set_grad(grad)
    config.set_verbose(verbose)

    if debug is not None:
        config.DEBUG = debug
    if config.DEBUG:
        if config.MEM is None:
            config.MEM = cudamem(config.DEVICE)

    print("device: %s, dtype: %s, boxmode: %s, rndmode: %s, verbose: %s"%(config.DEVICE,
                                                                          config.DTYPE,
                                                                          config.BOXMODE,
                                                                          config.RNDMODE,
                                                                          config.VERBOSE))
##
#
# Compose Transforms
#
class Compose(object):
    """Composes several transforms together, arguments are evaluated left to right.

    similar to torchvision.transforms.Compose, added probability that one transform will be skipped

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
        p   (float 0-1 [1.0]) probability that any of the transforms will be performed

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    __type__ = "Compose"
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, data):
        probs = bernoulli(self.p, len(self.transforms), dtype="uint8", device="cpu", grad=False)
        for i, _t in enumerate(self.transforms):
            if probs[i]:
                try:
                    data = _t(data)

                except Exception as _e:
                    logging.exception(" >> Compose() Exception on apply transform")
                    return None
        return data

    def __repr__(self):
        return _make_repr(self)

class Choose(object):
    """ Binary choice of transforms of or transform Composes
        Only affine and appearance transforms are valid
        Args
            a   (list of transforms)
            b   (list of transfomrs)
            p   (float [0.5])   bernoulli probability
            independent (bool [True])   batch dependence/indemendence

        Example
        >>> Cc = at.CenterCrop(center_choice=2, size=(224,224), shift=0.5, distribution="uniform",
                    zoom=1, zdistribution="normal", zskew=0.9)
        >>> Rs = at.Resize(size=(224,224))
        >>> Rt = at.Rotate(angle=0.1, p=0.7)
        >>> Fl = at.Flip(0.5,0.05, independent=1)
        >>> Ch = at.Choose([Rt, Rs],[Rt, Fl, Cc], p=0.9)
    """
    __type__ = "Compose"
    def __init__(self, a, b, p=0.5, independent=True, inplace=True):
        self._validate_transform(a)
        self._validate_transform(b)
        self.a = a
        self.b = b
        self.p = p
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, _ = update_kwargs(self, **kwargs)
        # TODO move to functional
        # TODO generalize sampling
        # sample -> list of items -> convert to batch
        # item   -> list of tensor, target, label -> convert to batch
        # tensor -> convert to batch

        # batch -> tensors, targettensors, list of labels)
        # TODO change targettensorlist to sparse tensors

        independent = args['independent']
        inplace = args['inplace']
        nesting = inspect_data_sample(data)

        if nesting < 2:
            independent = False
        if nesting == 0:
            data, types = unfold_tensors(data, inplace=inplace)
            inplace = True

        if config.DEBUG:
            print("Choose()", ["tensor", "data", "sample"][nesting])

        _size = len(data) if args['independent'] else 1
        probs = bernoulli(args['p'], _size, dtype="uint8", device="cpu", grad=False)

        if not independent:
            _transforms = [args['a'], args['b']][probs[0]]
            for _t in _transforms:
                data = _t(data)
        else:
            _data = []
            for i, p in enumerate(probs):
                _transforms = [args['a'], args['b']][p]
                for _t in _transforms:
                    data[i] = _t(data[i])
                _data.append(data[i])
            # TODO how to handle has targets and has labels
            data = F.merge(_data)
        return data

    def _validate_transform(self, transforms):
        _valid = ("Compose", "Affine", "Appearance")
        for transform in transforms:
            assert transform.__type__ in _valid, "invalid tranform %s, only %s allowed"%(transform, str(_valid))

    def __repr__(self):
        return _make_repr(self)


class Transform(object):
    def _make_repr(self, exclude_keys=None):
        rep = self.__class__.__name__+"("
        for i, (key, value) in enumerate(self.__dict__.items()):
            if exclude_keys is not None and key in exclude_keys:
                continue
            value = value if not isinstance(value, str)  else f"'{value}'"
            sep = "" if not i else ", "
            rep += f"{sep}{key}={value}"
        return rep + ")"

class IOTransform(Transform):
    __type__ = "IO"

class AppearanceTransform(Transform):
    def __init__(self, inplace=True):
        self.inplace = inplace
    __type__ = "IO"


#
# IO Transforms
#
class Open(object):
    """Open a filename as torch tensor

    Args:
        file_name   (string or list of strings): valid existing filename or filename list
            TODO extend and validate format support
        dtype       (string): torch data format (Default "float32")
        device      (string): "cuda" or "cpu"; opens casts tensor on creation to device
        grad
        out_type:   (string): Default:'torch' or 'numpy']
        transforms: (torchvision transforms in crop, resize, transpose) [None]

    Returns:
        Tensor, default float32, in range 0,1, default NCL, NCHW, or NCHWD depending on data type
    """
    __type__ = "IO"
    def __init__(self, dtype=None, device=None, grad=None, inplace=None, out_type="torch",
                 channels=None, transforms=None, verbose=None):
        self.device = config.set_device(device)
        self.dtype = config.set_dtype(dtype)
        self.grad = config.set_grad(grad)
        self.inplace = config.set_inplace(inplace)
        self.out_type = out_type if out_type in ("torch", "numpy") else "torch"
        self.channels = channels
        self.transforms = transforms
        self.verbose = config.set_verbose(verbose)
        config.INIT = True

    def __call__(self, file_name, **kwargs):
        """
        Args:
            file_name   (str, list),  valid file name(s)
            **kwargs    any argument from __init__, locally
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.open_file(file_name, dtype=args["dtype"], device=args["device"], grad=args["grad"],
                           out_type=args["out_type"], channels=args["channels"],
                           transforms=args["transforms"], verbose=args["verbose"])

    def __repr__(self):
        return _make_repr(self)

class Save(object):
    __type__ = "IO"
    def __init__(self, folder=None, name="00000000", ext=".jpg", bpp=8, conflict=1, backend=None):
        """
        # TODO .npy, .hp5, .exr ...
        Args
            folder      (str None)  if None, save to current folder
            name        (str "00000000")  if None save number sequence
            ext         (str ['.jpg']) possible: (.jpg, .png, .tif, .webp)
            bpp         (int, [8])  16bpp (.png, .tif)
            conflict    (int [0])   -1 overwrite, 0, do nothing, 1, increment
            backend     (str [None]) None: first of found (accimage PIL, cv2)
        """
        self.folder = folder
        self.name = name
        self.ext = ext
        self.bpp = bpp
        self.conflict = conflict
        self.backend = backend

    def __call__(self, data, **kwargs):
        """
        Args:
            file_name   (str, list),  valid file name(s)
            **kwargs    any argument from __init__, locally
        """
        args, kw = update_kwargs(self, **kwargs)

        return F.save(data, folder=args["folder"], name=args["name"], ext=args["ext"],
                      bpp=args["bpp"], conflict=args["conflict"], backend=args["backend"])

    def __repr__(self):
        return _make_repr(self)

class BBox(object):
    """Add Bounding Box Metadata
    bounding boxes need to be [[n],[y,x],[y,x]]
    # TODO ADD ANNOTATION
    # TODO replace all with config.BOXMODE  .. ?
    TODO arg: input boxmode, arg draw boxmode optional.
    """
    __type__ = "IO"
    def __init__(self, boxmode=None):
        self.mode = config.set_boxmode(boxmode)

    def __call__(self, data, boxdata):
        return F.addbbox(data, boxdata)

    def __repr__(self):
        return _make_repr(self)

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Differences with torchvision:
    * Returns a tensor NCHW format
    * if device and dtype are passed, tensor is typed and as dvice

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (N x C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """
    __type__ = "IO"
    def __init__(self, dtype=None, device=None):
        self.dtype = dtype
        self.device = device

    def __call__(self, pic, **kwargs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor
            kwargs  can change init args on call
        Returns:
            Tensor: Converted image
        """
        args, kw = update_kwargs(self, **kwargs)
        return F.to_tensor(pic=pic, dtype=args["dtype"], device=args["device"])

    def __repr__(self):
        return _make_repr(self)

class ToNumpy(object):
    """Convert a ``torch.Tensor`` to ``numpy.ndarray``
        if dims != HW returns tensor as numpy tensor order
        if maxcols is not None returns grid of HWC
        if max cols is None returns ndarray NHWC format
        ncols and pad can be entered per trasformation or in class init
        Args:
            ncols   (int None)      if None, returns (n,h,w,c) otherwise (h*ncols, w*size//ncols, c)
            pad     (int [0])       pad between columns and rows
            bpp     (int [0])       0 leaves as float, 8 and 16 converts to np.uint8 and np.uint16
            inplace (bool [False])  if True and tensor is cpu, shares memory
            clean   (bool [False])  if True deletes torch tensor
    """
    __type__ = "IO"
    def __init__(self, ncols=None, bpp=0, clean=False, pad=0, inplace=True):
        self.ncols = ncols
        self.pad = pad
        self.bpp = bpp
        self.clean = clean
        self.pad = pad
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """Returns tensor NCHW as numpy array
        Args:
            data    (torch tensor or tuple of tensors)
            **kwargs    any argument from __init__, locally
        """
        args, kw = update_kwargs(self, **kwargs)

        return F.to_numpy(data, ncols=args["ncols"], pad=args["pad"], clean=args["clean"],
                          bpp=args["bpp"], inplace=args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class Show(object):
    """
    Converts to numpy ndarray and displays with matplotlib
    If spatial information exists displays it.
    negative data and data > 1 gets normalized to 0,1
    Args:
        ncols       (int [None]),   number of columns, if None: ncols = min(6, number of images)
        pad         (int),    nb of pixels between image,
                        applied only if more than one element in batch
        show_target (int/bool) if true show target boxes
        annot       (list), show target attributes, possible values
                    ["name", "uid", "size", "context"]
        width       (int,  [20]), matplotlib width
        height      (int, [None]) if None matches height to image
        path        (str, [None]) if not None saves image to path
        as_box      (bool, [False]) display paths as bounding boxes
        max_imgs    (int, [0]) if > 0 shows only requested number of images
        unfold_channels (bool, [False]) if True, show grayscale

    kwargs:
        mode        (str "xyhw") boxmode for showing target boxes
    """
    __type__ = "IO"
    def __init__(self, ncols=None, pad=0, show_targets=1, annot=None,
                 width=20, height=None, as_box=0, path=None, max_imgs=0, unfold_channels=False):
        self.ncols = ncols
        self.pad = pad
        self.show_targets = show_targets
        self.annot = annot
        self.width = width
        self.height = height
        self.path = path
        self.as_box = as_box
        self.max_imgs = max_imgs
        self.unfold_channels = unfold_channels

    def __call__(self, data, **kwargs):
        """
        Args:
            data        (torch tensor or tuple of tensors) NCHW
            **kwargs    any argument from __init__, locally
                        w shorthand for width

            # Too many kwargs
            extra kwargs:
                alpha       if alpha exists as 4th channel, mask alpha, 1: gray, 2: red
                bbox        draw a bounding box around all subboxes
                as_box      (bool/int),  if true, disregard rotation angle and draw rectangle
                hist        (bool/int) shows
                crop        show only a crop of image, format y0,y1,x0,x1
                title
                color       color of bounding boxes
                lwidth      linewidth of bounding boxes
                mode        box mode [config.BOXMODE]
        """
        args, kw = update_kwargs(self, **kwargs)

        width = kw['w'] if 'w' in kw else args["width"]


        if isinstance(data, np.ndarray):
            _div = 1.0 if data.dtype != np.uint8 else 255.
            data = torch.from_numpy(data).to(dtype=torch.float32).div_(_div)
        elif isinstance(data[0], np.ndarray):
            _div = 1.0 if data[0].dtype != np.uint8 else 255.
            data[0] = torch.from_numpy(data[0]).to(dtype=torch.float32).div_(_div)

        return F.show(data, ncols=args["ncols"], pad=args["pad"], show_targets=args["show_targets"],
                      annot=args["annot"], width=width, height=args["height"],
                      path=args["path"], as_box=args["as_box"], max_imgs=args["max_imgs"],
                      unfold_channels=args["unfold_channels"], **kw)
    def __repr__(self):
        return _make_repr(self)
##
#
#   TensorCompose Transforms
#
class Fork(object):
    """
        clones tensor into batch of tensors
        Args:
            data, tensor, target_tensor, labels
            number, (int, default 2) number of outputs
    """
    __type__ = "TensorCompose"
    def __init__(self, number=2):
        self.number = number

    def __call__(self, data, **kwargs):
        """
        Args:
            data
            **kwargs    any argument from __init__, locally
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.fork(data, args["number"])
    def __repr__(self):
        return _make_repr(self)


class Merge(object):
    """
    Merges [tensor, ..., ] or [(tensor, target, label), ..., ]
    Returns [batchtensor, [target, ..., ], [label, ..., ]]
    """
    __type__ = "TensorCompose"
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        """
        Args:
            data: [tensor, ..., ] or [(tensor, target, label), ..., ]
        """
        return F.merge(data)

    def __repr__(self):
        return _make_repr(self)

class Resize(object):
    """Resize the input torch Tensor Image to the given size.

    Args:
        size (sequence or positive int): Desired output size. If size is sequence,
            output size will be matched to this.
            Sequence length must equal Image Tensor shape length - 2.
            If size is an int, smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str|int, optional): Desired interpolation.
            Default is ``linear`| ``1``, options ``nearest, linear, area`` or ``[0,1,2]``
    """
    __type__ = "Affine"
    def __init__(self, size=None, interpolation="linear", square=False, align_corners=False):
        self.size = size
        self.interpolation = interpolation
        self.square = square
        self.align_corners = align_corners

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
            **kwargs    any argument from __init__, locally
        Returns:
           tensor, target_tensor.
        """
        args, _ = update_kwargs(self, **kwargs)
        return F.resize(data, args["size"], args["interpolation"], args["square"],
                        args["align_corners"])

    def __repr__(self):
        return _make_repr(self)

class Scale(object):
    """Scale input torch Tensor Image by a given scalar.
    Function implemented to handle random scaling before final size, or crop
    Args:
        scale (sequence or positive real number): Desired scale factor.
            If size is sequence, output size will be scaled by each entry.
            Sequence length must equal Image Tensor shape length - 2.
            If size is an int or float, all dimensions will be scaled in unison.
            * if p == 0: scale is the scalar by which the image size will be mul
            * if p > 0 and distribution is uniform, values will be picked between 1 and scale
            * if p > 0 and distribution is normal, values will be picked with a 3 std
            normal distribution bound by 1 and scale
                * if mean is not None the distribution will have skewness
        interpolation (str|int, optional): Desired interpolation.
            Default is ``linear`| ``1``, options ``nearest, linear, area`` or ``[0,1,2]``
        p   (float 0 - 1, default:0) bernoulli probability that scale will be affected
            by distribution, 0 means scale will be to target
        distribution (str, default "normal"), "normal" | "uniform" distribution
            centered on scale value with 3rd SD at 16 pixels, clipped at 16 pixels and current size
        mean (float, default 1/2 scale), only valid for normal distribution,
            values between scale and 1.0 if mean is set and not equal to 1/2 scale,
            distribution will have skewness
    """
    __type__ = "Affine"
    def __init__(self, p=1, scale=1.0, distribution="normal", interpolation="linear", mean=None):
        self.scale = scale
        self.interpolation = interpolation
        self.p = p
        self.distribution = distribution
        self.mean = mean

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs    any argument from __init__, locally
        Returns:
           tensor, target_tensor, labels
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.rescale(data, args["scale"], args["p"], args["distribution"],
                         args["interpolation"], args["mean"])

    def __repr__(self):
        return _make_repr(self)

class Zoom(object):
    """ Zoom image
    """
    __type__ = "Affine"
    def __init__(self, p=1, scale=None, distribution=None, independent=1,
                 center_choice=1, center_sharpness=1, interpolation="linear", pad_color=0.5,
                 inplace=True):

        self.p = p
        self.scale = scale
        self.distribution = distribution
        self.independent = independent
        self.center_choice = center_choice
        self.center_sharpness = center_sharpness
        self.interpolation = interpolation
        self.inplace = inplace
        self.pad_color = pad_color

    def __call__(self, data, **kwargs):
        args, _ = update_kwargs(self, **kwargs)

        if args["scale"] == 0:
            return data

        return F._zoom(data, scale=args["scale"], interpolation=args["interpolation"],
                       inplace=args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class Rotate(object):
    """
    Args:
        angle           (float, tuple of floats), rotation angle, or angle parameters in units
                    if p > 0
                         if angle is tuple, range between angle[0], angle[1]
                         if angle is float range between -angle, angle
                         if distribution is normal, angle = 3rd SD

        p:              (float) 0-1 beroulli probability that randomness will occur
        distribution:   (enum config.RnDMode) default None-> config.RNDMODE,  normal / uniform
        independent:    (int, default:1): if 0 all members of a batch receive the same rotation
        center_choice   (int, default 2)
                            1: image center
                            2: target items boudning box center
                                3: mean (1,2) ...

        center_sharpness (int: default 1) # unused
                            0: exact
                            1: normal dist, 5  sigma
                            2: normal dist, 3, stds
                            3: normal dist, 1, std


        interpolation   (str ['linear'])|'nearest'|'cubic'|'lanczos'
        units           (str ['pi'])|'radians'|'degree'
        scalefit        (float [1.0]) 0-1 zoom image to avoid padding
        inplace         (bool [True])
        debug           (bool [False]) # force ops on torch


    # TODO
    accelerate rotation with
        pytorch/aten/src/ATen/native/cudnn/AffineGridGenerator.cpp
        pytorch/aten/src/ATen/native/cpu/GridSamplerKernel.cpp
    Fourier Rotation
    """
    __type__ = "Affine"
    def __init__(self, angle=0, p=1, distribution=None, independent=1, center_choice=1,
                 center_sharpness=1, interpolation="linear", units="pi", scalefit=1.0,
                 inplace=True, debug=0):

        self.angle = angle
        self.p = p
        self.distribution = distribution
        self.independent = independent
        self.center_choice = center_choice
        self.center_sharpness = center_sharpness
        self.interpolation = interpolation
        self.units = units
        self.scalefit = scalefit
        self.inplace = inplace
        self.debug = debug

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (list of tensors): annotations
                labels:

            **kwargs    any argument from __init__, locally
                angle, p, distribution, interpolation, units

        Returns:
            tensor, target_tensor, labels
        """
        args, _ = update_kwargs(self, **kwargs)

        if args["angle"] == 0:
            return data

        return F.rotate(data, args["angle"], args["p"], args["distribution"], args["independent"],
                        args["center_choice"], args["center_sharpness"], args["interpolation"],
                        args["units"], args["scalefit"], args["inplace"], args["debug"])
    def __repr__(self):
        return _make_repr(self)


class Flip(object):
    """ flips image x or y with probability
    Args
        p_x         (float default 1)     bernoulli prob of horizontal flip
        p_y         (float default 0.0)     bernoulli prob of vertical flip
        independent (int default 1)         0, batch behaves together, 1, one toss per element
        inplace     (bool [True])
    """
    __type__ = "Appearance"
    def __init__(self, px=1, py=0, independent=1, inplace=True):
        self.px = px
        self.py = py
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args
            data    tensor or tensor tuple
            **kwargs    any argument from __init__, locally
                px, py, independent
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.flip(data, args["px"], args["py"], args["independent"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

# class Affine(object):
#     """Random affine transformation of the image keeping center invariant
#     Args:
#         angle (sequence or float or int): Range of degrees to select from.
#             If degrees is a number instead of sequence like (min, max), the range of degrees
#             will be (-angle, +angle). Set to 0 to deactivate rotations.

#         translate (tuple, optional): tuple of maximum absolute fraction for horizontal
#             and vertical translations. For example translate=(a, b), then horizontal shift
#             is randomly sampled in the range -img_width * a < dx < img_width * a and vertical
#             shift is randomly sampled in the range -img_height * b < dy < img_height * b.
#             Will not translate by default.
#         scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
#             randomly sampled from the range a <= scale <= b. Will keep original scale by default.
#         shear (sequence or float or int, optional): Range of degrees to select from.
#             If degrees is a number instead of sequence like (min, max), the range of degrees
#             will be (-degrees, +degrees). Will not apply shear by default
#         resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
#             An optional resampling filter. See `filters`_ for more information.
#             If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
#     """
#     def __init__(self, angle, translate=None, scale=None, shear=None, resample=False,
#                  fillcolor=0):
#         if isinstance(angle, numbers.Number):
#             if angle < 0:
#                 raise ValueError("If angle is a single number, it must be positive.")
#             self.angle = (-angle, angle)
#         else:
#             assert isinstance(angle, (tuple, list)) and len(angle) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.angle = angle

#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate

#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale

#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
#                     "shear should be a list or tuple and it must be of length 2."
#                 self.shear = shear
#         else:
#             self.shear = shear

#         self.resample = resample
#         self.fillcolor = fillcolor

#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation

#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(random.uniform(-max_dx, max_dx)),
#                             np.round(random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)

#         if scale_ranges is not None:
#             scale = random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0

#         if shears is not None:
#             shear = random.uniform(shears[0], shears[1])
#         else:
#             shear = 0.0

#         return angle, translations, scale, shear

#     def __call__(self, img):
#         """
#             img (PIL Image): Image to be transformed.

#         Returns:
#             PIL Image: Affine transformed image.
#         """
#         ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
#         return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)


#####
#
# laplacian pyramids
#
class GaussDown(object):
    """Downsamples with Gaussian kernel and stride
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown
        Applies a Gauss Kernel with stride (2), normalizes to maintain image statististics
    Args:
        stride (int): number of rows to skip, Default (2) reduces image 1/2,

    """
    __type__ = "Affine"

    def __call__(self, data):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations
        Returns:
            tensor, target_tensor
        """
        return F.gauss_down(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class GaussUp(object):
    """Upsamples and blurs
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown
        Applies a Gauss Kernel with stride (2), normalizes to maintain image statististics
    Args:
        stride (int): number of rows to skip, Default (2) reduces image to 50%,
    """
    __type__ = "Affine"

    def __call__(self, data):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
        Returns:
            tensor, target_tensor
        """
        return F.gauss_up(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LaplacePyramid(object):
    """Creates Laplacian Pyramid, based on
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown
        Applies a Gauss Kernel with stride 2, normalizes to maintain image statististics
    Args:
        steps       (int): number of steps
                           0: pyramid until reaching the min_side
        pad         (int): 0, (1:default) if True and image size is odd, pads image
        min_side    (int): if any side falls below min side pyramid stops
                        last useful size of the pyramid is next level up
        augments    (list of augment.transforms)
        pshuffle    (scalar [0]) shuffles list of augments
    """
    __type__ = "Affine"
    def __init__(self, steps=0, pad_mode=0, min_side=4, augments=None, pshuffle=0):
        self.steps = steps
        self.pad_mode = pad_mode
        self.min_side = max(4, min_side)
        self.augments = augments
        self.pshuffle = pshuffle

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): format NCL, NCHW, NCHWD
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor, labels
        """
        args, kw = update_kwargs(self, **kwargs)
        with torch.no_grad():
            return F.laplace_pyramid(data, args["steps"], args["pad_mode"], args["min_side"],
                                     args["augments"], args['pshuffle'], **kw)

    def __repr__(self):
        return _make_repr(self)

class ScaleStack(object):
    """Stacks downscaled tensor and crops of upscales.

    Args:
        target_size     (int [512]) size of longest edge
        interpolation   (str ['cubic']) in nearest, linear, area, cubic
        num_scales      (int [4])  num images: range(num_scales+1)**2, largest img: num_scales**2
        verbose         (bool [False])
    """
    __type__ = "Affine"
    __inputs__ = ("CHW", "NCHW")
    __outputs__ = ("NCHW",)
    def __init__(self, target_size=512, interpolation="cubic", num_scales=4, verbose=False):
        self.target_size = target_size
        self.interpolation = interpolation
        self.num_scales = num_scales
        self.verbose = verbose

    def __call__(self, data, **kwargs):
        args, _ = update_kwargs(self, **kwargs)
        with torch.no_grad():
            return F.scale_stack(data, target_size=args["target_size"],
                                 interpolation=args["interpolation"],
                                 num_scales=args["num_scales"],
                                 verbose=args["verbose"])

    def __repr__(self):
        return _make_repr(self)

#####
#
# crops
#
class SqueezeCrop(object):
    """Crops the given Torch Image and Size

    Args:
        size (int): Desired output size of the square crop.
        ratio: (float) [0.5]) squeeze to crop ratio
            if ratio == 0: only squeezes
            if ratio == 1: only crops
    """
    __type__ = "Affine"
    def __init__(self, size=None, ratio=0.5, interpolation="linear"):
        if size is not None:
            assert (isinstance(size, int) and size > 0) or isinstance(size, iterable)
        assert (isinstance(ratio, numbers.Number) and 1 >= ratio >= 0) or isinstance(size, iterable)

        self.size = size
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor, labels
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.squeeze_crop(data, args["size"], args["ratio"], args["interpolation"])

    def __repr__(self):
        return _make_repr(self)

class ResizeCrop(object):
    """
    similar to CenterCrop
    similar to torchvision.transforms.RandomResizeCrop
    Given torch image to validate training. its kind of a silly crop
        size            (tuple, int) expected output size of each edge
        scale           (tuple) range of size of the origin size cropped
        ratio           (tuple) range of aspect ratio of the origin aspect ratio cropped
        interpolation   (str ["linear"]) in ['linear' 'cubic']
        p               (int [1])   0: no randomness

    """
    __type__ = "Affine"
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation="linear",
                 p=1):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor, labels
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.resize_crop(data, args["size"], args["scale"], args["ratio"],
                             args["interpolation"], args["p"])

    def __repr__(self):
        return _make_repr(self)

class CenterCrop(object):
    """Crops the given Torch Image at the center.
        size optional, if no size, crop to minimum edge
        acts on tensors

        random shift with choice of alpha(0-1) to edge and distribution
        random zoom with choice of alpha(0-1) to target size, if size set
    Args:
        size (sequence or int, [None]): Desired output size of the crop.
            If size is an int, a square crop (size, size) is made.
            If size is None, square crop of length of min dimension is made.

        center_choice   (int, [2])
                            1: image center
                            2: if targets exist: target items boudning box center
                                3: mean (1,2) ...
        # shift then crop
        shift           (float, [0.1]), maximum allowed center shift in random center crops
                            recommended 0 < shift < 0.5
        distribution    (enum config.RnDMode ["normal"]) "normal" / "uniform"

        # zoom then crop
        zoom            (float, [0]) 0-1; zoom, requires size to be set to work
                            zoom=0 crops exact pixels, zoom=1, crops min dim and resizes
        zdistribution   (enum config.RnDMode ["normal"]) "normal" / "uniform"
                            normal is bounded normal between zoom 0, 1
        zskew           (float, [0.5]), 0,1;  0.5 means normal
                            1 skewed towards full zoom, 0 skewed towards no zoom
    """
    __type__ = "Affine"
    def __init__(self, size=None, center_choice=2, shift=0.1, distribution="normal", zoom=0,
                 zdistribution="normal", zskew=0.5):
        if size is not None:
            assert (isinstance(size, numbers.Number) and size > 0) or isinstance(size, iterable)
            if isinstance(size, iterable):
                for _s in size:
                    assert isinstance(_s, numbers.Number) and _s > 0
        self.size = size
        self.center_choice = center_choice
        self.shift = shift
        self.distribution = distribution
        self.zoom = zoom
        self.zdistribution = zdistribution
        self.zskew = zskew

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor, labels
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.center_crop(data, args["size"], args["center_choice"], args["shift"],
                             args["distribution"], args["zoom"], args["zdistribution"],
                             args["zskew"])

    def __repr__(self):
        return _make_repr(self)

class TwoCrop(object):
    """Crops the given Torch to return
    Modifications over the original:
        size optional, if no size, crop to minimum edge
        acts on tensors
    Args:
        size (sequence or int, optional): Desired output size of the crop.
            If size is an int instead, a square crop (size, size) is made.
            If size is None, square crop of length of min dimension is made.
            Default (None)
        TODO
        shift
            shift (string, optional), random/randn
    Returns:
        tensor, target_tensor
    """
    __type__ = "TensorCompose"

    def __call__(self, data):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        return F.two_crop(data)#, self.size)

    def __repr__(self):
        return _make_repr(self)


# ### Pyramid Crop
# class PyramidCrop(object):
#     """Crop the given PIL Image into four corners and the central crop plus the flipped version of
#     these (horizontal flipping is used by default)
#     .. Note::
#          This transform returns a tuple of images and there may be a mismatch in the number of
#          inputs and targets your Dataset returns. See below for an example of how to deal with
#          this.
#     Args:
#         size (int): all crops are square
#         depth (int): number of subdivisions
#         overlap (float): amount of overlap on neighbor crops (should be small)

#     Example:
#          >>> transform = Compose([
#          >>>    PyramidCrop(size, depth, overlap), # list of PIL images
#          >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
#          >>> ])# returns a 4D tensor
#          >>> #In your test loop you can do the following:
#          >>> input, target = batch # input is a Nd tensor, target is 2d
#          >>> bs, ncrops, c, h, w = input.size()
#          >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
#          >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
#     """

#     def __init__(self, size, depth=2, overlap=0.1):
#         self.size = size
#         self.depth = depth
#         self.overlap = overlap
#         assert (isinstance(size, numbers.Number)), "Please provide only one dimension for size."

#     def __call__(self, tensor, target_tensor=None):
#         """
#         Args:
#             tensor        (tensor): Image to be cropped, format NCL, NCHW, NCHW
#             target_tensor (tensor): annotation tensor
#             Returns:
#         tensor, target_tensor
#         """
#         return F.pyramid_crop(tensor, target_tensor, self.size, self.depth, self.overlap)

#####
#
# Appearance Transforms
#
# TODO Saturate and Gamma to have the same set of options!
class Saturate(object):
    """Manipulates image saturation
        Saturate to 0 is equivalent to Desaturate
        Saturate to -1 inverts image saturations
        Saturate to 2 over saturates image modulated modulated with piecewise tanh
        Args:
            sat_a:          (float) saturation target
            sat_b:          (float, None) random saturation between _a and _b
            p:              (float, 1) 0-1 bernoulli probability of action
            distribution:   (str, default None), None takes distribution mode from config.py
                                "normal", "uniform", "bernoulli"

    """
    __type__ = "Appearance"
    def __init__(self, sat_a, p=1, sat_b=None, distribution=None, independent=1, inplace=True):
        self.sat_a = sat_a
        self.sat_b = sat_b
        self.p = p
        self.distribution = distribution
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.saturate(data, args["sat_a"], args["sat_b"], args["p"],
                          args["distribution"], args["independent"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class RGBToLab(object):
    """To CIELab 1931 space, utilizing von kriess transform
        L: <0, 100>   ~ normal distribution
        A: <100, 100> ~ laplace dist
        B: <100, 100> ~ laplace dist
    to return in range compose
    R2LN = at.Compose([.RGBToLab(), .NormToRange()])
    Args
        illuminant (str) Default:  'D65' (daylight)
            valid in['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E',
                     'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
        observer (int)  Default: 2 (as perceived 2degrees view angle)
            valid [2, 10]

    """
    __type__ = "Appearance"
    def __init__(self, illuminant="D65", observer=2, p=1, independent=0, inplace=True):
        self.illuminant = illuminant
        self.observer = observer
        self.p = p
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.rgb2lab(data, args["illuminant"], args["observer"], args["p"],
                         args["independent"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class LabToRGB(object):
    """To CIELab 1931 space, utilizing von kriess transform

    Args
        illuminant (str) Default:  'D65' (daylight)
            valid in['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'E',
                     'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
        observer (int)  Default: 2 (as perceived 2degrees view angle)
            valid [2, 10]
    """
    __type__ = "Appearance"
    def __init__(self, illuminant="D65", observer=2, p=1, independent=1, inplace=True):
        self.illuminant = illuminant
        self.observer = observer
        self.p = p
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        args, _ = update_kwargs(self, **kwargs)

        return F.lab2rgb(data, args["illuminant"], args["observer"], args["p"],
                         args["independent"], args["inplace"])
    def __repr__(self):
        return _make_repr(self)

class ColorShift(object):
    """Shift color in CIELab 1931 space, utilizing von kriess transform
    Color Shift can match:
    * Standard Illuminants
    * Another Image
    * Statistic of a Dataset (Full ImageNet)
    To match color space this function utilizes reverse method to greyworld algorithm,
    it assumes that clipped values in the high and low ranges best represent the color of light
    As with the grayworld, it is approximate and will be distorted by object color
    Im not sure if that has been described but I find it more effective than grayworld
    and comparable to color chart linear transormations

    The basic math is derived from colorscience/colour
     Computations utilize CIExyz and xy color spaces
     Random distributions are drawn from CIELab space
     * CIELab is based on 1931 and 1964 measurements of retinal color perception
        where, L: Luminance, a: Green-Red, b: Blue-Yellow
     * CIExyz is nonnegative transformation of CIELab, where x~a, y~L, z~b
     * xy is is a normalized version xyz, xy = xyz/y
    For clarity's sake nomenclature for describing datasets is Lab
    Lab color transforms are desirable over RGB because they dont introduce quantization in the
    histogram, since color is defined by more than one channel at a time.

    Args:
        target  (list, tuple, ndarray, str or image) determines the distribution mean
            if list, tuple or ndarray: interprets target as xyz values
            if str, utilizes either:
                standard illuminant in group
                    ['A','B','C','D50','D55','D60','D65','D75','E','F1','F2',
                    'F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']
                or Datasets
                    ['imagenet'] # TBD, 'WIDER',
            if tensor, extract illuminant from image whites

        luma    (bool, default True), alter the brightness

        p       (float 0-1) bernoulli probability that image will get color transform
                    if p == 0 matches target transform exactly

        distribution    (str, default normal)
                            "normal"    normal distribution in imagenet range
                            "imagenet"  matching imagenet distribution (approx Laplacian)
                            "uniform"   uniform distribution in imagenet range

        # low importance optional: how chroma is calculated
        low     (float default 0.2), utilizes values lower than 'low' to calculate gray
        high    (float default 0.9), utilizes values higher than 'high' to calculate gray

    Examples:
    >>> Cs = at.ColorShift()

    # normal distributed color transformation matching default values
    >>> Cs(data, target=None, independent=1, distribution="normal", luma=True, p=0.9)

    # color transformation matching ImageNet Lab distribution in chroma only on 1/2 elements
    >>> Cs(data, target=None, independent=1, distribution="imagenet", luma=False, p=0.5)

    # color transformation matching 'data1' for 'a' and 'b' channels of Lab
    >>> Cs(data, target=data1, luma=False, p=0)

    # color transformation matching daylight color temperature
    >>> Cs(data, target="D65", luma=False, p=0)
    """
    __type__ = "Appearance"
    def __init__(self, p=1, target=None, luma=True, distribution="normal",
                 independent=1, inplace=True, **kwargs):
        self.target = F.get_chroma(target)
        self.luma = luma
        self.p = p
        self.distribution = distribution
        self.independent = independent
        self.inplace = inplace
        self.low = 0.2 if "low" not in kwargs else kwargs["low"]
        self.high = 0.95 if "high" not in kwargs else kwargs["high"]

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        args, _ = update_kwargs(self, **kwargs)
        if "target" in kwargs:
            args["target"] = F.get_chroma(args["target"])

        return F.color(data, args["p"], args["target"], args["distribution"],
                       args["independent"], args["inplace"], args["luma"],
                       args["low"], args["high"])

    def _resolve_chroma(self, target):
        """Returns tuple, normalized chroma, luma"""
        return F.get_chroma(target)

    def __repr__(self):
        return _make_repr(self)

class Noise(object):
    """
        Args:
            blend       (float > 0 [0.15])  amount of noise added
            scale       (int, tuple >= 1     [1])  upscale noise
                if tuple scale in uniform range between scales
            p           (float 0-1  [0.5])  bernoulli chance of noise application
            mode        (int 0-3      [0])
                    # modes 0 1 and 2 need revisiting
                        0 muladd, attenuated at low scales
                        1 add, uniformly distributed acros all scales
                        2 gaussmod mul add, attenuated at high and low tones
                    3 lerp: blend 0-> 0
                    4 shuffle: blend 0 -> 1
            gray        (bool)     [True])  apply noise over only one channel
            independent (int, 0-1     [1])  independence over minibatch
            clamp       (bool [False])  clamps result to 1, 0
            inplace     (bool      [True])  acts in place
    """
    __type__ = "Appearance"
    def __init__(self, p=1, blend=0.15, scale=1,
                 mode=3, gray=True, independent=1,
                 clamp=False, inplace=True):
        self.p = p
        self.blend = blend
        self.scale = scale
        self.mode = mode
        self.gray = gray
        self.independent = independent
        self.clamp = clamp
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.noise(data, args["blend"], args["scale"], args["p"], args["mode"],
                       args["gray"], args["independent"], args["clamp"], args["inplace"])
    def __repr__(self):
        return _make_repr(self)

class HighResponse(object):
    """Composits grayscale, inverse grayscale and midtones of an image to maximize response
    Args:
        p           (float 0-1 [0.1])   bernoulli chance of transform
        independent (int, 0-1    [1])   independence over minibatch
        shuffle     (bool     [True])   shuffle channels
        inplace     (bool     [True])   acts in place
    """
    __type__ = "Appearance"
    def __init__(self, p=1, independent=1, shuffle=True, inplace=True):
        self.p = p
        self.independent = independent
        self.shuffle = shuffle
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.high_response(data, args["p"], args["independent"], args["shuffle"],
                               args["inplace"])
    def __repr__(self):
        return _make_repr(self)

class Gamma(object):
    """ Changes gamma deterministically or randomly
    Args:
        g           (float >0  [2.2])   input image gamma
        a           (float >0  [1.0])   target gamma a
        b           (float > 0 [None])   target gamma b, if None int between input and a with prob p
                        b is float > 0, continuous sample distribution

        p           (float 0-1 [0.1])   bernoulli chance of transform
        distribution(str  ["normal"])   "normal", "uniform", "multinoulli", None
                        None:       discrete between a and b, (or if b is None, a if p, else None)

        independent (int, 0-1    [1])   independence over minibatch
        inplace     (bool     [True])   acts in place
    """
    __type__ = "Appearance"
    def __init__(self, p=1, g=2.2, a=1.0, b=None, independent=1, distribution="normal",
                 inplace=True, verbose=False):
        self.g = g
        self.a = a
        self.b = b
        self.p = p
        self.distribution = distribution
        self.independent = independent
        self.inplace = inplace
        self.verbose = verbose

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.gamma(data, args["g"], args["a"], args["b"], args["p"],
                       args["distribution"], args["independent"], args["inplace"],
                       args["verbose"])

    def __repr__(self):
        return _make_repr(self)
    
class Clamp(object):
    """ Soft Clamp implementation /piecewise tanh between 1 and 0
    applies independently per item of a batch

    TODO: min, max
    Args
        soft    (bool [True]), if False is equivalent to torch.clamp()
        inplace (bool [True])
    """
    __type__ = "Appearance"
    def __init__(self, soft=True, independent=True, inplace=True):
        self.soft = True
        self.independent = independent
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.softclamp(data, soft=args["soft"], independent=args["independent"],
                           inplace=args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class Mix(object):
    """impmements mixAugment utilizing images from the same batch augmentded and mixed
    Args
        p       (float 0-1, [0.5]), probability image will get mixaugment
        inplace (bool [True])
    """
    __type__ = "Appearance"
    def __init__(self, p=1, inplace=True):
        self.p = p

        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.mix(data, args["p"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class FTClamp(object):
    """ clamp of lower range of fourier transform
        Args
            p       (float, 0-1)    bernouilli probability of effect
            both    (bool [True])   clamp both parts or only real part
            inplace (bool [True])
    """
    __type__ = "Appearance"
    def __init__(self, p=1, both=True, inplace=True):
        self.p = p
        self.both = both
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.ftclamp(data, args["p"], args["both"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class Mask(object):
    """ outputs a BW mask
    Args
        low (float 0-1), choose values below
        high (float 0-1), choose values above
        inplace
    Example
    >>> Mk = self.Mask()
    >>> # mask middle range
    >>> MK(x, low=0.3, high=0.6) # returns 0 for 0.6 > x > 0.3, 1 elsewhere
    >>> # mask upper and lower range
    >>> MK(x, low=0.6, high=0.3) # returns 1 for 0.6 > x > 0.3, 0 elsewhere
    """
    __type__ = "Appearance"
    def __init__(self, low=0.0, high=0.7, inplace=True):
        self.low = low
        self.high = high
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)
        return F.mask(data, args["low"], args["high"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

class Blur(object):
    """ blur image using a PSF on fourier space
    change, anisotropy and rotation
    Args:
        p       (float 0-1 [1]), bernoulli probability of blur
        x       (float 0-1 [0.0]) image percentage blur  (default 0%)
        y       (float 0-1 [0.05]) image percentage blur (default 5%)
        dx      (float 0-1 [0.o]) range of random blur
        dy      (float 0-1 [0.0]) range of random blur
        angle   (float [pi/4]) range of blur angle
        da      (float >= 0 [3]) range of random blur
        distribution (str ["uniform"]) normal or unirom
        independent (int [1]) as batch or independently
        inplace (bool [True])
        percent_units (bool [False]), False
    """
    __type__ = "Appearance"
    def __init__(self, p=1, x=0.0, y=0.05, dx=0., dy=0., angle=0, da=np.pi/2,
                 distribution="uniform", independent=1, clamp=False, inplace=True,
                 visualize=False):
        self.p = p
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.angle = angle
        self.da = da
        self.distribution = distribution
        self.independent = independent
        self.clamp = clamp
        self.inplace = inplace
        self.visualize = visualize

    def __call__(self, data, **kwargs):
        args, kw = update_kwargs(self, **kwargs)

        return F.blur(data, args["p"], args["x"], args["y"], args["dx"], args["dy"],
                      args["angle"], args["da"], args["distribution"], args["independent"],
                      args["clamp"], args["inplace"], args["visualize"])

    def __repr__(self):
        return _make_repr(self)
    
class Glow(object):
    """
    Takes upper register and blurs it over image
    Args
        p
    """
    __type__ = "Appearance"
    def __init__(self, p=1, threshold=0.7, blend=0.8, scale=1.75, clamp=False, inplace=True):
        self.p = p
        self.threshold = threshold
        self.blend = blend
        self.scale = scale
        self.clamp = clamp
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in, interpolated
                labels
            **kwargs, any of the __init__ args can be overridden
        Returns:
            tensor, target_tensor
        """
        args, kw = update_kwargs(self, **kwargs)

        return F.glow(data, args["p"], args["threshold"], args["blend"],
                      args["scale"], args["clamp"], args["inplace"])
    def __repr__(self):
        return _make_repr(self)
#####
#
# halftones
#

# class Halftone(object):
#     """
#     """

#     def __init__(self, scale=1, p=0.9, shift=0., inplace=False):

#         self.scale = scale
#         self.p = p
#         self.shift = shift
#         self.inplace = inplace

#     def __call__(self, data, **kwargs):
#         """
#         Args:
#             data: tuple of
#                 tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
#                 target_tensor (tensor): annotations in, interpolated
#                 labels
#             **kwargs, any of the __init__ args can be overridden
#         Returns:
#             tensor, target_tensor
#         """

#         args, kw = update_kwargs(self, **kwargs)
#####
#
# Normalizations
#


class Normalize(object):
    """ similar to torchvision.transforms.Normalize
    difference:
        float indicates target standard deviation

    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean    (sequence [None], float)
                    sequence:   per channel mean
                    None:       ImageNet mean
                    int, float: target mean
        std     (sequence [None], float)
                    sequence:   per channel stdev
                    None:       ImageNet  stdev
                    int, float: target stdev

        inplace (bool, [True])
    """
    __type__ = "Appearance"
    def __init__(self, mean=None, std=None, inplace=True):
        self.mean = self._set_mean(mean)
        self.std = self._set_std(std)
        self.inplace = inplace

    def _set_std(self, std):
        if std is None:
            return [0.23530918, 0.23156014, 0.23460476]
        _valid = (int, float, list, np.ndarray, torch.Tensor)
        assert isinstance(std, _valid), "found %s, expected %s"%(type(std), _valid)
        return std

    def _set_mean(self, mean):
        if mean is None:
            return [0.4993829, 0.47735223, 0.42281782]
        _valid = (int, float, list, np.ndarray, torch.Tensor)
        assert isinstance(mean, _valid), "found %s, expected %s"%(type(mean), _valid)
        return mean

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be scaled, format NCL, NCHW, NCHW
                target_tensor (tensor): annotations in format config.BOXMODE, interpolated
            **kwargs    any argument from __init__, locally
        Returns:
            tensor, target_tensor, labels
        """
        args, kw = update_kwargs(self, **kwargs)
        return F.normalize(data, args["mean"], args["std"], args["inplace"])

    def __repr__(self):
        return _make_repr(self)

# alias to normalize
# with better name
MeanCenter = Normalize

class UnNormalize(object):
    """UnNormalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will unnormalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] + mean[channel]) * std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace (bool [False])
        clip    (bool [False]) if True, clip between 0 and 1
    """
    __type__ = "Appearance"
    def __init__(self, mean=None, std=None, inplace=False, clip=False):
        self.mean = mean if mean is not None else [0.4993829, 0.47735223, 0.42281782]
        self.std = std if std is not None else [0.23530918, 0.23156014, 0.23460476]
        self.inplace = inplace
        self.clip = clip

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be unnormalized, format NCL, NCHW, NCHW
                target_tensor (tensor): annotation tensor; no action
        Return:
            tensor, target_tensor, labels
        """
        args, kw = update_kwargs(self, **kwargs)
        return F.unnormalize(data, args["mean"], args["std"], args["inplace"], args["clip"])

    def __repr__(self):
        return _make_repr(self)

# alias to unnormalize
# with better name
MeanUnCenter = UnNormalize

class NormToRange(object):
    """map tensor linearly to a range
        Args:
            minimum     (float [0.]) min value of normalization
            maximum     (float [1.]) max value of normalization
            excess_only (bool [False]) when True leave images within range untouched
            independent (bool [True]) when True normalize per item in batch
            per_channel (bool [True]) when True normalize per channel
            inplace     (bool [True])
    """
    __type__ = "Appearance"
    def __init__(self, minimum=0.0, maximum=1.0, excess_only=False, independent=True,
                 per_channel=True, inplace=False):
        self.minimum = minimum
        self.maximum = maximum
        self.excess_only = excess_only
        self.independent = independent
        self.per_channel = per_channel
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        """
        Args:
            data: tuple of
                tensor        (tensor): Image to be normalized to range, format NCL, NCHW, NCHW
                target_tensor (tensor): annotation tensor; no action
        Returns:
            tensor, target_tensor
        """

        args, kw = update_kwargs(self, **kwargs)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(dtype=torch.float32)
        elif isinstance(data[0], np.ndarray):
            data[0] = torch.from_numpy(data[0]).to(dtype=torch.float32)

        return F.norm_to_range(data, args["minimum"], args["maximum"], args["excess_only"],
                               args["independent"], args["per_channel"], args["inplace"])
    def __repr__(self):
        return _make_repr(self)
