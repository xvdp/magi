"""@ xvdp

Agumentation library with syntax derived to torchvision.transforms (TT).
Input to transformations are batches has to contain at least one NCHW torch.Tensor.
    * NCHW Image_Tensor

It can also contain a full batch as output my a Dataloader, containing Image and Index tensors
    * [NCHW Image_Tensor, N Index_Tensor]

Or batches with position data associated to those tensors, g.g. bounding boxes or segmentation paths
    * [NCHW Image_Tensor, N[M,2,2] Annotation_Tensor_List,  N Index_Tensor]

Annotation Tensor Lists will be transformed by Affine

"""
from typing import Union, Optional
from collections.abc import Callable
import logging
import numpy as np
import torch
from torch.distributions import Bernoulli
from koreto import Col

from .transforms_base import Transform, is_magi_transform
from . import functional_io as F
from .. import config
from ..utils import warn_grad_cloning, warn_np_dtypes


_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Image = Union[torch.Tensor, np.ndarray, list]

# pylint: disable=no-member
#
# IO Transforms
#
class Open(Transform):
    """Open a filename as torch tensor, not "properly a transform"

    Args:
        out_type:       (str ["torch"]) | 'numpy']


    Args modifiable on __call__:
        dtype       (str | torch.dtype [None])
            if None uses torch.get_default_dtype()
            if not None: changes torch.default_dtype
        channels:   (int [3]) | 1| 4 | None: if None defaults to what is stored
        transforms: (torchvision transforms) [None]

        torch only:
        device      (str ["cpu"]) | "cuda"
        grad        (bool [False]): if True, forces config.FOR_DISPLAY == False
        for_display:    (bool [None]) if None uses config.FOR_DISPLAY
                if (True | False) sets namesace globals -> config.FOR_DISPLAY
                for_display, clones tensors on every augment
                is overriden by grad == True
        force_global    (bool [False]) if set to True, changes, namespace globals for DTYPE

    Default Return:
        Tensor, float32, NCHW, 3 channles, no_grad, cpu
    """
    __type__ = "IO"
    def __init__(self,
                 dtype: Union[None, str, torch.dtype] = None,
                 device: str = "cpu",
                 grad: bool = False,
                 for_display: Optional[bool] = None,
                 out_type: str = "torch",
                 channels: int = 3,
                 transforms: Optional[Callable] = None,
                 force_global: bool = False) -> None:
        super().__init__(for_display=for_display)

        # out_type can only be set on __init__
        self.out_type = out_type if out_type in ("torch", "numpy") else "torch"
        self._force_global = force_global

        self.dtype = self._check_valid({"dtype": dtype})["dtype"]
        self.channels = self._check_valid({"channels": channels})["channels"]
        self.transforms = transforms

        if self.out_type == "torch":
            self.device = self._check_valid({"device": device})["device"]
            self.grad = grad
            warn_grad_cloning(for_display, grad, in_config=True)

    def __call__(self, name: Union[str, list, tuple], **kwargs) -> _Image:
        """
        Args:
            name        (str, list),  valid file name(s)
            **kwargs    arguments from __init__()  except out_type
        """
        kw_call = self.update_kwargs(**kwargs)
        kw_call = self._check_valid(kw_call)
        return F.open_file(name, **kw_call)

    def _check_valid(self, kwargs):
        # assert on out_type change
        # cant change out_type on __call__ only on init
        assert "out_type" not in kwargs or self.out_type == kwargs["out_type"], f"{Col.YB} Open().__call__() cannot change from '{self.out_type}' to '{kwargs['out_type']}' -> create new Open(out_type='{kwargs['out_type']}'){Col.AU}"

        # validate/warn is float[16,32,64] if torch, or that and uint8 if numpy
        if "dtype" in kwargs: # -> str
            kwargs["dtype"] = config.resolve_dtype(kwargs["dtype"], self._force_global) if self.out_type == "torch" else warn_np_dtypes(kwargs["dtype"])

        # validate/warn that channels are either 1,3,4
        if "channels" in kwargs: # -> int
            if kwargs["channels"] not in [1, 3, 4, None]:
                logging.warning(f"{Col.YB} channels={kwargs['channels']} not allowedm,  only allowed in [1,3,4, None)-> setting to 3{Col.AU}")
                kwargs["channels"] = 3

        if "device" in kwargs: # -> torch.device
            kwargs["device"] = config.get_valid_device(kwargs["device"])

        if "grad" in kwargs or "for_display" in kwargs: # ensure !grad or for_display != grad
            for_display = kwargs["for_display"] if "for_display" in kwargs else config.FOR_DISPLAY
            if 'grad' in self.__dict__: # only on torch output
                grad = kwargs["grad"] if "grad" in kwargs else self.grad
                warn_grad_cloning(for_display, grad, in_config=True)

        return kwargs


class Show(Transform):
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
    def __init__(self,
                 ncols: Optional[int] = None,
                 pad: int = 0,
                 show_targets: bool = True,
                 target_mode: Union[config.BoxMode, str] = "xywh",
                 width: int = 20,
                 height: int = 10,
                 save: Optional[str] = None,
                 unfold_channels: bool = False, 
                 for_display: Optional[bool] = None) -> None:
        """
        TODO: cleanup removed args
                #  target_rotation: bool = True,
                #  annot: Optional[str] = None,
                #  max_imgs: int = 0,
                # self.target_rotation = target_rotation
                # self.annot = annot
                # self.max_imgs = max_imgs
        """
        super().__init__(for_display=for_display)
        self.ncols = ncols
        self.pad = pad
        self.show_targets = show_targets
        self.target_mode = target_mode
        self.width = width
        self.height = height
        self.save = save
        self.unfold_channels = unfold_channels

    def __call__(self, data: Union[list, torch.Tensor, np.ndarray], **kwargs) -> None:
        """
        Args:
            data        (torch tensor or tuple of tensors) NCHW
            **kwargs    any argument from __init__, locally
                        w shorthand for width

        """
        self.update_shortcut(('w', 'width'), kwargs)
        self.update_shortcut(('h', 'height'), kwargs)
        kw_call = self.update_kwargs(allow_extra=True, **kwargs)
        return F.show(data, **kw_call)

##
#
# Compose Transforms
#
class Compose(Transform):
    """Composes several transforms together, arguments are evaluated left to right.

    similar to torchvision.transforms.Compose, added probability of transfom
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
        p   (float >0, <=1 [1.0]) Bernoulli probability that any of the transforms will be performed
            (list, tuple same len as transforms), Bernoulli probability per transform
        ..note: transforms of __type__ "Sizing" or "IO" are always 1.

    Example:
    # center crop always, desaturate with prob of 50%
    >>> transforms.Compose([CenterCrop(224), Saturate(a=0)], p=(1, 0.5))
    """
    __type__ = "Compose"
    def __init__(self,
                 transforms: Union[list, tuple],
                 p: _tensorish = 1.0,
                 for_display: Optional[bool] = None) -> None:
        super().__init__(for_display=for_display)

        self.transforms = transforms
        self.p = Bernoulli(probs=self._get_probs(transforms, p))

    def __call__(self, data: Union[torch.Tensor, list], **kwargs) -> Union[torch.Tensor, list]:
        probs = self.p.sample()
        for i, transform in enumerate(self.transforms):
            if probs[i]:
                if not is_magi_transform and not torch.is_tensor(data):
                    # apply non magi transforms to tensor only
                    data[0] = transform(data[0])
                else:
                    data = transform(data)

        return data

    @staticmethod
    def _get_probs(transforms: Union[list, tuple], p: _tensorish) -> torch.Tensor:
        """ Returns valid tensor of probabilities
        validates transforms passed to Compose are 'magi'
        IO and Sizing always have a probability of 1.
        """
        assert isinstance(p, _tensorish), f"'p: expected {_tensorish} got {type(p)}"
        if isinstance(p, (float, int)):
            p = [p] * len(transforms)
        elif isinstance(p, tuple):
            p = list(p)

        assert len(p) == len(transforms), f"expected probability per transform, got {len(transforms)} transforms and {len(p)} probabilities"
        for i, transform in enumerate(transforms):
            if is_magi_transform and transform.__type__ in ("Sizing", "IO"):
                p[i] = 1
        return torch.clamp(torch.as_tensor(p, dtype=torch.float32), 0, 1)
