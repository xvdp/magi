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
from typing import Any, Union
import logging
import torch
import torchvision.transforms as TT
from koreto import Col
from ._transforms import Transform, TransformAppearance
from . import functional_io as Fio
from .. import config
from ..utils import warn_grad_inplace, warn_np_dtypes

# pylint: disable=no-member
#
# IO Transforms
#
class Open(Transform):
    """Open a filename as torch tensor, not "properly a transform"

    Args:
        out_type:   (str ["torch"]) | 'numpy']
        inplace:    (bool [None]) if None uses config.INPLACE
                                  if (True | False) sets namesace globals -> config.INPLACE
                    is overriden by grad == True

    Args modifiable on __call__:
        dtype       (str | torch.dtype [None])
            if None uses torch.get_default_dtype()
            if not None: changes torch.default_dtype
        channels:   (int [3]) | 1| 4 | None: if None defaults to what is stored
        transforms: (torchvision transforms) [None]

        torch only:
        device      (str ["cpu"]) | "cuda"
        grad        (bool [False]): if True, forces config.INPLACE == False
        force_global    (bool [False]) if set to True, changes, namespace globals for DTYPE

    Default Return:
        Tensor, float32, NCHW, 3 channles, no_grad, cpu
    """
    __type__ = "IO"
    def __init__(self, dtype: Union[str, torch.dtype]=None, device: str="cpu", grad: bool=False,
                 inplace: bool=None, out_type: str="torch", channels: int=3, transforms: TT=None,
                 force_global: bool=False) -> Any:

        # out_type can only be set on __init__
        self.out_type = out_type if out_type in ("torch", "numpy") else "torch"
        self._force_global = force_global

        self.dtype = self._check_valid({"dtype": dtype})["dtype"]
        self.channels = self._check_valid({"channels": channels})["channels"]
        self.transforms = transforms

        if self.out_type == "torch":
            self.device = self._check_valid({"device": device})["device"]
            self.grad = grad
            warn_grad_inplace(inplace, grad, in_config=True)

    def __call__(self, name: Union[str, list, tuple], **kwargs):
        """
        Args:
            name        (str, list),  valid file name(s)
            **kwargs    arguments from __init__()  except out_type
        """
        kw_call, _ = self.update_kwargs(**kwargs)
        kw_call = self._check_valid(kw_call)
        return Fio.open_file(name, **kw_call)

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

        if "grad" in kwargs: # -> NoReturn, ensure !grad or inplace != grad
            warn_grad_inplace(config.INPLACE, kwargs["grad"], in_config=True)

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
        kw_call, kw_ = self.update_kwargs(**kwargs)
        if 'w' in kw_ and 'width' not in kwargs:
            kw_call['width'] = kw_['w']

        print(kw_call)
    

        # if isinstance(data, np.ndarray):
        #     _div = 1.0 if data.dtype != np.uint8 else 255.
        #     data = torch.from_numpy(data).to(dtype=torch.float32).div_(_div)
    
        # elif isinstance(data[0], np.ndarray):
        #     _div = 1.0 if data[0].dtype != np.uint8 else 255.
        #     data[0] = torch.from_numpy(data[0]).to(dtype=torch.float32).div_(_div)

        # return F.show(data, ncols=args["ncols"], pad=args["pad"], show_targets=args["show_targets"],
        #               annot=args["annot"], width=width, height=args["height"],
        #               path=args["path"], as_box=args["as_box"], max_imgs=args["max_imgs"],
        #               unfold_channels=args["unfold_channels"], **kw)
