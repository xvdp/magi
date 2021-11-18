"""@ xvdp
functional for  transforms.__type__ = "IO"

"""
from typing import Union, Optional
import os.path as osp
from urllib.parse import urlparse
import numpy as np
import torch
import torchvision.transforms as TT

from .. import config
from ..features import Item
from ..utils import open_img, check_contiguous, show_tensor, closest_square

# pylint: disable=no-member
def open_file(file_name: Union[str, list, tuple],
              dtype: str,
              device: Union[str, torch.device] = "cpu",
              grad: bool = False,
              out_type: str = "torch",
              channels: int = None,
              transforms: TT = None,
              verbose: bool = False) -> Union[torch.Tensor, np.ndarray, list]:
    """
    Args    file_name   (str, list), file, url or list of files and urls
                if list, and images same size, concatente, numpy or tensor, else list
            dtype       (str) if torch: floating_point types, if numpy': [uint8, float32, float64]

            device      (torch.device ['cpu'])  # torch only
            grad        (bool [False])  # torch only

            out_type    (str ['torch']) | numpy
            channels    (int [None: same as input]) | 1,3,4
            transforms  (torchvision.transforms)
    """
    if isinstance(file_name, (list, tuple)):
        batchlist = []
        size = None
        _concat = []

        for i, _file in enumerate(file_name):
            tensor = open_file(_file, dtype=dtype, device=device, grad=grad, out_type=out_type,
                               channels=channels, transforms=transforms, verbose=verbose)

            if tensor is not None:
                if i == 0:
                    size = tensor.shape

                _concat.append(size == tensor.shape)
                batchlist.append(tensor)

        if all(_concat):
            if out_type == "torch":
                tensor = torch.cat(batchlist, dim=0)
                return check_contiguous(tensor, verbose)
            elif out_type == "numpy":
                return np.stack(batchlist, axis=0)

        return batchlist

    assert osp.isfile(file_name) or urlparse(file_name).scheme, "filename not found"
    tensor = open_img(file_name, out_type=out_type, dtype=dtype, grad=grad, device=device,
                      backend=None, channels=channels, transforms=transforms)
    return tensor

###
#
# functional for Show()
#
def show(data: Union[list, tuple, torch.Tensor, np.ndarray],
         ncols: Optional[int] = None,
         pad: int = 0,
         show_targets: bool = False,
         target_mode: Union[config.BoxMode, str] = "xywh",
         width: int = 20,
         height: Optional[int] = 10,
         save: Optional[str] = None,
         unfold_channels: bool = False,
         **kwargs) -> None:
    """ data cam be
            Item comprised of [tensors, targets, ...]
            list of Items, tensors, ndarrays
            tensor
            ndarray
    """
    figsize = (width, height)

    if isinstance(data, (torch.Tensor, np.ndarray)):
        targets = None if 'targets' not in kwargs or not show_targets else kwargs.pop('targets')
        show_tensor(data, targets, target_mode, save=save, figsize=figsize, unfold_channels=unfold_channels,
                    ncols=ncols, pad=pad, **kwargs)

    elif isinstance(data, (list, tuple)):
        if isinstance(data, Item):
            tensors, targets, modes = item_to_tensor_targets(data, target_mode, show_targets)
        else:
            tensors, targets, modes = [], [], []
            if all(isinstance(item, Item) for item in data):
                for item in data:
                    _tensors, _targets, _modes = item_to_tensor_targets(item, target_mode,
                                                                        show_targets)
                    tensors += _tensors
                    targets += _targets
                    modes += _modes
            else:
                for x in data:
                    if isinstance(x, torch.Tensor, np.ndarray):
                        if (torch.is_tensor(x) and x.ndim==4) or (isinstance(x, np.ndarray) and x.ndim==3):
                            tensors.append(x)
                        else:
                            targets.append(x)
                    elif isinstance(x, str):
                        modes.append(x)

        subplots = (1, 1)
        if len(tensors) > 1:
            if ncols is None:
                subplots = closest_square(len(tensors))
            else:
                subplots = (len(tensors)//ncols + int(bool(len(tensors)%ncols)))

        for i, x in enumerate(tensors):
            _kw = {'figsize': figsize, 'show': False, 'subplot':(*subplots, i+1)}
            if i > 0:
                _kw['figsize'] = None
            if i == len(tensors) - 1:
                _kw['show'] = True
                _kw['save'] = save

            target = None if i >= len(targets) else targets[i]
            mode = None if i >= len(modes) else modes[i]

            show_tensor(x, target, mode, unfold_channels=unfold_channels, pad=pad,
                        **_kw, **kwargs)

def item_to_tensor_targets(data: Item, default: str = 'xyhw', show_targets: bool = True) -> tuple:
    """ unfolds item in triplet tensors, targets, formats
        targets, formats can be empty
    """
    tensors = data.get(kind="data_2d")
    assert len(tensors) > 0, "no tensors found"

    targets, formats = [], []
    if show_targets:
        targets, formats = get_2d_targets(data, default)
    return tensors, targets, formats



def get_2d_targets(data: Item, default: str = 'xyhw') -> tuple:
    """ returns tuple of lists, [target tensor], [format str]
    """
    idxs = data.get_indices(kind='pos_2d')
    formats = [data.form[i] for i in idxs] if 'form' in data.keys else [default]*idxs
    return [[data[i] for i in idxs]], [formats]

"""

show_tensor(x: Union[np.ndarray, torch.Tensor],
                targets: Union[None, np.ndarray, torch.Tensor],
                target_mode: str = 'xywh',
                figsize: Optional[tuple] = (10,10),
                subplot: Optional[tuple] = None,
                show: bool = True,
                save: Optional[str] = None,
                unfold_channels: bool = False,
                **kwargs): ncols, pad background

    data could be 
    tensor
    ndarray
    Item
    list of tensors. ndarray, items


      if isinstance(data, np.ndarray):
            _div = 1.0 if data.dtype != np.uint8 else 255.
            data = torch.from_numpy(data).to(dtype=torch.float32).div_(_div)

        elif isinstance(data[0], np.ndarray):
            _div = 1.0 if data[0].dtype != np.uint8 else 255.
            data[0] = torch.from_numpy(data[0]).to(dtype=torch.float32).div_(_div)


"""


# def show(data: Union[list, tuple, torch.Tensor, np.ndarray],
#          ncols: Optional[int] = None,
#          pad: int = 1,
#          show_targets: bool = True,
#          annot, width, height:
#          save: Optional[str] = None, as_box,
#          max_imgs: Optional[int] = None,
#          unfold_channels: bool = True,
#          **kwargs) -> None:
#     """ functional for transforms.Show()
#         Args
#             data    (ndarray img, list)   # image can be 4,3 or 2d
#                     ([ndarray img, [ndarray annot], ndarray tgts])
#                     (tensor img)

#                     ([tensor img, tensor_list annot, tensor tgts])           < one image with annotations
#                     ([tensor, ..., tensor] imgs)                             < list of images without anotations
#                     ([[tensor img, tensor_list annotd, tensor tgts], ...,[img anns, tgts]]
#     """
#     pass
#     # if "mode" in kwargs:
#     #     config.set_boxmode(kwargs["mode"])

#     allow_dims = [1, 3, 4]
#     # mode = config.BOXMODE if "mode" not in kwargs else kwargs["mode"]

#     if isinstance(data, np.ndarray):
#         showim(data, targets=None, labels=None, show_targets=show_targets, annot=annot, width=width,
#                height=height, path=path, as_box=as_box, **kwargs)
#         return

#     if isinstance(data[0], np.ndarray):
#         targets = None if len(data) == 1 else data[1]
#         labels = None if len(data) < 3 else data[2]
#         showim(data[0], targets=targets, labels=labels, show_targets=show_targets, annot=annot,
#                width=width, height=height, path=path, as_box=as_box, **kwargs)
#         return

#     # if "bbox" in kwargs and isinstance(data, (list, tuple)) and len(data) > 1:
#     #     if data[1] is not None and len(data[1]) > 0:
#     #         box = targets_bbox(data[1])
#     #         data = addbbox(data, box)

#     # list containing single batch of tensors
#     if isinstance(data, (tuple, list)) and len(data)==1:
#         data = data[0]
    
#     # list of single image tensors 1,C,H,W or C,W,H
#     if is_tensor_image_list(data):

#         images = []
#         if ncols is None:
#             ncols = min(len(data), 8)
#         images = []
#         for _data in data:
#             image, targets, labels = to_numpy(_data, ncols, pad=pad, allow_dims=allow_dims,
#                                               mode=mode)
#             images.append(image)

#         showims(images, targets=None, labels=None, show_targets=show_targets, annot=annot,
#                 width=width, height=height, path=path, as_box=as_box,
#                 unfold_channels=unfold_channels, ncols=ncols, **kwargs)

#     # if list of lists containing equal image tensors
#     elif is_tensor_batch(data):
#         if ncols is None:
#             ncols = min(len(data[0]), 8)
#         images, targets, labels = data_to_numpy(data=data, pad=pad, max_imgs=max_imgs, ncols=ncols,
#                                                 width=width, inplace=False, **kwargs)

#         showim(images, targets, labels, show_targets=show_targets, annot=annot, width=width,
#                height=height, path=path, as_box=as_box, unfold_channels=unfold_channels,
#                ncols=ncols, **kwargs)

#     elif torch.is_tensor(data) and data.dtype in (torch.float32, torch.uint8):

#         if ncols is None:
#             ncols = min(len(data), 8)
#         images, targets, labels = data_to_numpy(data=data, pad=pad, max_imgs=max_imgs, ncols=ncols,
#                                                 width=width, inplace=False, **kwargs)

#         showim(images, targets, labels, show_targets=show_targets, annot=annot, width=width,
#                height=height, path=path, as_box=as_box, unfold_channels=unfold_channels, **kwargs)
