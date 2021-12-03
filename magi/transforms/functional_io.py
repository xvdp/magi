"""@ xvdp
functional for  transforms.__type__ = "IO"

"""
from typing import Union, Optional
import logging
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
# TODO complete port of annotations
# TODO enable showing multiple targets per image and target types
# showim(images, targets, labels, show_targets=show_targets, annot=annot, width=width,
#                height=height, path=path, as_box=as_box, unfold_channels=unfold_channels, **kwargs)
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
    figsize = (width, height) if "figsize" not in kwargs else kwargs.pop('figsize')
    targets = None if 'targets' not in kwargs or not show_targets else kwargs.pop('targets')

    if isinstance(data, Item):
        if show_targets and targets is None:
            idcs = data.get_indices(kind="pos_2d")
            if idcs:
                i = idcs[0]
                targets = data[i]
                target_mode = data.form[i]
                if len(idcs) > 1:
                    names = [data.names[i] for i in idcs]
                    logging.warning(f"more than one kind of pos data {names}, showing only: '{data.names[i]}'")
        data = data.get(kind="data_2d")[0]

    if isinstance(data, (torch.Tensor, np.ndarray)):
        show_tensor(data, targets, target_mode, save=save, figsize=figsize,
                    unfold_channels=unfold_channels, ncols=ncols, pad=pad, **kwargs)

    elif isinstance(data, (list, tuple)):
        num = len(data)
        if ncols is None:
            subplots = closest_square(num)
            ncols = subplots[1]
        else:
            ncols = min(ncols, num)
            subplots = (num//ncols + int(bool(num%ncols)), ncols)

        for i, x in enumerate(data):
            _kw = {'figsize': figsize, 'show': False, 'subplot':(*subplots, i+1)}
            if i > 0:
                _kw['figsize'] = None
            if i == num - 1:
                _kw['show'] = True
                _kw['save'] = save

            _kw.update({k:kwargs[k] for k in kwargs if k not in _kw})

            show(x, ncols=pad, pad=pad, show_targets=show_targets, target_mode=target_mode,
                 unfold_channels=unfold_channels, **_kw)

            # show_tensor(x, unfold_channels=unfold_channels, ncols=ncols, pad=pad,
            #             **_kw, **kwargs)



        # # if isinstance(data, Item):
        # #     tensors, targets, modes = item_to_tensor_targets(data, target_mode, show_targets)
        # tensors, targets, modes = [], [], []
        # if all(isinstance(item, Item) for item in data):
        #     for item in data:
        #         _tensors, _targets, _modes = item_to_tensor_targets(item, target_mode,
        #                                                             show_targets)
        #         tensors += _tensors
        #         targets += _targets
        #         modes += _modes
        # else:
        #     for x in data:
        #         if isinstance(x, torch.Tensor, np.ndarray):
        #             if (torch.is_tensor(x) and x.ndim==4) or (isinstance(x, np.ndarray) and x.ndim==3):
        #                 tensors.append(x)
        #             else:
        #                 targets.append(x)
        #         elif isinstance(x, str):
        #             modes.append(x)

        # subplots = (1, 1)
        # if len(tensors) > 1:
        #     if ncols is None:
        #         subplots = closest_square(len(tensors))
        #     else:
        #         subplots = (len(tensors)//ncols + int(bool(len(tensors)%ncols)))

        # for i, x in enumerate(tensors):
        #     _kw = {'figsize': figsize, 'show': False, 'subplot':(*subplots, i+1)}
        #     if i > 0:
        #         _kw['figsize'] = None
        #     if i == len(tensors) - 1:
        #         _kw['show'] = True
        #         _kw['save'] = save

        #     target = None
        #     mode = None
        #     if i < len(targets):
        #         target = targets[i]
        #         mode = modes[i]

        #     _ncols = ncols if ncols is None else min(len(x), ncols)
        #     show_tensor(x, target, mode, unfold_channels=unfold_channels, ncols=_ncols, pad=pad,
        #                 **_kw, **kwargs)

def item_to_tensor_targets(data: Item, default: str = 'xyhw', show_targets: bool = True) -> tuple:
    """ unfolds item in triplet list(tensors), list(list(targets)), list(list(formats))
    An Item can have M data items: N,C,...      -> [tensor0, tensor1, ...]
    each data item, can have several targets    -> [[target0a, target0b, ...], [target1a, ...], ...]
    each target with its own mode               -> [[mode0a, mode0b, ...], [mode1a, , ...]
        and multiple targets of different modes, eg, boxes, paths,

    item to tensor_targets
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
