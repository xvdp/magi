"""@ xvdp"""
from typing import Any, Union
import os.path as osp
from urllib.parse import urlparse
import numpy as np
import torch
import torchvision.transforms as TT
from .. import config
from ..utils import open_img, check_tensor, check_contiguous

# pylint: disable=no-member
def open_file(file_name: Union[str, list, tuple], dtype: str, device: Union[str, torch.device]="cpu",
              grad: bool=False, out_type: str="torch", channels: int=None,
              transforms: TT=None, verbose: bool=False) -> Union[torch.Tensor, np.ndarray, list]:
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

    # if verbose:
    #     #print("config verbose", config.VERBOSE)
    #     if out_type == "numpy":
    #         print("Open(): F.open_file(): '%s' as numpy"%(file_name))
    #     print("Open(): F.open_file(): '%s' using device '%s'"%(file_name, device))

    tensor = open_img(file_name, out_type=out_type, dtype=dtype, grad=grad, device=device,
                      backend=None, channels=channels, transforms=transforms)

    # if config.DEBUG:
    #     _size = check_tensor(tensor)
    #     # if config.MEM is None:
    #     #     config.MEM = CudaMem(config.DEVICE)
    #     # config.MEM.report("open_file, sz: %d"%(_size/2**20))
    #     _msg = "open_file, sz: %dMB"%(_size/2**20)
    #     mem.report(memory=config.MEM, msg=_msg)
    #     config.MEMORY += _size

    if out_type == "torch" and tensor is not None:
        if config.INPLACE:
            tensor.unsqueeze_(0)
        else:
            tensor = tensor.unsqueeze(0)
        tensor = check_contiguous(tensor, verbose)
    return tensor
