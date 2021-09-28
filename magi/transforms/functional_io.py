"""@ xvdp"""
import os.path as osp
from urllib.parse import urlparse
import numpy as np
import torch
from ..utils import open_img, check_tensor, check_contiguous

# pylint: disable=no-member
def open_file(file_name, dtype, device, grad=False, out_type="torch", channels=None, transforms=None, verbose=False):
    """
    """
    if device == "cpu" and dtype == "float16":
        dtype = "float32"
        print(f"cpu does not support half, opening as float32")

    ## TODO fix accimage for uint8
    if out_type == "numpy":
        dtype = "uint8"


    if isinstance(file_name, (list, tuple)):
        batchlist = []
        size = None
        _covert_to_tensor = out_type == "torch"
        for i, _file in enumerate(file_name):
            tensor = open_file(_file, dtype=dtype, device=device, grad=grad, out_type=out_type,
                               channels=channels, transforms=transforms, verbose=verbose)

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

    # if config.DEBUG:
    #     _size = check_tensor(tensor)
    #     # if config.MEM is None:
    #     #     config.MEM = CudaMem(config.DEVICE)
    #     # config.MEM.report("open_file, sz: %d"%(_size/2**20))
    #     _msg = "open_file, sz: %dMB"%(_size/2**20)
    #     mem.report(memory=config.MEM, msg=_msg)
    #     config.MEMORY += _size

    if out_type == "torch":
        tensor.unsqueeze_(0)
        tensor = check_contiguous(tensor, verbose)
