"""@xvdp
"""
import inspect
from typing import TypeVar, Union
import os.path as osp

import torch
from torch.utils.data.dataset import Dataset, IterableDataset
import torchvision.transforms as TT

from koreto import Col

from ..transforms import Open
from ..utils import dtype_as_str
from ..config import load_dataset_path, write_dataset_path

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


# pylint: disable=no-member
class DatasetM(Dataset[T_co]):
    """ Image Dataset: Similar to torch Dataset, with extensions to use magi methods
    Args
        name        (str "")        # self.name = f"{self.__class__.__name__}_{name}"
            useful in instances of synthetic datasets or dataset subsets
        dtype       (str, torch.dtype [None])   # None: will open data with torch default_dtype, OR with dtype specified in dataset `Item().dtype=[]`
            setting dtype explicitly will set torch default dtype
        device      (str, torch.device ["cpu"]) # will return items with all tensors on the selected device -
            augments on dataloader will be faster but may saturate device. use with caution
        inplace     (bool [True])   # inplace augments use smaller footprint - but are incompatible with differentiable augmentation.
            since by more useual augmentation on dataloading is not required to be differentiable, by default is set to True
        grad        (bool [False])  # if set to True, tensors will be opened with grad - `inplace` will be overriden to -> False
        channels    (int [3]) |1|3|4|None      # by default Image datasets are RGB, channels==3 will force 3 channels on dsets with inconsistent image channels
            None will open images with the channels they were saved
            # TODO remove to generalize DatasetMagi for data other than images

        transforms  (torchvision.transform)
    """
    def __init__(self, name: str="", dtype: Union[str, torch.dtype]=None,
                 device: Union[str, torch.device]="cpu", inplace: bool=True,
                 grad: bool=False, channels: int=3, transforms: TT=None):

        self.name = f"{self.__class__.__name__}_{name}"
        self.dtype = dtype_as_str(dtype if dtype is not None else torch.get_default_dtype())
        self.device = device

        self.open = Open(out_type="torch", dtype=self.dtype, device=device, grad=grad,
                         inplace=inplace, channels=channels, transforms=transforms,
                         force_global=True)

        self.samples = []

    def get_dataset_path(self, path: str, *paths: str) -> str:
        """ keep dataset path registry easy and updated
        storing new paths if provided, deleting dead paths
        returning fastest path if more than one in config
        """
        if path is None:
            path = load_dataset_path(self.__class__.__name__, *paths)
        elif osp.isdir(path):
            write_dataset_path(self.__class__.__name__, path, *paths)
        assert path is not None and osp.isdir(path), f"{Col.YB}{self.__class__.__name__} data_root '{path}' not found, pass valid data_root arg{Col.AU}"
        return path


    def _make_dataset(self, **kwargs) -> None:
        NotImplementedError("Need to be implemented for dataset")

    def __repr__(self, exclude_keys: Union[list, tuple]=None) -> str:
        """ utility, auto __repr__()
        Args
            exclude_keys    (list, tuple [None])
        """
        rep = self.__class__.__name__+"("
        for i, (key, value) in enumerate(inspect.signature(self.__init__).parameters.items()):
            if key == "self" or (exclude_keys is not None and key in exclude_keys):
                continue

            if isinstance(value.default, str):
                value = f"='{value.default}'"
            elif not isinstance(value.default, type):
                value = f"={value.default}"
            elif key in self.__dict__:
                value = f"='{self.__dict__[key]}'"
            elif '*' in f"{value}":
                key = ""
            else:
                value = ""

            sep = "" if not i else ", "
            rep += f"{sep}{key}{value}"
        return rep + ")"

class AnnotatedDataset(Dataset[T_co]):
    """ torch Dataset extended to tag different elements
    """
    def __load_annotations__(self, annotation_file) -> None:
        raise NotImplementedError


    def __getitem__(self, index) -> T_co:
        raise NotImplementedError



class AnnotatedIterableDataset(IterableDataset[T_co]):
    """ torch Dataset extended to tag different elements
    """
    def __load_annotations__(self, annotation_file) -> None:
        raise NotImplementedError


    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
