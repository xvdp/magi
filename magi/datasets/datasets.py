"""@xvdp
"""
from typing import TypeVar
from ..transforms import Open
from magi import transforms

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

from torch.utils.data.dataset import Dataset, IterableDataset


class MDataset(Dataset[T_co]):
    """ Similar to torch Dataset, with extensions to use magi methods.
    """
    def __init__(self, dtype="float32", device="cpu", inplace=True, grad=False, channels=3,
                 torch_transforms=None):

        self.open = Open(out_type="torch", dtype=dtype, device=device, grad=grad,
                         inplace=inplace, channels=channels, transforms=torch_transforms,
                         force_global=True)

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
