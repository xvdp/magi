"""@xvdp """
from typing import Union
from nltk.corpus import wordnet
import torch
import torchvision.transforms as TT
from koreto import Col
from .folder import DatasetFolder_M

List = Union[list, tuple]

# pylint: disable=no-member
class ImageNet(DatasetFolder_M):
    """ Tested with ImageNet1000 ILSVCR_2012
    """
    def __init__(self, data_root=None, mode: str="train", name: str="", subset: Union[List, int]=None,
                 ordered: int=0, names: list=['image', 'target_index'], extensions: str=".jpeg",
                 dtype: Union[str, torch.dtype]=None, device: Union[str, torch.device]="cpu",
                 inplace: bool=True, grad: bool=False, channels: int=3, transforms: TT=None):

        _modes = ("train", "test", "val")
        assert mode in _modes, f"{Col.YB}ImageNet invalid mode='{mode}', expected {_modes}{Col.AU}"
        super().__init__(data_root, mode=mode, name=name, subset=subset, ordered=ordered,
                         names=names, extensions=extensions, dtype=dtype, device=device,
                         inplace=inplace, grad=grad, channels=channels, transforms=transforms)

        # ensure we have imagenet typically somethng like
        # train/
        #   n01440764/
        #       n01440764_10026.JPEG
        #       ...
        # val/
        #   n01440764/
        #       ILSVRC2012_val_00000293.JPEG
        #       ILSVRC2012_val_00000293.xml
        #       ...
        # test/
        #   n01440764_10026.JPEG
        #   ...
        # if osp.basename(self.data_root) in ("train", "val")

    def _get_target_names(self) -> None:
        self.target_names = []
        if self.classes:
            for wni in self.classes:
                tgt = wordnet.synset_from_pos_and_offset(wni[0], int(wni[1:])).lemma_names('eng')[0]
                self.target_names.append(tgt)
