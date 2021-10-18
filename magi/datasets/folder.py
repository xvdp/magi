"""@xvdp
DatasetFolder, classification datset where labels are associate to subfolders

.__getitem__() -> magi.features.Item()

"""
from typing import Union, Any
import os
import os.path as osp
import random
import numpy as np
import torch
from torchvision import transforms as TT

from magi.features.list_util import list_flatten, list_transpose

from .datasets import DatasetM
from ..features import Item, tolist, list_subset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

List = Union[tuple, list]
LooseList = Union[tuple, list, str]

# pylint: disable=no-member
class DatasetFolderM(DatasetM):
    """ similar torchvision.datasets.DatasetFolder using Magi methods

    
    Builds list self.samples = [Item(), ...]
    .__getItem__() returns Item()

    parses a folder with subfolders
    folder/
        <subfolder0>/
            <filen><ext> for <ext> in arg 'extensions'
        ...
        <subfoldern>/

    : assigning
    self.classes = [<subfolder0>, ..., <subfoldern>]
    self.class_to_idx = {<subfolder0>:0, ..., <subfoldern>:n}

    : if _get_class_names() method implemente
    self.class_names = [<name0>, ..., <namen>]

    : if arg 'ordered' samples cycle thru classes with 'ordered' num samples per class per cycle

    : if arg 'subset', only a subset of the classes is made into dataset

    Args
        data_root   (str) root directory of dataset
        mode        (str [""]) subfolder under which classes are stored, e.g. 'train', 'val' ...
        name        (str [""]) -> self.name = f"__class__.__name__{name}"
        subset      (list, int [None]) ->  returns a subset dataset with classes
            list of foldernames | list of indices | int random set of n folders
        ordered     (int [0]) -> if > 0 orders samples cycling classes with n 'ordered; elements per class
        extensions  (list|str) extensions considered in class folders
        dtype       (str [torch.get_default_dtype()])
        device      (str [cpu])
        inplace     (bool [True]) # augmenters run in place
        grad        (bool [False]) # sets inplace to False, for differentiable augments
        channels    (int [3]) | 1,4,None: if None, opens images as stored
        transforms  (torchvision.transforms)

    """
    def __init__(self, data_root, mode: str="", name: str="", subset: Union[List, int]=None, ordered: int=0, tags: list=None,
                 extensions: LooseList=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                 dtype: Union[str, torch.dtype]=None, device: Union[str, torch.device]="cpu",
                 inplace: bool=True, grad: bool=False, channels: int=3, transforms: TT=None):


        super().__init__(name=name, dtype=dtype, device=device, inplace=inplace, grad=grad,
                         channels=channels, transforms=transforms)

        self.data_root = osp.join(osp.abspath(osp.expanduser(data_root)), mode)
        assert osp.isdir(self.data_root), f"dataset root {self.data_root} not found"

        self.ext = [ext.lower() for ext in tolist(extensions)]
        self.ordered = ordered

        self.classes, self.class_to_idx = self._get_classes(self.data_root, subset)
        self.class_names = self._get_class_names(self.classes) # default: None


        # Item: what we want to pass to the learner
        # Item components

        self._tags = ['image', 'name', 'image_index', 'target_folder', 'target_index']
        self._meta = ['data_2d', 'path', 'image_id', 'class_name', 'class_id']
        self._dtype = [self.dtype, 'str', 'int', 'str', 'int']

        _tags = [1,-1]

        tags = [] if tags is None else [t for t in self._tags[1:-1] if t in tags]
        self.tags = ["image", *tags, "target_index"]  # for classification we need image, and target


        if self.class_names is not None:
            self._tags.insert(-1, "class")
            self._meta.insert(-1, "class_name")
            self._dtype.insert(-1, "str")

        self._make_dataset()

    def __getitem__(self, index:str=None) -> Item:
        """
        returns data[index(None)] if None, randint
        TODO apply seed to ensure data parallel

        Args:
            index (int): Index

        Returns:
            [sample, ..., label]
            or [sample, annotation, ..., label]
            or [sample, ..., index, label]
        """
        index = index if index is not None else torch.randit(0, len(self), (1,)).item()

        item =  self.samples[index].deepcopy()

        path_idx = item.get_indices("meta", "path")[0]
        path_name = item[path_idx] if self.tags is None or "name" in self.tags else item.pop(path_idx)

        image = self.open(path_name)
        item.insert(0, image, meta="data_2d", tags="image", dtype=self.dtype)
        item.to_torch(device=self.device)

        return item

    def __len__(self):
        return len(self.samples)

    def _get_classes(self, folder:str=None, subset: Union[List, int]=None) -> tuple:
        """
        Returns tuple (classes list, class:index dict).
        Args:
            folder  (str)  root directory path.
            subset  (list | int [None]) subset from classes_names, class_indices, num classes
        """
        classes = sorted([d.name for d in os.scandir(folder) if d.is_dir()])
        classes = list_subset(classes, subset)
        # reverse dict
        classes_idx = {classes[i]: i for i in range(len(classes))}
        return classes, classes_idx

    def _make_data_item(self, file_name, image_id, class_name, class_id):
        data = [file_name]
        if "image_index" in self._tags:
            data += [image_id]
        if "target_folder" in self._tags:
            data += [class_name]
        if "class" in self._tags:
            data += self.class_names[class_id]
        if "target_index" in self.tags:
            data += class_id

        return Item(data, tags=self._tags, meta=self._meta, dtype=self._dtype)

    def _make_dataset(self) -> None:
        """
        """
        self.samples = None # delete whatever was there
        samples = []
        class_sample_count = []

        for class_id, class_name in enumerate(self.classes):

            class_samples = []
            folder = os.path.join(self.data_root, class_name)
            files = sorted([f.path for f in os.scandir(folder)
                            if f.is_file() and osp.splitext(f.name)[1].lower() in self.ext])
            j = sum(class_sample_count)

            for i, file_name in enumerate(files):
                class_samples.append(self._make_data_item(file_name, i+j, class_name, class_id))

            class_sample_count.append(len(files))
            samples.append(class_samples)

        if self.ordered:
            self.samples = list_transpose(samples, self.ordered)
        else:
            self.samples = list_flatten(samples)


    @staticmethod
    def _get_class_names(classes: List) -> list:
        class_names = []
        ## implementation of mapping between folders and names
        ## eg. with ImageNet
        # from nltk.corpus import wordnet
        # class_names = []
        # for wnid in classes:
        #    class_names.append(wordnet.synset_from_pos_and_offset(wnid[0], int(wnid[1:])).lemma_names('eng')[0])
        return class_names

