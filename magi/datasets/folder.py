"""@xvdp
DatasetFolder, classification datset where labels are associate to subfolders

.__getitem__() -> magi.features.Item()

"""
from typing import Union
import os
import os.path as osp
import torch
from torchvision import transforms as TT
from koreto import Col

from magi.features.list_util import list_flatten, list_transpose

from .datasets import Dataset_M
from ..features import Item, TypedItem, tolist, list_subset, list_removeall


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

List = Union[tuple, list]
LooseList = Union[tuple, list, str]

# pylint: disable=no-member
class DatasetFolder_M(Dataset_M):
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
        ordered     (int [0]) -> if > 0 cycles classes with n 'ordered; elements per class
        extensions  (list|str) extensions considered in class folders
        dtype       (str [torch.get_default_dtype()])
        device      (str [cpu])
        for_display (bool [False]) # augmenters are cloned
        grad        (bool [False]) # forces for_display to False, for differentiable augments
        channels    (int [3]) | 1,4,None: if None, opens images as stored
        transforms  (torchvision.transforms)

    """
    def __init__(self, data_root=None, mode: str="", name: str="", subset: Union[List, int]=None,
                 ordered: int=0, names: list=['image', 'target_index'],
                 extensions: LooseList=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                 dtype: Union[str, torch.dtype]=None, device: Union[str, torch.device]="cpu",
                 for_display: bool=False, grad: bool=False, channels: int=3, transforms: TT=None):


        super().__init__(name=name, dtype=dtype, device=device, for_display=for_display, grad=grad,
                         channels=channels, transforms=transforms)

        data_root = self.get_dataset_path(data_root)
        self.mode = mode
        self.data_root = osp.join(data_root, mode)
        assert osp.isdir(self.data_root), f"{Col.YB}'{self.data_root}' not found{Col.AU}"

        self.ext = [ext.lower() for ext in tolist(extensions)]
        self.ordered = ordered

        self.classes, self.class_to_idx = self._get_classes(self.data_root, subset)
        self.target_names = []
        self._get_target_names() # implement per dataset

        # if no classes, remove all class info
        if not self.target_names:
            list_removeall(names, 'target_name')
        if not self.classes:
            list_removeall(names, ['target_folder', 'target_index'])

        self.item = self._define_item(names)

        self._make_dataset()

    def _define_item(self, names=None):
        """
        """
        if names is None:
            names = ['image', 'target_index']
        self.keep_names = names.copy() # on __getitem__
        if 'filename' not in names:
            names += ['filename']

        # names, meta, dtype
        _elems = {'image': ['data_2d', self.dtype],
                  'filename': ['path', 'str'],      # filename
                  'image_index': ['id', 'int'],     # image index
                  'target_folder': ['name', 'str'], # class folder, e.g. n04557648
                  'target_name': ['name', 'str'],   # class name, e.g. 'water_bottle' # requires self.target_names
                  'target_index': ['id', 'int']}    # class index

        meta = [_elems[name][0] for name in names]
        dtype = [_elems[name][1] for name in names]

        return TypedItem(names, meta, dtype)

    def _make_item(self, **kwargs):
        """
        Filters kwargs by names defined in self._define_item()
        need to pass the correct items in self._make_dataset()
        """
        names = self.item.__dict__['names']
        data = [None]*len(names)
        for i, key in enumerate(names):
            if key in kwargs:
                data[i] = kwargs[key]
        return self.item.spawn(data)

    def __getitem__(self, index:str=None) -> Item:
        """
        Returns Item(data[index(None)], ...)
        Args:
            index (int): Index if None, randint
        """
        index = index if index is not None else torch.randint(0, len(self), (1,)).item()
        item =  self.samples[index].deepcopy()

        path_idx = item.get_indices(meta="path")[0]
        path_name = item[path_idx] if "filename" in self.keep_names else item.pop(path_idx)

        item[0] = self.open(path_name)
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

    def _make_dataset(self) -> None:
        """ collects images to make dataset
        if list of subfolders, each sub folder becomes a class
        if flatt list, no class information is included

        optionally
        can return a dataset ordered (arg, 'ordered' in __init__()) cycling classes
        or a subset of the classes (arg, 'subset' in __init__())
        """
        is_file = lambda x: x.is_file() and osp.splitext(x)[-1].lower() in self.ext
        get_files = lambda x: sorted([d.path for d in os.scandir(x) if is_file(d)])
        self.samples = None # delete whatever was there

        if not self.classes:
            self.samples = []
            for i, path in enumerate(get_files(self.data_root)):
                self.samples += [self._make_item(filename=path, image_index=i)]
        else:
            i = 0
            samples = []
            for class_id, class_name in enumerate(self.classes):
                class_samples = []
                for _, path in enumerate(get_files(osp.join(self.data_root, class_name))):
                    _itemkw = {"filename":path, "image_index":i, "target_folder":class_name, "target_index": class_id}
                    if "target_name" in self.item.__dict__['names']:
                        _itemkw["target_name"] = self.target_names[class_id]
                    class_samples.append(self._make_item(**_itemkw))
                    i += 1
                samples.append(class_samples)

            self.samples = samples
            if self.ordered:
                self.samples = list_transpose(samples, self.ordered)
            else:
                self.samples = list_flatten(samples, depth=1)


    def _get_target_names(self) -> None:
        self.target_names = []
        ## implementation of mapping between folders and names
        ## eg. with ImageNet
        # from nltk.corpus import wordnet
        #   self.target_names = []
        # if self.classes:
        #     for wni in self.classes:
        #         tgt = wordnet.synset_from_pos_and_offset(wni[0], int(wni[1:])).lemma_names('eng')[0]
        #         self.target_names.append(tgt)
