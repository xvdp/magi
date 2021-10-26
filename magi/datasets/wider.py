"""@xvdp
wrapper to wider dataset with face annotations

"""
from typing import Union
import os.path as osp
import numpy as np
import torch
from torchvision import transforms as TT
from tqdm import tqdm
from .datasets import Dataset_M
from ..features import Item

# pylint: disable=no-member
class WIDER(Dataset_M):
    """ Wider Dataset (from http://shuoyang1213.me/WIDERFACE/)
    Annotated faces on crowds on a rangeof 20 activities.

    ..info
    download wider_face_split.zip, WIDER_train, WIDER_test, WIDER_val
    uncompress to a root folder e.g. WIDER
    WIDER/
        wider_face_split/
            ## annotations for train and val images, format, e.g.
                9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg # filename
                2                           # number of faces
                336 242 152 202 0 0 0 0 0 0 # x y w h blur expression illumination invalid occlusion pose
                712 278 126 152 0 0 0 0 0 0 # ...
        WIDER_train/
        WIDER_val/
        WIDER_test/ # image without annotations


    ..usage
    >>> W = WIDER(data_root="/media/z/Elements1/data/Face/WIDER")
    >>> W.__getitem__([index]) -> returns magi.features.Item()


    """
    def __init__(self, data_root :str=None, mode: str="train", names: list=None, name: str="",
                 dtype: str=None, device: str="cpu", for_display: bool=False, grad: bool=False,
                 channels: int=3, transforms: TT=None, **kwargs) -> None:
        """
        Args
            data_root   (str) path where dset was uncompressed
                if None it searches and .magi path registry and
                otherwise writes to .magi path registry
            mode        (str [train]) | val | test

            names        (list [None]) if None __getitem__() returns all names in self._names dataset definition
                if names is not None, items will return names ["bbox", "name"] on daset samples and ["image", "bbox"]
                and any other specified in `names=[]` arg available to WIDER dataset:
        ['blur','expression','illumination','invalid','occlusion','pose', 'index', # WIDER per bbox attributes
         'activity", 'wider_id', 'wordnet_id'] # folder name -> activity name, number, and wordnet noun

        Args from DatasetMagi
            name, dtype, device, for_display, grad, channels, transforms

        """
        super().__init__(name=name, dtype=dtype, device=device, for_display=for_display, grad=grad,
                          channels=channels, transforms=transforms)

        self.data_root = self.get_dataset_path(data_root)

        self.samples = []

        # dataset definition
        # names available to dataset

        self._names = ['image', 'bbox', 'name',
                      'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose',
                      'index', 'wider_activity', 'wider_id',  'wordnet_id']

        self._meta = ['data_2d', 'positions_2d', 'path',
                      'attr_id', 'attr_id', 'attr_id', 'attr_id', 'attr_id', 'attr_id',
                      'image_id', 'class_name', 'class_id', 'class_id']

        self._dtype = [self.dtype, self.dtype, "str",
                       'uint8', 'bool', 'bool', 'bool', 'uint8', 'bool',
                       'int', 'str', 'uint8', 'int']

        self.names = self._filter_names(names)
        self._make_dataset(mode=mode)

    def __getitem__(self, index:int=None) -> Item:
        """Return item as Item()
        """
        index = index if index is not None else torch.randint(len(self), (1,)).item()
        item =  self.samples[index].deepcopy()

        path_idx = item.get_indices(meta="path")[0]
        path_name = item[path_idx] if self.names is None or "name" in self.names else item.pop(path_idx)
        image = self.open(path_name) # open dtype, device and torch transforms from parent class
        item.insert(0, image, meta="data_2d", names="image", dtype=self.dtype)
        item.to_torch(device=self.device)

        return item

    def __len__(self) -> int:
        return len(self.samples)

    def _filter_names(self, names):
        """ names .names[]
        """
        if names is not None:
            names = names if isinstance(names, list) else [names]
        return names

    def _make_dataset(self, mode: str="train") -> None:
        """ .load_dset(mode=<'train' | 'val' | 'test'>)
        """
        _modes = ("train", "val", "test")
        assert mode in _modes, f"invalid mode '{mode}'; select from {_modes}"

        # validate paths: images
        image_folder = osp.join(self.data_root, f"WIDER_{mode}", "images")
        assert osp.isdir(image_folder), f"images not found in '{image_folder}'..., download images and unzip"

        # validate paths: annotations
        _name = "wider_face_test_filelist.txt" if mode == "test" else f"wider_face_{mode}_bbx_gt.txt"
        image_list_file = osp.join(self.data_root, "wider_face_split", _name)
        assert osp.isfile(image_list_file), f"annotations not found in '{image_list_file}'..., download and unzip"

        # read annotations to Item list
        self.samples = self._read_annotations(image_list_file, image_folder)

    def _read_annotations(self, image_list_file: list, folder: str) -> list:
        """ returns list of Item
            Bounding Boxes are grouped per image for augmentation [N,M,2,2] in format x,y,w,h
            where N, is batch size, M, number of faces per image,

        with keys for entries
            .names   entry name: image, bbox, name, *attributes, activity, wider_id, wordnet_id
            .meta   entry type: data_2d, position_2d, path, *[attr_id]*6, class_name, *[class_id]*3
            .dtype  entry dtype: self.dtype, self.dtype, str, int, str, uint8, int, int, int

        class_id includes both WIDER class_id, and Wordnet class_id
            eg, in 9--Parade, class_name = "Parade", wider_id = 9, wordnet_id = 8428485
        if annotation file not found, download and place in path

        # with open(osp.join(self.paths['annotations'], f"readme.txt"), 'r') as _fi:
        #     dset_names = _fi.read().split("\n")[-1].split(", ")
        # attrs = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid',
        #         'occlusion', 'pose']
        # ['x1', 'y1', 'w', 'h'] -> bbox
        # ['blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose'] -> attributes
        """
        # read image list
        with open(image_list_file, 'r') as _fi:
            text = _fi.read().split("\n")

        images = []
        i = 0

        # sample names
        names = self._names[1:]
        meta = self._meta[1:]
        dtype = self._dtype[1:]

        _keep_indices = None
        if self.names is not None:
            _immutable_names = ["bbox", "name"]
            _keep_indices = [i for i in range(len(names)) if names[i] in self.names + _immutable_names]
            names = [names[i] for i in _keep_indices]
            meta = [meta[i] for i in _keep_indices]
            dtype = [dtype[i] for i in _keep_indices]

        # ['blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']
        attrnames = self._names[3:9]

        while i < len(text) - 1:
            line = text[i]
            if not line:
                i += 1
                continue

            # class from filename
            class_name = osp.basename(osp.dirname(line))
            wordnet_id = self._wnid_dict(class_name)[0]  # wordnet_id e.g. 7144834
            class_name = class_name.split("--")
            assert (class_name[0].strip()).isnumeric(), f"<{class_name}> is not numeric!, line {i}"
            class_id = int(class_name[0])               # class_id # e.g. 9
            class_name = "--".join(class_name[1:])      # class_name # e.g. Press_Conference
            # Item values
            values = [len(images), class_name, class_id, wordnet_id]

            # filename
            name = osp.join(folder, line)
            i += 1

            # bboxes and attributes
            num_faces = text[i]

            if num_faces.strip().isnumeric():
                num_faces = int(num_faces)
                i += 1

                bbox = []
                attributes = []
                for _ in range(num_faces):
                    face = np.fromstring(text[i], dtype="int", sep=" ")
                    bbox.append(face[:4])
                    attributes.append(face[4:])
                    i += 1

                # bbox: N, M, nb_pos, nb_values
                # nb_pos: 2: pos, offset
                # nb_values: 2: x,y or w,h
                bbox = np.stack(bbox, axis=0).reshape(1, -1, 2, 2)
                attributes = np.stack(attributes, axis=0)
                # per attribute: N, M
                attrs = {attrnames[i]:attributes[:, i].reshape(1,-1) for i in range(len(attrnames))}

                # Item values
                values = [bbox, name, *attrs.values()] + values
                if _keep_indices is not None:
                    values = [values[i] for i in _keep_indices]

            else: # images without bboxes
                values = [name] + values
                idcs = [i for i in range(len(names)) if names[i] not in ["bbox"]+attrnames]
                if len(images) == 0:
                    names = [names[i] for i in idcs]
                    meta = [meta[i] for i in idcs]
                    dtype = [dtype[i] for i in idcs]

            images.append(Item(values, names=names, meta=meta, dtype=dtype))
        return images

    @staticmethod
    def _wnid_dict(name: str=None) -> list:
        wnids = {
            'Face': [5600637],
            '0--Parade': [8428485, 8460395],
            '1--Handshaking': [6632097],
            '2--Demonstration': [1177703],
            '3--Riot': [1170502, 13977043],
            '4--Dancing': [428270],
            '5--Car_Accident': [2958343, 7301336],
            '6--Funeral': [7451463],
            '7--Cheering': [7251779],
            '8--Election_Campain': [181781, 7472929],
            '9--Press_Conference': [7144834],
            '10--People_Marching': [290579],
            '11--Meeting': [8307589, 8310389, 1230965],
            '12--Group': [31264],
            '13--Interview': [7196075],
            '14--Traffic': [8425303],
            '15--Stock_Market': [4323026],
            '16--Award_Ceremony': [13268146],
            '17--Ceremony': [7450842],
            '18--Concerts': [6892775],
            '19--Couple': [7988857],
            '20--Family_Group': [8078020, 7970406],
            '21--Festival': [517728],
            '22--Picnic': [7576438],
            '23--Shoppers': [10592397],
            '24--Soldier_Firing': [10622053, 123234],
            '25--Soldier_Patrol': [8216176],
            '26--Soldier_Drilling': [10622053],
            '27--Spa': [8678615, 4080705],
            '28--Sports_Fan': [10639925],
            '29--Students_Schoolkids': [10665698],
            '30--Surgeons': [10679174],
            '31--Waiter_Waitress': [10763383, 10763620],
            '32--Worker_Laborer': [9632518, 10241300],
            '33--Running': [293916],
            '34--Baseball': [471613],
            '35--Basketball': [480993],
            '36--Football': [469651],
            '37--Soccer': [478262],
            '38--Tennis': [482298],
            '39--Ice_Skating': [448640],
            '40--Gymnastics': [433802],
            '41--Swimming': [442115],
            '42--Car_Racing': [449517],
            '43--Row_Boat': [445351, 2858304],
            '44--Aerobics': [625858],
            '45--Balloonist': [9835348],
            '46--Jockey': [10223177],
            '47--Matador_Bullfighter': [9836519],
            '48--Parachutist_Paratrooper': [10397482],
            '49--Greeting': [6630017],
            '50--Celebration_Or_Party': [7450651, 8252602],
            '51--Dresses': [3236735],
            '52--Photographers': [10426749],
            '53--Raid': [976953],
            '54--Rescue': [93483],
            '55--Sports_Coach_Trainer': [9931640],
            '56--Voter': [10760340],
            '57--Angler': [9793946],
            '58--Hockey': [467995, 463543],
            '59--people--driving--car': [7942152],
            '61--Street_Battle': [958896]}

        if name is not None and name in wnids:
            return wnids[name]
        return wnids

# class FeaturesDict({
#     'faces': Sequence({
#         'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
#         'blur': tf.uint8,
#         'expression': tf.bool,
#         'illumination': tf.bool,
#         'invalid': tf.bool,
#         'occlusion': tf.uint8,
#         'pose': tf.bool,
#     }),
#     'image': Image(shape=(None, None, 3), dtype=tf.uint8),
#     'image/filename': Text(shape=(), dtype=tf.string),
# })
