"""@xvdp
wrapper to wider dataset with face annotations

"""
from typing import Union
import os.path as osp
import numpy as np
import torch
from .datasets import MDataset
from ..containers import DataItem


# pylint: disable=no-member
class WIDER(MDataset):
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
    >>> W.__getitem__([index]) -> returns magi.features.DataItem()


    >>> W.set_filters()
    """
    def __init__(self, data_root:str, mode: str="train", dtype: str="float32", device: str="cpu",
                 torch_transforms=None, filters: dict=None, **kwargs) -> None:
        """ Args
            data_root   (str) path where dset was uncompressed
            mode        (str [train]) | val | test
            dtype       (str [float32])
            device      (str [cpu]) | cuda
            filters     (list [None]) if None __getitem__() returns all tags
                if filters is passed, {"tags":['image', 'bbox']} and any tags in filters= returned
                in ['name', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose',
                    'activity', 'wider_id',  'wordnet_id', 'wordnet_face_id']
        """
        super().__init__(dtype=dtype, device=device, inplace=True, grad=False, channels=3,
                         torch_transforms=torch_transforms)

        self.data_root = data_root
        assert osp.isdir(data_root), f"cd {data_root} not found"

        self.images = []
        self.dtype = dtype
        self.device = device
        self.load_dset(mode=mode)
        self.filters = None
        if filters is not None:
            self.set_filters(**filters)

    def __getitem__(self, index:int=None) -> DataItem:
        """Return item as DataItem()

        .tags['image', 'bbox', 'name', 'blur', 'expression', 'illumination', 'invalid', 'occlusion',
               'pose', 'activity', 'wider_id',  'wordnet_id', 'wordnet_face_id']
        """
        index = index if index is not None else torch.randint(3, (1,)).item()
        item =  self.images[index]
        image = self.open(item.get("meta", "path")[0])
        item.insert(0, image, meta="data_2d", tags="image")
        if self.filters is not None:
            item.keep(**self.filters)
        item.to_torch(device=self.device)
        return item

    def __len__(self) -> int:
        return len(self.images)

    def set_filters(self, **kwargs):
        if kwargs:
            self.filters = {"tags":["image", "bbox"]} # always return image nd bound box
            self.filters.update({k:kwargs[k] for k in kwargs if k in ['tags', 'meta']})
        else:
            self.filters = None

    def load_dset(self, mode: str="train") -> None:
        """
        .load_dset(mode=<'train' | 'val' | 'test'>)
        """
        _modes = ("train", "val", "test")
        assert mode in _modes, f"invalid mode '{mode}'; select from {_modes}"

        _paths = {"annotations":"wider_face_split", "images":osp.join(f"WIDER_{mode}", "images")}
        self.paths = {k:osp.join(self.data_root, _paths[k]) for k in _paths}
        for key, path in self.paths.items():
            assert osp.isdir(path), f"{key} path '{path}' not found..."

        if mode == "test":
            image_list = osp.join(self.paths['annotations'], "wider_face_test_filelist.txt")
        else:
            image_list = osp.join(self.paths['annotations'], f"wider_face_{mode}_bbx_gt.txt")

        self.images = self.read_annotations(image_list, self.paths['images'], mode=mode)


    def read_annotations(self, image_list: list, folder: str, mode: str="train") -> list:
        """ returns list of DataItem

        Bounding Boxes are groupd per image for

        with keys for entries
            .tags
            .meta   entry type: position_2d, path, class_name, class_id, attr_id

        class_id includes both WIDER class_id, and Wordnet class_id
            eg, in 9--Parade, class_name = "Parade", wider_id = 9, wordnet_id = 8428485
        if annotation file not found, download and place in path

        """
        # dataset tags
        # with open(osp.join(self.paths['annotations'], f"readme.txt"), 'r') as _fi:
        #     dset_tags = _fi.read().split("\n")[-1].split(", ")
        dset_tags = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid',
                'occlusion', 'pose']

        # read image list
        assert osp.isfile(image_list), f"image list file {image_list} not found, download annotations from <http://shuoyang1213.me/WIDERFACE/>"
        with open(image_list, 'r') as _fi:
            text = _fi.read().split("\n")
        ## train and val images, format, e.g.
        # 9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg
        # 2
        # 336 242 152 202 0 0 0 0 0 0
        # 712 278 126 152 0 0 0 0 0 0

        images = []
        i = 0
        _wn_face = self._wnid_dict('Face')
        while len(text):
            name = text.pop(0)
            if not name:
                continue

            # activity category from folder, e.g. 9--Press_Conference
            dirname = osp.dirname(name)
            _wnid = self._wnid_dict(dirname) # wordNet ID
            _wid_class = dirname.split("--")

            if len(_wid_class) < 2:
                print(f"name missing?: -> {name}")
                continue

            _wid = int(_wid_class[0])
            _class = "--".join(_wid_class[1:])

            name = osp.join(folder, name)
            _dtags = {t:[] for t in dset_tags[4:]}

            if mode not in "test" and text[0].strip().isnumeric():
                _num_faces = int(text.pop(0).strip())

                bboxes = []
                for _ in range(_num_faces):
                    this_box = np.fromstring(text.pop(0), dtype=self.dtype, sep=" ")
                    bboxes.append(this_box[:4].reshape(2,2))
                    attrs = this_box[4:].astype(int)
                    for k, tag in enumerate(_dtags):
                        _dtags[tag].append(attrs[k])

                bboxes = np.stack(bboxes, axis=0)

                item = DataItem([bboxes, name, *_dtags.values(), _class, _wid, _wnid[0], _wn_face],
                                tags=["bbox", "name", *_dtags.keys(), "activity", "wider_id",
                                      "wordnet_id", "wordnet_face_id"],
                                meta=["position_2d", "path", *["attr_id" for _a in range(len(_dtags))],
                                      "class_name", "class_id", "class_id", "class_id"],
                                )
                images.append(item)
            else:
                item = DataItem([name], tags=["name"], meta=["path"])
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
