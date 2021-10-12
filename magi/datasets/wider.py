"""@xvdp
wrapper to wider dataset with face annotations

"""
from typing import Union
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
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
    def __init__(self, data_root:str, mode: str="train", dtype: str=None, device: str="cpu",
                 torch_transforms=None, keep: list=None, **kwargs) -> None:
        """ Args
            data_root   (str) path where dset was uncompressed
            mode        (str [train]) | val | test
            dtype       (str [float32])
            device      (str [cpu]) | cuda
            keep        (list [None]) if None __getitem__() returns all tags
                ["image", "bbox", "name"] # are always kept
                # if keep is passed, with any of tags below
                ['blur','expression','illumination','invalid','occlusion','pose', "activity", "wider_id","wordnet_id", "wordnet_face_id"]
        """
        super().__init__(dtype=dtype, device=device, inplace=True, grad=False, channels=3,
                         torch_transforms=torch_transforms)

        self.data_root = data_root
        assert osp.isdir(data_root), f"cd {data_root} not found"

        self.images = []
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device
        self.keep = keep

        self.load_dset(mode=mode)

    def __getitem__(self, index:int=None) -> DataItem:
        """Return item as DataItem()

        .tags['image', 'bbox', 'name', 'blur', 'expression', 'illumination', 'invalid', 'occlusion',
               'pose', 'activity', 'wider_id',  'wordnet_id', 'wordnet_face_id']
        """
        index = index if index is not None else torch.randint(3, (1,)).item()
        item =  self.images[index].copy()
        image = self.open(item.get("meta", "path")[0], dtype=self.dtype)
        item.insert(0, image, meta="data_2d", tags="image", dtype=self.dtype)
        item.to_torch(device=self.device)

        ## TODO - add transforms?
        return item

    def __len__(self) -> int:
        return len(self.images)

    def set_keep(self, *args):
        if not len(args) or args[0] is None:
            self.keep = None
        self.keep = list(set(["image", "bbox", "name"]  + args))

    def load_dset(self, mode: str="train") -> None:
        """
        .load_dset(mode=<'train' | 'val' | 'test'>)
        """
        _modes = ("train", "val", "test")
        assert mode in _modes, f"invalid mode '{mode}'; select from {_modes}"

        # validate paths: images
        image_folder = osp.join(self.data_root, f"WIDER_{mode}", "images")

        # validate paths: annotations
        image_list_file = osp.join(self.data_root, "wider_face_split")
        if mode == "test":
            image_list_file = osp.join(image_list_file, "wider_face_test_filelist.txt")
        else:
            image_list_file = osp.join(image_list_file, f"wider_face_{mode}_bbx_gt.txt")

        assert osp.isdir(image_folder), f"images not found in '{image_folder}'..., download images and unzip"
        assert osp.isfile(image_list_file), f"annotations not found in '{image_list_file}'..., download and unzip"

        # read annotations to DataItem list
        self.images = self.read_annotations(image_list_file, image_folder, mode=mode)


    def read_annotations(self, image_list_file: list, folder: str, mode: str="train") -> list:
        """ returns list of DataItem
            Bounding Boxes are grouped per image for augmentation purposes [N, 2,2] in format x,y,w,h

        with keys for entries
            .tags   entry name: image, bbox, name, attributes, activity, wider_id, wordnet_id, wordnet_face_id
            .meta   entry type: data_2d, position_2d, path, *[attr_id]*6, class_name, *[class_id]*3
            .dtype  entry dtype: self.dtype, self.dtype, str, int, str, uint8, int, int, int


        class_id includes both WIDER class_id, and Wordnet class_id
            eg, in 9--Parade, class_name = "Parade", wider_id = 9, wordnet_id = 8428485
        if annotation file not found, download and place in path
    
        # with open(osp.join(self.paths['annotations'], f"readme.txt"), 'r') as _fi:
        #     dset_tags = _fi.read().split("\n")[-1].split(", ")
        # dset_tags = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid',
        #         'occlusion', 'pose']
        # ['x1', 'y1', 'w', 'h'] -> bbox
        # ['blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose'] -> attributes

        """

        # read image list
        with open(image_list_file, 'r') as _fi:
            text = _fi.read().split("\n")

        images = []
        i = 0
        while i < len(text) - 1:
            line = text[i]
            if not line:
                i += 1
                continue

            # class from filename
            class_name = osp.basename(osp.dirname(line))
            wordnet_id = self._wnid_dict(class_name)   # wordnet_id e.g. 7144834
            class_name = class_name.split("--")
            assert (class_name[0].strip()).isnumeric(), f"fails <{class_name}> is notnumeric!, at line {i}"
            class_id = int(class_name[0])               # class_id # e.g. 9
            class_name = "--".join(class_name[1:])      # class_name # e.g. Press_Conference

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
                for j in range(num_faces):
                    face= np.fromstring(text[i], dtype="int", sep=" ")
                    bbox.append(face[:4])
                    attributes.append(face[4:])
                    i += 1

                bbox = np.stack(bbox, axis=0).reshape(-1,2,2)
                attributes = np.stack(attributes, axis=0)
                attrnames = ['blur','expression','illumination','invalid','occlusion','pose']
                attrs = {attrnames[i]:attributes[:,i] for i in range(len(attrnames))}

                images.append(DataItem([bbox, name, *attrs.values(), class_name, class_id, wordnet_id],
                                        tags=["bbox", "name", *["attr_id"]*len(attrnames), "class_name", "class_id", "wordnet_id"],
                                        meta=["positions_2d", "path", *attrs.keys(), "str", "int", "int"],
                                        dtype=["float32", "str", 'uint8', 'bool', 'bool', 'bool', 'uint8', 'bool', "str", "uint8", "int"]
                                        )
                            )
            else: # images without bboxes
                images.append(DataItem([name], tags=["name"], meta=["path"], dtype=["str"]))
        return images



    # def _read_annotations(self, image_list_file: list, folder: str, mode: str="train") -> list:
    #     """ returns list of DataItem
    #         Bounding Boxes are grouped per image for augmentation purposes [N, 2,2] in format x,y,w,h

    #     with keys for entries
    #         .tags   entry name: image, bbox, name, blur, expression, illumination, invalid, occlusion, pose, activity, wider_id, wordnet_id, wordnet_face_id
    #         .meta   entry type: data_2d, position_2d, path, *[attr_id]*6, class_name, *[class_id]*3
    #         .dtype  entry dtype: self.dtype, self.dtype, str, uint8, bool, bool, bool, uint8, bool, str, uint8, int, int, int

    #     class_id includes both WIDER class_id, and Wordnet class_id
    #         eg, in 9--Parade, class_name = "Parade", wider_id = 9, wordnet_id = 8428485
    #     if annotation file not found, download and place in path
    #     """
    #     # with open(osp.join(self.paths['annotations'], f"readme.txt"), 'r') as _fi:
    #     #     dset_tags = _fi.read().split("\n")[-1].split(", ")
    #     # dset_tags = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid',
    #     #         'occlusion', 'pose']

    #     # read image list
    #     with open(image_list_file, 'r') as _fi:
    #         text = _fi.read().split("\n")
    #         ## train and val images, format, e.g.
    #         # 9--Press_Conference/9_Press_Conference_Press_Conference_9_129.jpg
    #         # 2
    #         # 336 242 152 202 0 0 0 0 0 0
    #         # 712 278 126 152 0 0 0 0 0 0

    #     images = []
    #     _wn_face = self._wnid_dict('Face')

    #     lno = 0
    #     itno = 1
    #     print(f"Reading, {image_list_file} with {len(text)} entries" )

    #     while len(text):
    #         name = text.pop(0); lno +=1
    #         if not name:
    #             continue
    #         name = osp.join(folder, name)               # image path
    #         if not osp.isfile(name):
    #             print(f"image not found '{name}'', skipping ...")
    #             continue

    #         # class, activity category from folder
    #         class_name = osp.basename(osp.dirname(name))
    #         wordnet_id = self._wnid_dict(class_name)   # wordnet_id e.g. 7144834
    #         class_name = class_name.split("--")
    #         class_id = int(class_name[0])               # class_id # e.g. 9
    #         class_name = "--".join(class_name[1:])      # class_name # e.g. Press_Conference

    #         attributes = {t:[] for t in ['blur','expression','illumination','invalid','occlusion','pose']}
    
    #         # test files contain no bbox annotations, skip
    #         if mode not in "test" and text[0].strip().isnumeric():
    #             _num_faces = int(text.pop(0).strip()); lno +=1

    #             bboxes = []
    #             for _ in range(_num_faces):
    #                 this_box = np.fromstring(text.pop(0), dtype=self.dtype, sep=" "); lno +=1
    #                 bboxes.append(this_box[:4].reshape(2,2))
    #                 attrs = this_box[4:].astype(self.dtype)

    #                 for k, tag in enumerate(attributes):
    #                     attributes[tag].append(attrs[k])
    #             bboxes = np.stack(bboxes, axis=0)

    #             _items = [bboxes, name, *attributes.values(), class_name, class_id, wordnet_id[0], _wn_face]
    #             tags = ["bbox", "name", *attributes.keys(), "activity", "wider_id","wordnet_id", "wordnet_face_id"]
    #             meta = ["position_2d", "path", *["attr_id" for _a in range(len(attributes))], "class_name", "class_id", "class_id", "class_id"]
    #             dtype = [self.dtype, "str", 'uint8', 'bool', 'bool', 'bool', 'uint8', 'bool', "str", "uint8", "int", "int"]

    #             # prefilter on load.
    #             # alternatively everything could be loaded, and filtered on __getitem__() with DataItem.keep
    #             if self.keep is not None:
    #                 _keepidx = [i for i in range(len(tags[i])) if tags[i] in self.keep]
    #                 _items = [_items[i] for i in _keepidx]
    #                 tags = [tags[i] for i in _keepidx]
    #                 meta = [meta[i] for i in _keepidx]
    #                 dtype = [dtype[i] for i in _keepidx]

    #             itno += 1
    #             print(f"lines ->{lno}, Dataitem_{itno}")

    #             item = DataItem(_items, tags=tags, meta=meta, dtype=dtype)
    #             images.append(item)
    #         else:
    #             item = DataItem([name], tags=["name"], meta=["path"])
    #     return images


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
