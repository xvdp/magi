"""@xvdp

include vidi in project
get logger
"""
import os
import os.path as osp
from PIL import Image
import vidi
from torch.utils.data import Dataset
from x_log import logger

# ###
# # Datasets
#
def sparse_image_list(folder, frame_range=None, extensions=(".png", ".jpg", ".jpeg")):
    """
    Args
    """
    images = sorted([f.path for f in os.scandir(folder)
                     if f.name[-4:].lower() in extensions])
    if frame_range is None:
        return images

    subset = []
    if isinstance(frame_range, int):
        frame_range = range(frame_range)
    elif (isinstance(frame_range, (list, tuple)) and len(frame_range) < 4):
        frame_range = range(*frame_range)

    for i in frame_range:
        if i < len(images):
            subset.append(images[i])

    return subset


class VideoDataset(Dataset):
    """ VideoDataset with variations on loading grid.

    Example:
        config = 'x_periment_scripts/eclipse_512_sub.yml'
        opt = x_utils.read_config(config)
        opt.sample_size = x_utils.estimate_samples(x_utils.GPUse().available, grad=1, **opt.siren)*2
        opt.shuffle=False
        kw = {k:opt[k] for k  in opt if k in list(inspect.signature(VideoDataset.__init__).parameters)[1:]}
    """
    def __init__(self, data_path, frame_range=None,
                 sample_fraction=1., sample_size=None,
                 strategy=-1, loglevel=20, device="cpu"):
        """
        Args
            sample_fraction     float(1.)   sample_size = data size * sample_fraction
            sample_size         int [None]  overrides sample fraction number of samples load

            # loading strategy in original dataset works best: single data array per epoch.
            strategy     int [-1]    -1: fully random samples, single iter per epoch
                                    0: fully random samples, all samples for the data, per per epoch
                                    1: shuffled random samples, all a samples for tehd ata
                                    2: complete sparsest sets, and dense set blocks, all samples 2x per epoch
                                    # 2 is no good, random wwrks better, 0,1,-1
        """
        super().__init__()
        log = logger("VideoDataset", level=loglevel)
        self.data = None
        self.sidelen = None
        self.mgrid = None

        self.sample_size = sample_size # naming change N_samples, overrides sample_fraction
        self.sample_fraction = sample_fraction
        self.sampler = None         # defined in strategy: 1
        self.grid_indices = None    # defined in strategy: 2
        self.grid_offset = None     # defined in strategy: 2
        self.strides = None         # defined in strategy: 2
        self.strategy = strategy

        as_centered_tensor = lambda x, device="cpu": torch.as_tensor((np.asarray(x, dtype=np.float32) - 127.5)/127.5, device=device)

        if 'npy' in data_path:
            self.data = torch.as_tensor(np.load(data_path), device=device)
        elif 'mp4' in data_path:
            # self.data = as_centered_tensor(skvideo.io.vread(data_path), device=device)
            self.data = as_centered_tensor(vidi.ffread(data_path), device=device)

        elif osp.isdir(data_path):
            images = sparse_image_list(data_path, frame_range)
            self.data = torch.stack([as_centered_tensor(Image.open(image), device=device) for image in images], dim=0)

        else:
            raise NotImplementedError("mp4, mov not yet re implemented")

        self.sidelen = torch.as_tensor(self.data.shape[:-1])
        self.channels = self.data.shape[-1]
        self.data = self.data.view(-1, self.data.shape[-1])
        log[0].info(f"Loaded data, sidelen: {self.sidelen.tolist()}, channels {self.channels}")
        log[0].info(f"         => reshaped to: {tuple(self.data.shape)}")

        if sample_size is not None:
            self.sample_fraction = min(1, (sample_size / self.sidelen.prod()).item())
        else:
            self.sample_size = max(1, int(self.sample_fraction * self.sidelen.prod()))

        log[0].info(" max sample_size, {}, fraction, {}".format(self.sample_size,
                    round(self.sample_fraction, 4)))

        # load entire data per epoch if it fits
        if self.sample_fraction >= 1:
            self.sample_fraction = 1
            self.mgrid = get_mgrid(self.sidelen)
            self.strategy = 0
            log[0].info(f" strategy: {self.strategy}, load grid {tuple(self.mgrid.shape)} in one step")

        # strategies 2+ sample ordered grids
        if self.strategy == 2:
            stride_pyramid = get_stride_tree(self.sidelen, self.sample_size)
            self.strides = torch.tensor([stride_pyramid[0] for _ in range(len(self.sidelen))])
            self.grid_indices, self.grid_offset =  get_inflatable_grid(self.sidelen, self.strides)
            self.sample_size = self.grid_indices.shape[0]
            log[0].info(f" strategy: {self.strategy}, iters: {len(self.grid_offset) * 2}")
            log[0].info(f"    strides: {self.strides.tolist()}, max mgrid block: {self.grid_indices[-1].tolist()}")
            log[0].info(f"    -> sample size: {self.sample_size}")


        elif self.strategy == 1:
            self.sampler = torch.randperm(self.sidelen.prod())
            log[0].info(f" strategy: {self.strategy}, randperm iters: {int(1/self.sample_fraction)}")

        # original, 1 iter per epoch
        elif self.strategy == -1:
            log[0].info(f" strategy: {self.strategy}, single sample per epoch")


    def shuffle(self):
        """ strategy 1 needs to shuffle indices"""
        if self.sampler is not None:
            self.sampler = torch.randperm(self.sidelen.prod())

    def __len__(self):
        if self.strategy == -1:
            return 1 # original strategy outputs single sample per
        elif self.strategy == 2:
            return len(self.grid_offset) * 2
        return int(1/self.sample_fraction)

    def _item(self, idx):
        with torch.no_grad():
            if self.sample_fraction < 1.:
                # non repeating sample grid/ sparsest and densest blocks
                if self.strategy == 2:
                    if idx < len(self.grid_offset): # sparsest
                        coords = self.grid_indices * self.strides + self.grid_offset[idx]
                    else: # inflate contiguous minigrids to offset idx
                        coords = self.grid_offset.max(dim=0).values * self.grid_offset[idx%len(self.grid_offset)] + self.grid_indices
                    data = self.data[flatten_igrid(coords, self.sidelen)].view(-1, self.channels)
                    coords = 2*coords/(self.sidelen-1) - 1
                else:
                    # possibly repeating
                    if self.strategy < 1:
                        coords = torch.randint(0, self.data.shape[0], (self.sample_size,))
                    # non repeating shuffled samples
                    elif self.strategy == 1:
                        coords = self.sampler[idx: idx+self.sample_size]

                    data = self.data[coords, :]
                    coords = 2*expand_igrid(coords, self.sidelen)/(self.sidelen-1) - 1
            else:
                coords = self.mgrid
                data = self.data
        return coords, data

    def __getitem__(self, idx):
        """
        """
        coords, data = self._item(idx)
        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class VideoDataset2(VideoDataset):
    """ VideoDataset returning (input,target) pairs
    """
    def __init__(self, data_path, frame_range=None,
                 sample_fraction=1., sample_size=None,
                 strategy=-1, loglevel=20, device="cpu"):
        super().__init__(data_path, frame_range, sample_fraction,
                         sample_size, strategy, loglevel, device)

    def __getitem__(self, idx):
        """
        """
        return self._item(idx)


