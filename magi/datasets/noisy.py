"""@xvdp """
import inspect
from typing import Union
import torch
from torch.utils.data.dataset import Dataset
from ..features import Item
from ..utils import warn_grad_cloning
from ..config import resolve_dtype, get_valid_device
# pylint: disable=no-member
# pylint: disable=not-callable

class Noise(Dataset):
    """
    Noise Generator
        normal, uniform, deterministic noise
        TODO multi scale noise, brown,
    """
    def __init__(self, noise_type: Union[str, list, tuple]="normal", channels: int=3,
                 size: Union[list, tuple]=(256,256), classes: int=1000, dataset_size: int=128000,
                 dtype: Union[str, torch.dtype]=None, device: Union[str, torch.device]="cpu",
                 seed: int=0, for_display: bool=False, grad: bool=False):
        """
        Args
            noise_type  (str[normal]) uniform, brownian
                # TODO add perlin, multi scale normal, brown, poisson.
            channels
            num_classes (int, [1000]) if noise_type is an array, ignore number of classes.
            size        (int, [1280000]) number of iterations before re-seedes, if None, no seeding
            device
            dtype

            for_display: bool=False
        """
        _valid_noise = ("normal", "uniform")
        if isinstance (noise_type, (list, tuple)):
            assert all([n in _valid_noise for n in noise_type])
            if len(noise_type) == 1:
                noise_type = noise_type[0]
            else:
                classes = len(noise_type)
                self.classes = noise_type

        if isinstance (noise_type, str):
            assert noise_type in _valid_noise
            self.classes = list(range(classes))

        self.noise_type = noise_type

        assert classes > 1, "cannot have single class"
        assert isinstance(size, (list, tuple)) and len(size) in (1,2,3), f"1,2,3 dims expected, found {len(size)}"

        self.name = f"{self.__class__.__name__}{str(noise_type).capitalize()}{classes}"
        self.dtype = resolve_dtype(dtype)
        self.device = get_valid_device(device)
        self.size = [1, channels, *size]
        self.seed = seed

        self.grad = grad
        warn_grad_cloning(for_display, grad, in_config=True)

        self.__len__ = dataset_size
        self._dims = len(size)
        self._counter = 0
        self._initialize()

    def _initialize(self):
        self._counter = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)

    def _gen_noise(self, index: str) -> torch.Tensor:
        """ Returns noise tensor
        """
        _noise = self.noise_type if not isinstance(self.noise_type, (tuple, list)) else self.noise_type[index]
        if _noise == "normal":
            return torch.randn(self.size, device=self.device, dtype=torch.__dict__[self.dtype])
        if _noise == "uniform":
            return torch.rand(self.size, device=self.device, dtype=torch.__dict__[self.dtype]) - 0.5

    def __getitem__(self, index=None):
        """
        Returns Item([sample, label])
        """
        if self._counter >= self.__len__:
            print("Re seeding Noise dataset")
            self._initialize()

        label = torch.randint(len(self.classes), (1,)).item()
        tensor = self._gen_noise(label)
        if self.grad:
            tensor.requires_grad = self.grad
        self._counter += 1

        return Item([tensor, label], names=['image', 'target_index'],
                     kind=[f'data_{self._dims}d', 'id'], dtype=[self.dtype, 'int'])

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
