"""@ xvdp

Agumentation library with syntax derived to torchvision.transforms (TT).
Input to transformations are batches has to contain at least one NCHW torch.Tensor.
    * NCHW Image_Tensor

It can also contain a full batch as output my a Dataloader, containing Image and Index tensors
    * [NCHW Image_Tensor, N Index_Tensor]

Or batches with position data associated to those tensors, g.g. bounding boxes or segmentation paths
    * [NCHW Image_Tensor, N[M,2,2] Annotation_Tensor_List,  N Index_Tensor]

Annotation Tensor Lists will be transformed by Affine
"""
from typing import Union
import math
import numpy as np
import torch
from ..utils import torch_dtype

from .. import config

_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]

###
# Base classes of Transforms contain class attribute '__type__'to
# Transform             (object)    + update_kwargs, auto __repr__
# TransformAppearance   (Transform)
#
class Transform(object):
    """ base transform class
    """
    __type__ = "Transform"

    def __init__(self, for_display: bool=None) -> None:
        """ for_display: True will clone data on this transform
            to set globally config.set_for_display(bool) or set in datasets
        """
        self.for_display = for_display

    def __repr__(self, exclude_keys: Union[list, tuple]=None) -> str:
        """ utility, auto __repr__()
        Args
            exclude_keys    (list, tuple [None])
        """
        rep = self.__class__.__name__+"("
        for i, (key, value) in enumerate(self.__dict__.items()):
            if exclude_keys is not None and key in exclude_keys:
                continue
            if isinstance(value, str):
                value = f"'{value}'"
            elif isinstance(value, torch.Tensor):
                value = f"tensor({value.tolist()})"

            sep = "" if not i else ", "
            rep += f"{sep}{key}={value}"
        return rep + ")"

    def update_kwargs(self, exclude_keys: Union[list, tuple]=None, **kwargs) -> tuple:
        """ utility
            __call__(**kwargs) changes transform functional
                without changing instance attributes
            Args:
                exclude_keys    (list), read only keys
        """
        exclude_keys = [] if exclude_keys is None else exclude_keys
        out = {k:v for k,v in self.__dict__.items() if k[0] != "_" and k not in exclude_keys}

        unused = {}
        for k in kwargs:
            if k in out:
                out[k] = kwargs[k]
            else:
                unused[k] = kwargs[k]

        if 'for_display' in  out and out['for_display']  is None:
            out['for_display'] = config.FOR_DISPLAY

        return out, unused

class TransformApp(Transform):
    """ base transform class
    """
    __type__ = "Appearance"


# pylint: disable=no-member
class Randomize:
    """ distribuition is initalized to tensor shape without batch size
    it will then be sampled by batch size [N], or [1]

    TODO: Binomial? a or b instead of a*s +b(1-s)
    TODO Multinomial
    TODO re-init with input data: e.g. saturation, 
        Resplace Values vs offset values?

    Args
        a, b            (float, tensor):  range ends
            if normal distribution, loc = (a+b)/2, a, b -> sigma3
            if a|b are tensors len(a|b) can equal 1 or num channels
        p               (float) > 0 <=1     Bernoulli p of randomization
        shape           (list, tuple) shape of expected tensor
            eg for NCHW: (1, 3, 200, 200); only channels and ndims are used
        distribution    (torch.distribution [Uniform]) | None | Normal | ... see beloa
        per_channel     (bool False]) if True, every channel gets different p and value
        dtype           (torch.dtype [None]) None: default dtype
        device          (str, cpu)  # TODO fix to accept torch.dtype
    Shape: expented tnsor shape (1CHW) or NCHW -

    distribution: from https://pytorch.org/docs/stable/distributions.html
        None -> a with a probability p

        .Uniform(low=a, high=b) default

        # (loc, scale) valued
            loc = (a+b)/2
            scale = (|(a+b)/2 -a|/3)) # 3std (normal ~99.9%, laplace ~99.3%)
        .Normal(loc=, scale=)
        .LogNormal(loc=, scale=)
        .Laplace(loc=, scale=)
        .Gumbel(loc=, scale=)
        .Cauchy(loc=, scale=)

        # single valued
        .Poisson(rate=a)
        .Exponential(rate=a)
        .Dirichlet(concentration=a)
        .Chi2(df=a)

        # other
        .Gamma(concentration=a, rate=b)
        .Kumaraswamy(concentration1=a, concentration2=b)
        .FisherSnedecor(df1=a, df=b)
        .Beat(concentration1=a, concentration0=b)
        .VonMises(loc=a, concentration=b)
        .Weibull(scale=a, concentration=b)
        .kl_divergence(p=a, q=b)

        # could support, modifying how 'b' is resized
        .MultivariateNormal(loc=(a+b)/2, cov=NotImplemented)
    """
    def __init__(self, a: Union[int, float, torch.Tensor], b: Union[int, float, torch.Tensor]=None,
                 p: float=1., shape: Union[list, tuple]=(1,3), distribution: str="Uniform",
                 per_channel: bool=False, per_sample: bool=False,
                 dtype: torch.dtype=None, device: str="cpu", clamp: Union[list, tuple]=None):

        self.a = a
        self.b = b
        self.p = p
        self.clamp = clamp

        self.batch_size = 1 if not per_sample else shape[0]
        self.shape = None
        self.dtype = torch_dtype(dtype, force_default=True)
        self.device = torch_device(device)

        _channels = 1 + (shape[1] - 1) * per_channel
        self.shape = [_channels, *[1]*(len(shape)-2)] # strip N, reduce HW... -> 1
        _to = {"dtype":self.dtype, "device":device}
        assert self.clamp is None or (isinstance(self.clamp, (list, tuple)) and len(self.clamp) == 2)

        # Bernoulli settings
        assert p > 0 and p <= 1, f"Bernoulli prob > 0 and <= 1 required, got {p}"
        self._Bernoulli = None
        self.p = torch.tensor([p]*_channels, **_to).view(*self.shape)
        if p < 1:
            self._Bernoulli = torch.distributions.Bernoulli(probs=self.p)

        # <Distribution> Settings
        self._Dist = None
        self.a = self._to_sized_tensor(self.a, self.shape, **_to)

        if distribution is not None:
            if distribution in ['Poisson', 'Exponential', 'Dirichlet', 'Chi2']:
                self._Dist = torch.distributions.__dict__[distribution](self.a)
            else:
                # TODO, implmenent new distribution centered on sample
                assert self.b is not None, f"expected arg 'b' for {distribution}(a=, b=)"
                self.b = self._to_sized_tensor(self.b, self.shape, **_to)

                if distribution in ['Normal', 'LogNormal', 'Laplace', 'Gumbel', 'Cauchy']:
                    self.a, self.b = self.loc_scale_from_range(self.a, self.b, sigma=3)
                self._Dist = torch.distributions.__dict__[distribution](self.a, self.b)

    @staticmethod
    def _to_sized_tensor(x, shape, dtype, device):
        """
        Args
            x       int, float, tuple, list, torch.Tensor
            shape   tuple expected shape excluding batch
        """
        # Distribution settings
        _isnum = lambda x: isinstance(x, (int, float))
        _istensorsingle = lambda x: isinstance(x, torch.Tensor) and (x.ndim == 0 or (x.ndim == 1 and len(x) == 1))
        _islistsingle = lambda x: isinstance(x, (list, tuple)) and len(x) == 1
        _isone = lambda x: _isnum(x) or _istensorsingle(x) or _islistsingle(x)

        if _isone(x):
            x = [x]* shape[0]
        return torch.as_tensor(x, dtype=dtype, device=device).view(*shape)

    @staticmethod
    def loc_scale_from_range(a, b, sigma=3):
        """
        """
        loc = (a + b) / 2
        scale = (loc - a )/sigma
        if isinstance(scale, float):
            scale = math.fabs(scale)
        elif isinstance(scale, torch.Tensor):
            scale.abs_()
        return loc, scale

    def sample_p(self, sample_shape: Union[list, tuple]=(1,)):
        """ return bernouli sample or 1"""
        if self._Bernoulli is None:
            return self.p
        return self._Bernoulli.sample(sample_shape)

    def sample(self, batch_size: int=None):
        """ return distribution sample or value a"""
        sample_shape = [batch_size if batch_size is not None else self.batch_size]
        if self._Dist is None:
            out = torch.stack([self.a for _ in range(sample_shape[0])])# no distribution sampling
        else:
            out = self._Dist.sample(sample_shape)
        if self.clamp:
            out = torch.clamp(out, *self.clamp)
        return out, self.sample_p(sample_shape)

    def to(self, dtype=None, device=None):
        _to = {}
        if dtype is not None and torch_dtype(dtype, force_default=True) != self.dtype:
            self.dtype = torch_dtype(dtype, force_default=True)
            _to['dtype'] = self.dtype
        if device is not None and torch_device(device) != self.device:
            self.device = torch_device(device)
            _to['device'] = self.device
        if _to:
            self.a = self.a.to(**_to)
            if self.b is not None:
                self.b = self.b.to(**_to)
            self.p = self.p.to(**_to)

            if self._Dist is not None:
                for key in self._Dist.__dict__:
                    if torch.is_tensor(self._Dist.__dict__[key]):
                        self._Dist.__dict__[key] = self._Dist.__dict__[key].to(**_to)
            if self._Bernoulli is not None:
                self._Bernoulli.probs = self._Bernoulli.probs.to(**_to)


def torch_device(device):
    """TODO clean up device and dtype handling
    this is going to bomb somewhere
    """
    if isinstance(device, str):
        if not ':' in device:
            device += ':0'
        device = torch.device(device)
    return device
