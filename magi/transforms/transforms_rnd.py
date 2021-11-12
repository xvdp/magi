"""@xvdp """
from typing import Union, Optional
import inspect
from inspect import getfullargspec, signature
import numpy as np
import torch
import torch.distributions as tdist
from koreto import Col

from magi.utils.torch_util import broadcast_tensors, to_tensor, squeeze_trailing

from ..utils import torch_dtype, torch_device


# pylint: disable=no-member
_tensorish = (int, float, list, tuple, np.ndarray, torch.Tensor)
_Tensorish = Union[_tensorish]
_Dtype = Union[None, str, torch.dtype]
_Device = Union[str, torch.device]


class Distribution:
    """Base class for managed distributions

    """
    def __init__(self,
                 seed: Optional[int] = None,
                 dtype: _Dtype = None,
                 device: _Device = "cpu") -> None:

        if seed is not None:
            torch.manual_seed(seed)

        self.dtype = torch_dtype(dtype, force_default=True)
        self.device = torch_device(device)
        self.batch_mask = None
        self._to = {"dtype":self.dtype, "device":self.device}
        self._keys = []
        self.__ = None

    def to(self, dtype: _Dtype = None, device: Optional[_Device] = None) -> None:
        """ converts Distribution tensors and sampler output to device/dtype
        """
        _to = {}
        if dtype is not None and torch_dtype(dtype, force_default=True) != self.dtype:
            self.dtype = torch_dtype(dtype, force_default=True)
            _to['dtype'] = self.dtype
        if device is not None and torch_device(device) != self.device:
            self.device = torch_device(device)
            _to['device'] = self.device

        if _to:
            self._to = _to
            for key in self._keys:
                self.__dict__[key] = self.__dict__[key] .to(**_to)

            if self.__ is not None:
                for key, val in self.__.__dict__.items():
                    if torch.is_tensor(val):
                        self.__.__dict__[key] = val.to(**_to)


    def get_batch_mask(self, dims: tuple) -> None:
        """ self.batch_mask is a binary mask to expansion of sample to batch shape
        if distribution is singleton valued (eg, low=0.1, high=0.3) all dims are expanded:
        eg. dims=(0,1,3) will generate an batch_mask [1,0,1,1] and on data NCHW will return shape (N,C,1,W)

        if distribution args are tensors with a width, eg. shaped (1,34,34) corresponding to C,H,W
        batch_mask will be capped at before the width dimension
        and leading singletons will be squeezed-> values: (34,34), batch_mask [1,0] (if dims=(0,1,3),)
        and sampler will return (N,1,34,34)

        values entered to the distribution with shape (1,1,K) will return samples of shape N,C,H,K (if dims=(0,1,2) )
        or shape (1,1,1,K) if dims=None)
        """
        if isinstance(dims, int):
            dims = (dims,)

        # shape of values- either constant or from distribution
        if 'vals' in self.__dict__ and self.vals is not None:
            _batch_shape = self.vals[0].shape
        elif 'p' in self.__dict__ and self.__ is None:
            _batch_shape = self.p.shape
        else:
            _batch_shape = self.__._batch_shape

        # values are over CHW.. not N, batch is either expandable or not but it is there
        if dims is None:
            batch_mask = torch.zeros([1])
        else:
            batch_mask = torch.as_tensor([int(i in dims) for i  in range(max(dims)+1)])
        _squeeze_front = [i+1 for i in range(len(_batch_shape)) if _batch_shape[i] > 1]

        if _squeeze_front:
            batch_mask = batch_mask[:min(_squeeze_front)]
            # batch_mask[min(_squeeze_front):] = 0

            _squeeze_front = min(_squeeze_front)
            # If parameters need squeezing, reinit distribution
            if self.__ is not None:
                params = {key:val for key, val in self.__.__dict__.items()
                          if torch.is_tensor(val) and "_" not in key}

                for key, val in params.items():
                    if torch.is_tensor(val):
                        for i in range(_squeeze_front):
                            params[key] = val.squeeze(0)
                self.__ = self.__.__class__(**params)

        self.batch_mask = torch.as_tensor(batch_mask)


    def __repr__(self, exclude_keys: Union[list, tuple] = ()) -> str:
        """ utility, auto __repr__()
        Args
            exclude_keys    (list, tuple [None])
        """
        rep = self.__class__.__name__+"("
        exclude_keys = list(exclude_keys) + [key for key in self.__dict__
                                             if (key[0] == "_" and not key == "__")]
        for i, (key, value) in enumerate(self.__dict__.items()):
            if key not in exclude_keys:
                if isinstance(value, str):
                    value = f"'{value}'"
                elif isinstance(value, torch.Tensor):
                    value = f"tensor({value.tolist()})"

                sep = "" if not i else ", "
                rep += f"{sep}{key}={value}"
        return rep + ")"



class Values(Distribution):
    """ Initializes samplers for randomizing augmentation transforms based on torch.distributions.
    Values().sample(shape) -> torch tensor with broadcasting rules that depend on:
        1. a,b, input values, if any input is wider 1, iit is considered Channels, making smallest sample().ndim = 2
        2. expand_dims, dims not specified and dims trailing wider dims are width 1
        3. shape, sample().ndim == min(1, shape) unless (1.) applies

    Args:
    'a', 'b', **kwargs in ('a','b' ... 'z') not in (i j k p)
    **kwargs in keywords to requested distribution, e.g. 'df1', 'df2' for 'FisherSnedecor'...
        value arguments of type (int, float, tuple, list, ndarray, tensor [None])
        A minimum of 1 value argument needs to be passed
        Value arguments can be singleton, or have shape, Index 0 of arg corresponds to channels
            eg. an arg of shape (3,40,40) is interpreted as C=3, H=40, W=40
        Non signleton dims should match expected input
            e.g, arg shape (3,) should operate on tensors NC... where C==3

    'distribution'  (str [None])    valid torch distribution name in torch.distributions type <type>
        common used cases: 'Normal', 'Uniform'
        .sample(), .sample(shape=(...)) -> returns tensor of ndim = len(shape) | 1
        -> constant
            if distribution=None and one arg passed, e.g. a=5, sample() -> tensor([5.])
        -> discrete sampling from char_args, a,b,c,...z {not in i,j,k,p}
            if distribution=None | 'Categorical' | 'Binomial' and more than arg passed
        -> continuous sampling from char args or kwargs
            if distribution in 'Normal' | 'Uniform' | &c.

        Discrete sampling of more than one char args supports 2 distributions:
            'Categorical' (default if distribution=None) default 'probs'=[1]*len(char_args)
              # categorical =~ integer uniform
            'Binomial'    default 'probs'= 0.5, i.e if args in (a,b,c,d,e) c is most probable
              # binomial =~ integer normal centered in len(char_args)*probs
        Only Binomial and Categorical are leveraged to sample input character values,

        Continuous Distributions. Any exponential torch.distributions
        Managed Continuous distributions:
            If inputs are passed as char args 'a', 'b'...
            Uniform:    low, high are sorted
            Normal, Laplace, Gumbel:    when 'center' is True, 'a' and 'b' determine range at std=3

        ..warning. Not all distributions were tested.
        Only when passed by char arg, parameters are converted to tensor and broadcasted, managed

    'expand_dims'   (tuple, int [0])    dimensions expanded to requested shape, default batch dim N
        if value args are singletons any dimension is expanded to .sample(shape)
        e.g. a=0, b=1, expand_dims=(0,1,3) .sample(shape=(4,5,12,12)) -> tensor shape (4,5,1,12)
        return shape is 1 for all trailing dims of non sigletons and non expanded dims
        e.g. if a=[3,2,1] b=[4,5,6], expand_dims=(0,1,3) 
            .sample(shape=(4,5,12,12)) -> tensor shape (4,3,1,1)

    'clamp'     (tuple [None]) min max tuple to distribution values
    'center'    (bool [True]) centrs 'Normal', 'Laplace', 'Gumbel' if args passed as 'a', 'b'
    'seed'       (int [None]) manual_seed
    'dtype'           (torch.dtype [None]) None: default dtype
    'device'          (str, torch.device, ['cpu'])

    Examples:
    >>> import torch
    >>> from magi.transforms import Values

    # constant
    >>> v = Values(torch.tensor([[1,0.2,0.3]]))
    >>> v.sample()
    tensor(a=[[[1.0000, 0.2000, 0.3000]]])
    >>> v.sample().shape
    torch.Size([1, 1, 3]) # input a, shape (1,3)

    # Categorical
    v = Values(a=torch.tensor([1, 0.2, 0.3]), b=2, expand_dims=(0,1))
    >>> sample = v.sample(4)
    >>> sample
    tensor([[1.0000, 0.2000, 0.3000],
            [1.0000, 0.2000, 0.3000],
            [1.0000, 0.2000, 0.3000],
            [2.0000, 2.0000, 2.0000]])
    >>> sample.shape
    torch.Size([4, 3])

    >>> v.sample([4,12,23,32]).shape
    torch.Size([4, 3, 1, 1])

    # Catengorical many inputs
    >>> v = Values(a=0.1, b=20, c=300, d=4000, expand_dims=(0,1,3))
    >>> assert torch.any(v.sample(1000) == 0.1)
    >>> assert torch.any(v.sample(1000) == 20.)
    >>> assert torch.any(v.sample(1000) == 300.)
    >>> assert torch.any(v.sample(1000) == 4000.)

    >>> v.sample([4,3,100,100]).shape
    torch.Size([4, 3, 1, 100])

    # Uniform, args passed by keyword
    >>> v = Values(low=0.1, high=20, expand_dims=(0,1,3), distribution="Uniform")
    >>> v.sample((4,3,40,40)).shape
    torch.Size([4, 3, 1, 40])

    # Uniform (if passed as char args: sorted)
    >>> v = Values(a=0.1, b=[-20,20,1], expand_dims=(0,1,3), distribution="Uniform")
    >>> v.__.low
    tensor([-20.0000,   0.1000,   0.1000])
    >>> v.__.high
    tensor([ 0.1000, 20.0000,  1.0000])
    >>> v.sample((4,1,12,12)).shape
    torch.Size([4, 3, 1, 1])

    # Normal (args pasesed by arg, center=True) - args -> range
    >>> v = Values(a=0.1, b=[-20,20,1], expand_dims=(0,1,3), distribution="Normal", center=True, seed=1)
    >>> v.sample()
    tensor([[-7.7345, 10.9353,  0.5593]])

    >>> v.__.loc
    tensor([-9.9500, 10.0500,  0.5500])  # loc is center between a,b

    >>> v.to(device='cuda', dtype=torch.float16)    # convert sampling to device dtype
    >>> v.sample()
    tensor([[-10.9453,  18.9219,   0.5288]], device='cuda:0', dtype=torch.float16)

    Info on distributions: https://pytorch.org/docs/stable/distributions.html
    of call transforms_rnd.print_distributions()
    """
    def __init__(self, a: Union[None, int, float, torch.Tensor] = None,
                 b: Union[None, int, float, torch.Tensor] = None,
                 distribution: Optional[str] = None,
                 expand_dims: Union[None, tuple, int] = 0,
                 clamp: Union[None, list, tuple] = None,
                 center: bool = True,
                 seed: Optional[int] = None,
                 dtype: _Dtype = None,
                 device: _Device = "cpu",
                 **kwargs) -> None:
        super().__init__(seed=seed, dtype=dtype, device=device)

        _args = {}
        self.vals = None    # if constant, categorical or binomial
        self.batch_mask = None # one hot mask with dim expansions
        self.clamp = clamp  # if not None clamp output values
        assert clamp is None or (isinstance(clamp, (list, tuple)) and len(clamp)==2), f"{Col.YB}clamp: (min,max) tuple req found {clamp}{Col.AU}"
        self._keys = []     # used in repr and .to(device=, dtype=)

        # Single character args ['a', 'b' ... 'z'] not in (i j k p)
        # a and b are initialized as many distributions require 2 elements
        kch_args = {ch[0]: ch[1] for ch in (('a', a), ('b', b)) if ch[1] is not None}
        kch_args.update({ch: kwargs.pop(ch) for ch in 'cdefghlmnoqrstuvwxyz' if ch in kwargs})

        ch_args = []
        for _, val in kch_args.items():
            ch_args.append(squeeze_trailing(val, min_ndim=0))
        if ch_args:
            ch_args = list(broadcast_tensors(*ch_args, dtype=self.dtype))

        _total_args = len(ch_args) + self.has_distribution_kargs(**kwargs)
        assert _total_args > 0, f"{Col.YB}minimum a single argument required {Col.AU}"

        # 1.  sample() -> constant a
        _single_valued = ['Chi2', 'Exponential', 'HalfCauchy', 'HalfNormal', 'Poisson', 'Bernoulli']
        if  _total_args == 1 and distribution not in _single_valued:
            self.vals = ch_args[0]
            if len(self.vals) > 1:
                self.vals = self.vals.unsqueeze(0)

        # 2. sample() ->  categorical / binomial of constants (a, b, c), ...
        elif distribution in (None, "Categorical", "Binomial"): # Uniform discrete sampling
            if distribution == "Binomial":
                _args['total_count'] = len(ch_args)
                size = 1
            else:
                size = len(ch_args)
                distribution = "Categorical"
            _args['probs'] = self._get_probs((size,), distribution, kwargs)
            self.__ = torch.distributions.__dict__[distribution](**_args)
            self.vals = squeeze_trailing(torch.stack(ch_args), min_ndim=0)

        # 3. continuous distributions
        else:
            distribution = 'Uniform' if distribution is None else distribution
            assert self.is_distribution_valid(distribution), f"{Col.YB}'{distribution}' not recognized{Col.AU}"

            # only ascii key (a,..) args checked for broadcastability
            _args = self._distribution_kwargs(distribution, kwargs)
            if not ch_args: # validate all parameters
                assert not any(v is None for v in _args), f"{Col.YB}missing args: {_args}{Col.AU}"

            # centered distributions can be recentered to a,b range
            elif distribution in ['Normal', 'Laplace', 'Gumbel'] and center:
                a = None if len(ch_args) == 0 else ch_args[0]
                b = None if len(ch_args) < 2 else ch_args[1]
                self.loc_scale_from_range(a, b, _args, sigma=3)

            else:
                self._fill_missing_params(ch_args, _args)
                if distribution == "Uniform":
                    _args['low'], _args['high'] = self._sort_values(_args['low'], _args['high'])

            self.__ = torch.distributions.__dict__[distribution](**_args)

        if self.vals is not None:
            self._keys.append('vals')
        self.get_batch_mask(expand_dims)

    def sample(self, shape: Union[int, tuple] = (1,)) -> torch.Tensor:
        """ Return distribution sample or value a
            Args
                shape   (tuple)   sample will output tensor of same ndim
        """
        shape = (shape,) if isinstance(shape, int) else shape
        size = len(self.batch_mask[:len(shape)])
        # sample shape: mask of input shape size
        sample_shape = torch.maximum(torch.as_tensor(shape[:size]) * self.batch_mask[:size],
                                     torch.ones(1)).to(dtype=torch.int)

        if self.__ is None:
            out = self.vals
            # print("const shape, input shape -> sample_shape", out.shape, shape, sample_shape)
            out =  torch.broadcast_to(out, sample_shape.tolist())
            # out = torch.stack([self.a for _ in range(sample_shape[0])])# constant
        elif self.__.__class__.__name__ in ('Categorical', 'Binomial'):
            out = self.vals[self.__.sample(sample_shape)]
        else:
            out = self.__.sample(sample_shape)

        if len(out.shape) < len(shape):
            out = out.view(*out.shape, *[1] * (len(shape) - len(out.shape)))
        # print("out shape", out.shape)

        if self.clamp is not None:
            out = torch.clamp(out, *self.clamp)
        return out

    @staticmethod
    def _get_probs(size: tuple, msg: str, kwargs: dict) -> torch.Tensor:
        """Return torch tensor of probabilities, """
        for key in ['probs', 'logts']:
            if key in kwargs:
                out = torch.as_tensor(kwargs.pop(key), torch.get_default_dtype())
                assert out.shape == size, f"{Col.YB}Expected {key} sized {size} for {msg} distribution{Col.AU}"
                return out
        return torch.ones(size).div(2.)

    @staticmethod
    def _sort_values(a, b):
        if isinstance(a, (float, int)) and isinstance(b, (float, int)) and a > b:
            a, b =  sorted([a, b])

        elif torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.any(a > b):
            _ab, _ = torch.stack([a, b]).sort(axis=0)
            a = _ab[0]
            b = _ab[1]
            if any(b==a):
                b[b==a] = b[b==a] + 0.001
        return squeeze_trailing(a, min_ndim=0), squeeze_trailing(b, min_ndim=0)

    @staticmethod
    def loc_scale_from_range(a, b, _args, sigma=3):
        """ Normal distribution from range
        """
        _args['loc'] = (a + b) / 2 if _args['loc'] is None else _args['loc']
        _args['scale'] = (_args['loc'] - a )/sigma if _args['scale'] is None else _args['scale']
        _args['scale'] = to_tensor(_args['scale']).abs()

        for arg in ['loc', 'scale']:
            _args[arg] = squeeze_trailing(_args[arg], min_ndim=0)

    @staticmethod
    def _distribution_kwargs(dist: str, kwargs: dict) -> dict:
        """ Return params passed by name and required params: None
        """
        distros = [key for key,val in tdist.__dict__.items() if isinstance(val, type)
                   and 'Transform' not in key]
        assert dist in distros, f"{Col.YB}distribution '{dist}' not in {distros} {Col.AU}"

        out = {}
        dist_params = signature(tdist.__dict__[dist]).parameters
        # either probs or logits required but both are marked optional in torch.distribution
        _probs = ('probs' in dist_params and ('probs' not in kwargs and 'logits' not in kwargs))

        for key, value in dist_params.items():
            if key in kwargs:
                out[key] = kwargs.pop(key)
                if isinstance(out[key], _tensorish):
                    out[key] = squeeze_trailing(out[key], min_ndim=0)
            elif value.default is inspect._empty or (_probs and key == 'probs'):
                out[key] = None
        return out

    @staticmethod
    def _fill_missing_params(ch_args: list, _args: dict) -> None:
        """Merge params passed by distribution key and character params"""
        for key, value in _args.items():
            if value is None:
                _args[key] = ch_args.pop(0)

    @staticmethod
    def is_distribution_valid(distribution: str) -> bool:
        """ True if distribution in torch.distributions"""
        _valid =  [key for key,val in tdist.__dict__.items() if isinstance(val, type)
                   and 'Transform' not in key]

        return distribution in _valid

    @staticmethod
    def has_distribution_kargs(**kwargs) -> int:
        """ Count of arguments belonging to any distribution
        # could fail if one passes df1, df2 but a Uniform distro
        """
        _D = [key for key,val in tdist.__dict__.items() if isinstance(val, type)
              and 'Transform' not in key]
        dist_args = []
        for _d in _D:
            dist_args += [a for a in getfullargspec(tdist.__dict__[_d]).args
                          if a not in ('self', 'validate_args')]
        return sum([1 for k in kwargs if k in dist_args])


class Probs(Values):
    """ thin wrapper over Values for Bernoulli probablility
    Args
        p           (float, tensor [1] Bernoulli probability
        expand dims (tuple)     None: single value per batch
                                0:  value per sample
                                1:  value per channel
                                ....
        seed        (int)       manual random reseed
    """
    def __init__(self,
                 p: float = 1.0,
                 expand_dims: Union[None, tuple, int] = 0,
                 seed: Optional[int] = None,
                 dtype: _Dtype = None,
                 device: _Device = "cpu") -> None:

        args = {'distribution':'Bernoulli', 'probs':p, 'expand_dims':expand_dims}
        if torch.all(torch.as_tensor(p) == 1):
            args = {'distribution':None, 'a':1, 'expand_dims':None}

        super().__init__(**args, seed=seed, dtype=dtype, device=device)


def print_distributions():
    """ print torch distributions
    """
    _D = [d for d in tdist.__dict__ if isinstance(tdist.__dict__[d], type) and 'Transform' not in d]
    for _d in _D:
        print(f"{_d:35}{[a for a in getfullargspec(tdist.__dict__[_d]).args if a not in 'self']}")




"""

# use categorical to blend in a, b, c

>>> M = tdist.__dict__['Categorical'](probs=torch.tensor([1, 1, 1]))
>>> M.sample([10])
tensor([2, 2, 1, 0, 0, 0, 2, 1, 0, 0])
>>> M = tdist.__dict__['Categorical'](probs=torch.tensor([0.1, 1, 2]))
>>> M.sample([10])
tensor([2, 2, 2, 2, 2, 1, 2, 2, 2, 2]

>>> M = tdist.__dict__['OneHotCategorical'](probs=torch.tensor([0.1, 1, 2]))
>>> M.sample([5])
tensor([[0., 0., 1.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.]])

>>> M = tdist.__dict__['ContinuousBernoulli'](probs=torch.tensor([0.1, 1, 2]))
>>> M.sample([5])
tensor([[0.1478, 0.9985, 0.9818],
        [0.2182, 0.9426, 0.9980],
        [0.5013, 0.9896, 0.8375],
        [0.4649, 0.9586, 0.9085],
        [0.0334, 0.9941, 0.8922]])
>>> M = tdist.__dict__['Bernoulli'](probs=0.5)
>>> M.sample([10])
tensor([1., 1., 0., 0., 1., 0., 1., 1., 1., 1.])

# Binomial, 
>>> M = tdist.__dict__['Binomial'](total_count=3, probs=0)
>>> M.sample([10])
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>>> M = tdist.__dict__['Binomial'](total_count=3, probs=0.5)
>>> M.sample([10])
tensor([2., 1., 1., 1., 3., 2., 1., 1., 2., 2.])


# Multinomial
>>> M = tdist.__dict__['Multinomial'](total_count=3, probs=torch.Tensor([1]))
>>> M.sample([5])
tensor([[3.],
        [3.],
        [3.],
        [3.],
        [3.]])
>>> M = tdist.__dict__['Multinomial'](total_count=3, probs=torch.Tensor([1,1]))
>>> M.sample([5])
tensor([[2., 1.],
        [1., 2.],
        [1., 2.],
        [2., 1.],
        [0., 3.]])
>>> M = tdist.__dict__['Multinomial'](total_count=3, probs=torch.Tensor([1,1,0.1]))
>>> M.sample([5])
tensor([[1., 2., 0.],
        [3., 0., 0.],
        [2., 0., 1.],
        [2., 1., 0.],
        [1., 2., 0.]])


Bernoulli                          ['probs', 'logits']
Beta                               ['concentration1', 'concentration0']
Binomial                           ['total_count', 'probs', 'logits']
Categorical                        ['probs', 'logits']
Cauchy                             ['loc', 'scale']
Chi2                               ['df']
ContinuousBernoulli                ['probs', 'logits', 'lims']
Dirichlet                          ['concentration']
Distribution                       ['batch_shape', 'event_shape']
ExponentialFamily                  ['batch_shape', 'event_shape']
Exponential                        ['rate']
FisherSnedecor                     ['df1', 'df2']
Gamma                              ['concentration', 'rate']
Geometric                          ['probs', 'logits']
Gumbel                             ['loc', 'scale']
HalfCauchy                         ['scale']
HalfNormal                         ['scale']
Independent                        ['base_distribution', 'reinterpreted_batch_ndims']
Kumaraswamy                        ['concentration1', 'concentration0']
Laplace                            ['loc', 'scale']
LKJCholesky                        ['dim', 'concentration']
LogNormal                          ['loc', 'scale']
LogisticNormal                     ['loc', 'scale']
LowRankMultivariateNormal          ['loc', 'cov_factor', 'cov_diag']
MixtureSameFamily                  ['mixture_distribution', 'component_distribution']
Multinomial                        ['total_count', 'probs', 'logits']
MultivariateNormal                 ['loc', 'covariance_matrix', 'precision_matrix', 'scale_tril']
NegativeBinomial                   ['total_count', 'probs', 'logits']
Normal                             ['loc', 'scale']
OneHotCategorical                  ['probs', 'logits']
OneHotCategoricalStraightThrough   ['probs', 'logits']
Pareto                             ['scale', 'alpha']
Poisson                            ['rate']
RelaxedBernoulli                   ['temperature', 'probs', 'logits']
RelaxedOneHotCategorical           ['temperature', 'probs', 'logits']
StudentT                           ['df', 'loc', 'scale']
Uniform                            ['low', 'high']
VonMises                           ['loc', 'concentration']
Weibull                            ['scale', 'concentration']
"""