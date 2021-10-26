"""@xvdp

Random handler
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


# , layout=torch.strided, device=None, requires_grad=False
"""
import math
import collections
from numbers import Number
import random
import torch

from .. import config
Iterable = collections.abc.Iterable

# pylint: disable=no-member
# pylint: disable=not-callable

def _check_grad(grad):
    if grad:
        print("DOES THIS REQUIRE GRAD?!, check")

def shuffle(ls):
    random.shuffle(ls)


def _validate_dtype(dtype):
    if dtype is None:
        dtype = torch.__dict__[config.check_dtype(dtype)]
    elif isinstance(dtype, str):
        assert dtype in torch.__dict__, "'%s' is not a valid dtype"%dtype
        dtype = torch.__dict__[dtype]
    else:
        assert isinstance(dtype, torch.dtype)
    return dtype

def bernoulli(p, samples, dtype=torch.int, device="cpu", grad=False):
    _check_grad(grad)
    dtype = _validate_dtype(dtype)
    if isinstance(p, tuple):
        p = torch.tensor(p, dtype=torch.float, device=device, requires_grad=False)
    return torch.zeros(samples, dtype=dtype, device=device, requires_grad=False).bernoulli_(p=p)
    """
    TODO #FIX
    ---> 35
    return torch.zeros(samples, dtype=dtype, device=device, requires_grad=grad).bernoulli_(p=p)
    RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    """

def normal(mean, std, samples, dtype=torch.float, device="cpu", grad=False):
    _check_grad(grad)
    dtype = _validate_dtype(dtype)
    return torch.zeros(samples, dtype=dtype, device=device, requires_grad=False).normal_(mean, math.fabs(std))

def uniform(a, b, samples, dtype=torch.float, device="cpu", grad=False):
    _check_grad(grad)
    dtype = _validate_dtype(dtype)
    return torch.zeros(samples, dtype=dtype, device=device, requires_grad=False).uniform_(a, b)

def mutlinoulli(low, high, samples, dtype=torch.float, device="cpu", grad=False):
    _check_grad(grad)
    dtype = _validate_dtype(dtype)
    if isinstance(samples, int):
        samples = (samples,)
    assert isinstance(samples, tuple), "samples must be tuple"
    return torch.randint(low=low, high=high, size=samples, dtype=dtype, device=device,
                         requires_grad=False)

def bound_normal(mean, a=0, b=1, stds=3, samples=1, dtype=None, device=None, grad=False):
    """ Returns samples from a bound normal distribution,
    skewed distribution with mode/mean defined by x, and bounds by a and b
    Args
        mean    (number or tensor)   tensor of sampled means, b > mean > a
        a       (float, 0) min bound
        b       (float, 1) max bound
        stds    (float, 3) num of standard deviations
        samples    (int)      number of elements of distribution
        dtype   (str, torch.dtype)
        device  (str)
        grad    (bool)

    Examples
    """
    _check_grad(grad)
    dtype = _validate_dtype(dtype)
    if isinstance(mean, (Number, list, tuple)):
        _device = config.check_device(device)
        mean = torch.tensor(mean, dtype=dtype, device=_device, requires_grad=False)

    assert isinstance(mean, torch.Tensor), "tensor required"
    _msg = "a and b must be numbers, not %s, %s"%(str(type(a)), str(type(b)))
    assert isinstance(a, Number) and (b, Number), _msg
    assert isinstance(samples, int), "samples must be an int > 0, not %s"%str(type(samples))
    assert samples > 0, "samples must be an int > 0, not %d"%samples
    _msg = "illegal value, mean:%.2f must be between a:%.2f and b:%.2f"%(mean.item(), a, b)
    assert (mean >= a).all() and (mean <= b).all(), _msg
    return bound_normal_proc(mean=mean, a=a, b=b, stds=stds, samples=samples)

def bound_normal_proc(mean, a=0, b=1, stds=3, samples=1000000):
    """
    Args
        mean    (tensor)   tensor of sampled means, b > mean > a
        a       (float, 0) min
        b       (float, 1) max
        stds    (float, 3) num of standard deviations
        samples (int)      number of elements of distribution
    """
    mean.sub_(a).div_(b-a)
    mean = torch.clamp(mean, 0.01, 0.99).unsqueeze_(-1)

    # mean > 0.5 = 1- mean
    _y = -torch.log2(0.5 - torch.abs(0.5 - mean))
    _w = torch.zeros(samples, dtype=mean.dtype, device=mean.device).normal_(mean=1/2, std=math.fabs(1/2/stds))
    _w = _w.add(torch.zeros_like(_y)).pow_(_y)
    _w[_w != _w] = torch.zeros(1, dtype=_w.dtype, device=_w.device, requires_grad=False)

    mean = torch.clamp(torch.round(mean)*(1 - 2*_w) + _w, 0, 1).add_(a).mul_(b-a)
    mean[mean != mean] = torch.zeros(1, dtype=_w.dtype, device=_w.device, requires_grad=False)
    return mean

# def laplace(loc, scale, size, dtype, device, grad):

#     Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
#     m.sample()

# WIP WIP TODO
def normal_positions(means, stds, sizes, dtype, device, grad=False):
    """ generate multiple distributions and concatenate to return
    N, (y,x),  conditioned on (h,w)  and (stdh, stdw)

    N, (j,i), conditioned on (sz_j, sz_i) and (std_j, std_i)

    """
    normals = []
    if not isinstance(stds, Iterable):
        stds = [stds for stds in range(len(means))]

    # for mean in means:
    #     normal(mean, std, size, dtype, device, grad), shift{shift} _shift{_shift}

def totuple2(val, mode):
    """ to tuple 2d, converts number or tuple to tuple of mode
        val     (number, tuple, list)
        mode    (str) "normal", "uniform"
    """
    assert isinstance(val, (Number, tuple, list)), "only number tuple or list allowed"
    _isnum = isinstance(val, Number)
    _istup = isinstance(val, (tuple, list))
    if _istup:
        assert len(val) == 2, "two item tuples"

    if mode == "normal":
        if _isnum:
            return (0.0, val)
        mean = (val[0], val[1])/2.0
        angle = math.fabs(val[1] - mean)
        return mean, angle

    if mode == "uniform":
        if _isnum:
            return -val, val
        return val

    print("mode unsupported %s"%mode)
    return None

def get_random_rotation(num, angle, p, distribution="normal"):
    """ returns rotation [N, angle] rotation values
    # TODO: this is incorrect
    if p == 1, and angle is not tuple, all angles should be thesame
    """

    #angles = torch.zeros(size, dtype=tensor.dtype, device=tensor.device)
    prob = bernoulli(p, num)#, dtype, device, grad)

    if isinstance(distribution, str):
        distribution = config.RndMode[distribution]

    if distribution == config.RndMode.normal:
        _mean, _angle = totuple2(angle, "normal")
        angles = normal(_mean, _angle/3, num)#, dtype, device, grad)
    else:
        _a, _b = totuple2(angle, "uniform")
        angles = uniform(_a, _b, num)#, dtype)#, device, grad)

    angles.mul_(prob)

    return angles

def get_random(tensor, a, b, p, distribution, independent, grad=None):
    """ random probability distribution spanning a and b with a bernoulli probability p
        returns distribution between a and b conditioned by bernoulli distribution p
        where p results in zero returns one
        TODO merge with get_random_rotation
    """
    size = 1
    dtype = tensor.dtype
    device = tensor.device
    grad = tensor.requires_grad if grad is None else grad

    if independent:
        size = len(tensor)

    prob = bernoulli(p, size, dtype, device, grad)

    if config.RndMode(distribution) == config.RndMode.normal:
        _mean = (a + b)/2
        a = math.fabs(_mean - a)
        out = normal(_mean, a/3, size, dtype, device, grad)
    else:
        out = uniform(a, b, size, dtype, device, grad)

    if not independent:
        out = torch.cat([out for i in range(len(tensor))])
    out = out* prob + torch.ones_like(out) * (1 - prob)
    return out
