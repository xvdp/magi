"""@xvdp
test for randomization samlers
"""
from magi.transforms import Values, Probs
from magi.transforms.functional_base import p_all, p_none, p_some
import torch

# pylint: disable=no-member
torch.set_default_dtype(torch.float32) # reset float16 if set by previous test


def test_constant():

    a=torch.tensor([[1,0.2,0.3]])
    constant = Values(a)
    shape = (4,3,10,10)
    assert constant.sample(shape).ndim == len(shape)


    a=torch.tensor([1,0.2,0.3])
    constant = Values(a, expand_dims=None)

    assert constant.sample([34]).shape == constant.sample().shape == (1,3)

    dic = {'a':-1, 'b':0, 'distribution':'Uniform', 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == (1,)
    assert sat.sample(3).shape == (3,)
    assert sat.sample((3,3,3,3)).shape ==(3, 1, 1, 1)


    dic = {'a':[-1, 1,2], 'b':0, 'distribution':'Uniform', 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == (1,3)

def test_categorical():
    v = Values(a=torch.tensor([1,0.2,0.3]), b=2, expand_dims=(0,1))
    assert torch.equal(v.batch_mask, torch.tensor([1]))
    assert v.sample(4).shape == (4,3)

    assert v.sample([4,12,23,32]).shape == (4,1,1,1)

    v = Values(a=0.1, b=20, c=300, d=4000, expand_dims=(0,1,3))
    v.sample().shape == (1,)
    v.sample(23).shape == (23,)

    assert torch.any(v.sample(1000) == 0.1)
    assert torch.any(v.sample(1000) == 20.)
    assert torch.any(v.sample(1000) == 300.)
    assert torch.any(v.sample(1000) == 4000.)

    assert v.sample((4,3,30,30)).shape == (4, 3, 1, 30)

    v = Values(a=[0.1,1,2], b=20, expand_dims=(1,3))
    assert torch.equal(v.batch_mask, torch.tensor([0]))

    v = Values(a=[0.1,1,2], b=20, expand_dims=(0,1,3))
    assert torch.equal(v.batch_mask, torch.tensor([1,]))

def test_uniform():
    v = Values(a=0.1, b=20, expand_dims=(0,1,3), distribution="Uniform")
    assert torch.equal(v.batch_mask, torch.tensor([1, 1, 0, 1])) # Ok

    v = Values(a=[0.1,1,2], b=20, expand_dims=(0,1,3), distribution="Uniform")
    assert torch.equal(v.batch_mask, torch.tensor([1]))


def test_matching():
    # input value size > 1 places it in the Channels
    dic = {'a':[-1,1,2], 'b':0, 'distribution':'Categorical', 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1, 3])
    assert sat.sample((5,)).shape == torch.Size([5, 3])
    assert sat.sample((5,4,6,4)).shape ==torch.Size([5, 1, 1, 1])

    dic = {'a':-1, 'b':0, 'distribution':'Categorical', 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1])
    assert sat.sample((5,)).shape == torch.Size([5])
    assert sat.sample((5,4,6,4)).shape ==torch.Size([5, 1, 1, 1])
    
    dic = {'a':-1, 'b':0, 'distribution':'Uniform', 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1])
    assert sat.sample((5,)).shape == torch.Size([5])
    assert sat.sample((5,4,6,4)).shape ==torch.Size([5, 1, 1, 1])

    # Constants do broadcast
    dic = {'a':-1, 'b':None, 'distribution':None, 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1])
    assert sat.sample((5,)).shape == torch.Size([5])
    assert sat.sample((5,4,6,4)).shape ==torch.Size([5, 1, 1, 1])

    dic = {'a':[-1,2,3], 'b':None, 'distribution':None, 'expand_dims':0}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1, 3])
    assert sat.sample((5,)).shape == torch.Size([5, 3])
    assert sat.sample((5,3,6,4)).shape ==torch.Size([5, 3, 1, 1])

    # TODO: expand beyond width?
    dic = {'a':[-1,2,3], 'b':None, 'distribution':None, 'expand_dims':(0,3)}
    sat = Values(**dic)
    assert sat.sample().shape == torch.Size([1, 3])
    assert sat.sample((5,)).shape == torch.Size([5, 3])
    # assert sat.sample((5,4,6,4)).shape ==torch.Size([5, 3, 1, 4]) << will fail, should not


def test_ps():
    pd = {'p':[0.5,0.2,0.1], 'expand_dims':0}
    ps = Probs(**pd)

    assert ps.sample().shape == torch.Size([1, 3])
    assert ps.sample((5,)).shape == torch.Size([5, 3])
    assert ps.sample((5,4,6,4)).shape ==torch.Size([5, 1, 1, 1]) ###

    pd = {'p':[1], 'expand_dims':0}
    ps = Probs(**pd)
    assert ps.sample().shape == torch.Size([1])
    assert ps.sample((5,)).shape == torch.Size([1])
    assert ps.sample((5,4,6,4)).shape ==torch.Size([1,1,1,1])

    pd = {'p':1, 'expand_dims':0}
    ps = Probs(**pd)
    assert ps.sample().shape == torch.Size([1])
    assert ps.sample((5,)).shape == torch.Size([1])
    assert ps.sample((5,4,6,4)).shape ==torch.Size([1,1,1,1])

    pd = {'p':0.5, 'expand_dims':0}
    ps = Probs(**pd)
    assert ps.sample().shape == torch.Size([1])
    assert ps.sample((5,)).shape == torch.Size([5])
    assert ps.sample((5,4,6,4)).shape ==torch.Size([5,1,1,1])

def test_fill_missing_args():
    dic = {'a':0.5, 'distribution':'Bernoulli'}
    v = Values(**dic)
    p = Probs() # default returns constant
    p.sample()
    assert p.vals is not None and p.__ is None


def test_p():
    # test that gradients cuda nad dtype dont break ptests
    o = torch.ones([1,3,13,17], dtype=torch.float16, device="cuda")
    o.requires_grad = True

    assert p_all(o)

    o = torch.ones([1,3,13,17], dtype=torch.float16, device="cuda")
    o[-1,-1,-1,-1] = 0
    o.requires_grad = True

    assert p_some(o)

    o = o.mul(0.)
    assert p_none(o)

