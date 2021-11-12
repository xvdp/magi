"""@xvdp
"""
# import torch
# from magi.transforms import Randomize


# _single = ('Poisson', 'Exponential', 'Dirichlet', 'Chi2')
# _double = ('Normal', 'LogNormal', 'Laplace', 'Gumbel', 'Cauchy', 'Uniform', 'VonMises',
#             'Gamma', 'FisherSnedecor','Beat', 'Kumaraswamy', 'Weibull', 'kl_divergence')

# #pylint: disable=no-member
# def test_none():
#     _len = 100
#     D = Randomize(a=0)
#     d, p = D.sample(_len)
#     assert not torch.any(d) and len(d) == _len
#     assert torch.all(p) and len(p) == 1

# def test_none_p():
#     _len = 1000
#     D = Randomize(a=0, p=0.5)
#     d, p = D.sample(_len)
#     assert not torch.any(d) and len(d) == _len
#     assert torch.any(p) and len(p) == _len

# def test_singles():
#     for distribution in _single:
#         D = Randomize(a=4, p=0.5, distribution=distribution)
#         d = D.sample(5)

# def test_doubles():
#     for distribution in _double:
#         D = Randomize(a=1, b=3, p=0.5, distribution=distribution)
#         d = D.sample(5)

# def test_normrange_opts():

#     b=1
#     a=0
#     sz=20
#     R = Randomize(a=a, b=b, p=0.5, shape=(1,3,1,1), distribution='Uniform', per_channel=True, per_sample=True, dtype=torch.float32)
#     s,p = R.sample(2)
#     s,p = R.sample(sz)
#     assert s.shape == (sz, 3, 1, 1)
#     assert p.shape == (sz, 1)
#     assert torch.all(R._sample >= 0) and torch.all(R._sample <= 1)
#     assert torch.equal(s, R._sample) and torch.equal(p, R._p)

#     R.to(device='cuda')
#     s,p = R.sample(5)
#     assert torch.equal(s, R._sample) and torch.equal(p, R._p)
#     assert s.device.type == 'cuda' and p.device.type == 'cuda'




