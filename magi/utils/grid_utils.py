""" @xvdp
mgrid functions - mgrid is siren's indexing scheme returning a grid of shape
    (positions, indexing_dim)
    it differs from numpy or torch meshgrid in the index ordering

    >>> get_mgrid(sidelen, ranges) # is generalized n dimension grid

    e.g get_mgrid([10,20]) - will generate a 2d grid with 200 (x,y) positions
        get_mgrid([10,20,30,40]) - will generate a 4d grid with (x,y,z,w) positions

    arg ranges=<list(list,)> -> contiguous subgrid on sub range other than <-1,1>
    e.g.
        get_mgrid([10,20,30], ranges=[[3,6],[2,5],[1,3]]) -> 3,3 subgrid within the 10*20*30 grid

    arg indices=True -> mgrid as permutation indices, or permutations in range

    since RAM will quickly fill out, ranges weere added to mgrid
    e.g. get_mgrid([10,20,30], [[5,10],[20,25],[30,35]])
            only generates a 25 position (x,y,z) subgrid within the 6000 positions

    >>> get_subgrid(indices, sidelen) returns positions only of indices within subgrid

    e.g. get_subgrid([34,1231,2], [10,20,30]) -> (3,3) array
"""
import numpy as np
import torch

# pylint: disable=no-member
# pylint: disable=not-callable

def get_mgrid(sidelen, ranges=None, indices=False, strides=1, flat=False, device="cpu"):
    """  nd generic dim meshgrid
    mesh grid indexing [-1,...,1] per dimension

    Breaks get_mgrid sortcut, needs to specifically pass each dimension,
    ie. (sidelen=[256,256]) for a square image, insetad of (sidelen=256, dim=2)
    Args
        sidelen     list, grid steps per dimension, tuples, ints or other iterables will fail
        ranges      list of ranges [None] | [[from:to(exlusive)], ....]
        indices     returns permutation indices [0,n] instead of [-1,1]
        strides     list, stride between samples

    Examples:
        >>> get_mgrid(sidelen=[121,34,34]) # return full meshgrid
        >>> get_mgrid(sidelen=[121,34,34], indices=True) # mesh grid indices

        >>> get_mgrid(sidelen=[121,34,34], ranges=[[0,10]]) # slice in dim 0
        >>> get_mgrid(sidelen=[121,34,34], ranges=[[0,10],[3,4],[5,6]]) # slice all dims
        >>> get_mgrid(sidelen=[121,34,34], ranges=[None, None,[5,6]]) # slice last dim

        >>> get_mgrid(sidelen=[121,34,34], strides=[6, 3, 3]) # strided grid
        >>> get_mgrid(sidelen=[121,34,34], strides=[6, 3, 3], indices) # strided grid

    """
    # sidelen = np.asarray(sidelen)
    sidelen = torch.as_tensor(sidelen).cpu()
    strides = _asarray(strides, len(sidelen))

    # if ranges is None:
    #     return _get_mgrid(sidelen, indices=indices, strides=strides)

    # sidelen in range
    ranges, sublen,  = _check_ranges(sidelen, ranges)
    # strided sidelen
    # sublen = np.ceil(sublen/strides).astype(int)
    sublen = torch.ceil(sublen/strides).to(dtype=torch.int64)

    out = []
    _offset = 0 if indices else 1
    for i, side in enumerate(sidelen):
        # grid step #
        step = 1 if indices else 2/(side - 1)
        # repeats
        pre = sublen[:i].prod() # prod(pre+post) = prod(sidelen)
        post = sublen[i+1:].prod()

        out += [((torch.arange(ranges[i][0], ranges[i][1], strides[i])*step) - _offset
                ).repeat_interleave(post).repeat(pre).view(-1,1)]
    out = torch.cat(out, dim=1)
    if indices:
        if flat:
            out = flatten_igrid(out, sidelen)
            # out  = out.mul(torch.tensor([sidelen[i:].prod() for i in range(1, len(sidelen)+1)])).sum(dim=1)
        out = out.to(dtype=torch.int64)
    return out.to(device=device)

def expand_igrid(indices, sidelen):
    """ expand index grid
    """
    sidelen = torch.as_tensor(sidelen).to(device=indices.device)
    return torch.stack([indices//sidelen[i+1:].prod()%sidelen[i]
                        for i in range(len(sidelen))], dim=1)

def flatten_igrid(grid, sidelen):
    """ flatten index grid
    """
    sidelen = torch.as_tensor(sidelen)
    return grid.mul(torch.tensor([sidelen[i:].prod() for i in
                                  range(1, len(sidelen)+1)], device=grid.device)).sum(dim=1)

def mgrid_from_igrid(indices, sidelen):
    """
    """
    sidelen = torch.as_tensor(sidelen).to(device=indices.device)
    return 2*indices/(sidelen-1) - 1

def _aslist(iterable):
    if isinstance(iterable, (list, tuple)):
        return list(iterable)
    if isinstance(iterable, np.ndarray, torch.Tensor):
        return iterable.tolist()
    assert isinstance(iterable, list), f"convert to list, {type(iterable)} not supported"

def _asarray(item, size):
    if isinstance(item, (int, np.int64)):
        item = torch.tensor([item for _ in range(size)])
    else:
        item = torch.as_tensor(item)
    return item


    # if isinstance(item, np.ndarray):
    #     pass
    # elif isinstance(item, (list, tuple)):
    #     item = np.asarray(item)
    # elif isinstance(item, (int, np.int)):
    #     item = np.array([item for _ in range(size)])
    # elif isinstance(item, torch.Tensor):
    #     item = item.cpu().numpy()
    # else:
    #     raise NotImplementedError(f"wrong type {type(item)}")
    # if len(item) == 1:
    #     item = np.concatenate([item for _ in range(size)])
    # return item

def _check_ranges(sidelen, ranges):
    """ sanity check, range between 0,sidelen[i]
    """
    # sidelen  = _aslist(sidelen)
    # sidelen = np.asarray(sidelen)
    sidelen = torch.as_tensor(sidelen)

    sublen = []
    ranges = _aslist(ranges) if isinstance(ranges, (np.ndarray, list, tuple, torch.Tensor)) else []

    for i in range(len(ranges), len(sidelen)):
        ranges += [[0, sidelen[i]]]

    for i, rng in enumerate(ranges):
        if not isinstance(rng, list) or not len(rng) == 2:
            ranges[i] = [0, sidelen[i]]
        else:
            ranges[i][0] = max(0, ranges[i][0])
            ranges[i][1] = min(ranges[i][1], sidelen[i])
            ranges[i][1] = max(ranges[i][1], ranges[i][0]+1)
        sublen += [ranges[i][1] - ranges[i][0]]
    # ranges = np.asarray(ranges)
    # sublen = np.asarray(sublen)
    ranges = torch.as_tensor(ranges).cpu()
    sublen = torch.as_tensor(sublen).cpu()
    return ranges, sublen

def get_subsidelen(sidelen, max_samples):
    """ returns a dictionary of step: [sidelen,..] <= max samples
    """
    sidelen = np.asarray(sidelen)
    pyr = get_stride_tree(sidelen, max_samples)

    out = {pyr[0]: torch.as_tensor([sidelen//pyr[-1]])}
    for i in range(1, len(pyr)):
        out[pyr[i]] = grid_permutations(sidelen//pyr[len(pyr)-1-i], pyr[i])
    return out

def get_stride_tree(sidelen, max_samples):
    """  calculates sparse occupancy matrices strides
    current: uniformely spaced, but SHOULD bias towards importance of dimension connectivity

    max stride  loads the entire dataset, sparsest
    min stride  loads contiguous block, densent, smallest.

    """
    sidelen = torch.as_tensor(sidelen)
    elems = sidelen.prod()
    offset = elems/max_samples
    if offset <= 1:
        return torch.tensor([1])
    top = torch.ceil(offset**(1/len(sidelen)))
    strides = 2**torch.arange(torch.floor(torch.log2(top)), -1,-1).to(dtype=torch.int64)
    #.to(dtype=torch.int64)
    if top > strides[0]:
        strides = torch.sort(torch.cat([top.view(1).to(dtype=torch.int64),
                                        strides]), descending=True)[0]
    return strides

def get_sparse_grid(sidelen, strides):
    if isinstance(strides, int):
        strides = [strides for _ in range(len(sidelen))]

    indices = get_mgrid(sidelen, strides=strides, indices=True)
    offsets = get_mgrid(strides, indices=True)
    return indices, offsets

def get_inflatable_grid(sidelen, strides):
    if isinstance(strides, int):
        strides = [strides for _ in range(len(sidelen))]
    strides = torch.as_tensor(strides)
    sidelen = torch.as_tensor(sidelen)

    indices = get_mgrid(sidelen/strides, indices=True)
    offsets = get_mgrid(strides, indices=True)
    return indices, offsets

    # handled by datalpader
    # shuffled_perms = torch.randperm(np.prod(strides))

def grid_permutations(sidelen, sizes):
    """  full set of permutations on  data size sidelen, of sizes
    Eg
    sidelen = [40,60]
    sizes = [2,2]
    ranges = [[[0,20],[0,30]], [[0,20][30,60]], [[20,40]],[0,30]], [[20,40]][30,60]]
    """
    if isinstance(sizes, (int, np.int64)):
        sizes = [sizes for i in range(len(sidelen))]
    ranges = [[],[],[]]
    for i, side in enumerate(sidelen):
        for j in range(sizes[i]):
            ranges[i].append([side - (sizes[i]-j)*side//sizes[i], side - (sizes[i]-j-1)*side//sizes[i]])

    _perm = get_mgrid(sizes, indices=True)
    for i in range(1, len(_perm[0])): # flattern
        _perm[:,i] += 1 +_perm[:,i-1].max()

    return torch.as_tensor(ranges).view(-1,2)[_perm]
