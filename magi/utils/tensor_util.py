"""@ xvdp

Tensor manipulation utilities

target tensor format converters
batch, datapoints, dims # this only supports 2d 

# xy to yx order
yx_yx__xy_xy(data)
    alias: xy_xy__yx_yx(), xyxy__yxyx(), yxyx__xyxy()
    [batch, [x,y]]  -> [batch, [y,x]]

#
yx_yx__yxyx(data)
    alias: xy_xy__xyxy()
    [batch, datapoints, dims] -> [batch, datapoints*dims]

yxyx__yx_yx(data)
    alias: xyxy__xy_xy()
    [batch, datapoints*dims] -> [batch, datapoints, dims]

# data, dims to dims, data
n_yx_yx__yyn_xxn(data)
    alias: n_xy_xy__xxn_yyn
    [batch, datapoints,dims] -> [dims, batch*datapoints]

yyn_xxn__n_yx_yx(data, items)
    alias: xxn_yyn__n_xy_xy()
    [batch, datapoints,dims] -> [dims, batch*datapoints]

# absolute to relative bounding boxes
yx_yx__yx_hw(data)
    alias: xy_xy__xy_wh()
    [batch, datapoints, dims] -> [batch, relative_datapoints, dims]

yx_hw__yx_yx(data)
    alias: yx_hw__yx_yx()
    [batch, relative_datapoints, dims] -> [batch, datapoints, dims]

# bounding box as path
yxyx__path(data)
    alias: yxyx__path()
    [batch, datapoints*dims] -> [batch, datapoints*dims*dims] 

yxyx__splitpath(data)
    alias: xyxy__splitpath()
    [batch, datapoints*dims] -> [batch, datapoints*dims, dims] 

splitpath__yx_yx(data)
    alias: splitpath__xy_xy
    [batch, datapoints*dims, dims] -> [batch, datapoints, dims]
path__yxyx(data):
    alias: path__xyxy()
    [batch, datapoints*dims, dims] -> [batch, datapoints*dims]


# bounding box to centerxy, height, width, angle
yxhw__yxhwa(data, angle)
    alias: xywh__yxwha()

yxyx__yxhwa(data, angle):
    alias: xyxy__xywha()

yx_yx__yxhwa(data, angle):
    angle: xy_xy__xywha()

yxhwa__yxhw(data)
    alias: xywha__xywh()

yxhwa__path(data)
    alias: xywha__path()

yxhwa__splitpath
    alias: xywha__splitpath(0
"""
import math
import logging
import torch
from koreto import Col
from .. import config
# pylint: disable=no-member
# pylint: disable=not-callable


# def check_tensor(data, dtype=None, device=None, grad=None):
#     """ Validates tensor or tensor list dtype, device and or grad
#         Validates contiguity
#         Returns size in bytes
#     """
#     size = 0
#     if isinstance(data, (list, tuple)):
#         for _t in data:
#             size += check_tensor(_t, dtype=None, device=None, grad=None)
#         return size

#     assert isinstance(data, torch.Tensor), "tensor expected, found '%s'"%str(type(data))
#     size = get_tensor_size(data)
#     assert_tensor_type(data, dtype)
#     assert_tensor_device(data, device)
#     assert_tensor_grad(data, grad)
#     return size



# class TensorList(list):
#     """ List / Dictionary Composite To handle batches

#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args)
#         for k in kwargs:
#             self.__dict__[k] = kwargs[k]

#     def __setattr__(self, name: str, value: Any) -> None:
#         return super().__setattr__(name, value)

#     def __types__(self):
#         return [type(item) for item in self]


def assert_tensor_list_eq(tensor_list, tolerant=False):
    shapes = [t.shape for t in tensor_list]
    devices = [t.device for t in tensor_list]
    dtypes = [t.dtype for t in tensor_list]

    toshape = None
    todtype = None
    todevice = None

    if not tolerant:
        assert all(shapes[0] == d for d in shapes), "all shapes need to be the same, got %s"%str(shapes)
        assert all(dtypes[0] == d for d in dtypes), "all dtypes need to be the same, got %s"%str(dtypes)
        assert all(devices[0] == d for d in devices), "all devices need to be the same, got %s"%str(devices)

    else:
        if not all(shapes[0] == d for d in shapes):
            assert all(len(shapes[0]) == len(d) for d in shapes), "all shapes need to be length got %s"%str(shapes)
            _shapes = {item:shapes.count(item) for item in shapes}
            toshape = list(_shapes.keys())[list(_shapes.values()).index(max(_shapes.values()))]

        if not all(dtypes[0] == d for d in dtypes):
            _dtypes = {item:dtypes.count(item) for item in dtypes}
            todtype = list(_dtypes.keys())[list(_dtypes.values()).index(max(_dtypes.values()))]

        if not all(devices[0] == d for d in devices):
            _devices = {item:devices.count(item) for item in devices}
            todevice = list(_devices.keys())[list(_devices.values()).index(max(_devices.values()))]
    return toshape, todtype, todevice

def assert_tensor_type(data, dtype=None):
    _msg = "expected dtype %s, got %s: tensor %s"%(str(dtype), str(data.dtype), str(data.shape))
    assert dtype is None or dtype == data.dtype, _msg

def assert_tensor_grad(data, grad=None):
    _msg = "expected requires_grad %s, got %s: tensor %s"%(str(grad), str(data.requires_grad),
                                                           str(data.shape))
    assert grad is None or grad == data.requires_grad, _msg

def assert_tensor_device(data, device=None):
    _msg = "expected device %s, got %s: tensor %s"%(device, data.requires_grad, str(data.shape))
    assert device is None or device == data.device, _msg

def get_tensor_size(data):
    """computes size of tensor in memory
    """
    size = data.numel() * data.element_size()
    if data.requires_grad:
        size *= 2
    return size



# def is_tensor_list(tensor_list, dtype=None, device=None, inplace=True, msg=""):
#     """ checks tensor_list is tuple or list of tensors of the same dtype.
#         if dtype is given, match that dtype
#         if `tensor_list` is empty or None return None
#     """
#     if not isinstance(tensor_list, (list, tuple)):
#         return False

#     _is_tensor_list = False
#     for i, tensor in enumerate(tensor_list):
#         if tensor is None or not tensor:
#             continue
#         elif torch.is_tensor(tensor):
#             _is_tensor_list = True
#             tensor_list[i] = get_tensor(tensor, dtype=dtype, device=device, inplace=inplace, msg="")
#         else:
#             assert not _is_tensor_list, "error, contains tensor and non tensor types {}".format(tensor_list)
#             _is_tensor_list = False

#     if _is_tensor_list:
#         return tensor_list
#     return _is_tensor_list

# def get_tensor(data, dtype=None, device=None, for_display=False, msg=""):
#     """ return tensor within parameters
#     """
#     if not torch.is_tensor(data):
#         return None
#     if for_display:
#         data = data.clone().detach()
#     if dtype is not None and data.dtype != dtype:
#         data.to(dtype=dtype)
#     if device is not None:
#         data.to(device=device)
#     return check_contiguous(data, verbose=config.DEBUG, msg=msg)

# def check_tensor_list(tensor_list, dtype=None, device=None, for_display=False, msg=""):
#     """ checks tensor_list is tuple or list of tensors of the same dtype.
#         if dtype is given, match that dtype
#         if `tensor_list` is empty or None return None
#     """
#     if tensor_list is None or not tensor_list:
#         return None

#     _msg = "expected list of tenors got, '%s'"%type(tensor_list)
#     assert isinstance(tensor_list, (list, tuple)), _msg

#     for i, tensor in enumerate(tensor_list):
#         if tensor is None:
#             continue
#         assert torch.is_tensor(tensor), "expected list of tensors, got list of '%s'"%type(tensor)
#         if dtype is None:
#             dtype = tensor.dtype
#         else:
#             _msg = "expected all tensors of same dtype, got '%s' and '%s'"%(dtype, tensor.dtype)
#             assert dtype == tensor.dtype, _msg

#         if device is None:
#             device = tensor.device
#         else:
#             _msg = "expected all tensors on same device, got '%s' and '%s'"%(device, tensor.device)
#             assert device == tensor.device, _msg

#         if for_display:
#             tensor_list[i] = tensor_list[i].clone().detach()

#     return tensor_list

# def refold_tensors(data, types):
#     out = []
#     for i, datum in enumerate(data):
#         if types[i] in ("tensor", "Tensor"):
#             print("type", types[i], i, len(datum), datum )
#             datum = check_contiguous(torch.cat(datum), verbose=config.DEBUG, msg="folding data")
#         out.append(datum)
#     return out

# def unfold_tensors(data, msg="", inplace=True):
#     """ return tuple of lists
#             data, types
#     General function to handle transform inputs
#         Unfolds list of tensor, target tensors, labels
#         Checks that tensors are contiguous
#         Args:
#             msg     (str) if not contiuous, append to log
#             inplace (bool [True]) if False, makes a copy


#         Currently data inputs expected in format
#         tuple/ list
#             tensor:             image tensor
#             target_tensor_list: list of tensors of same dtype and device as tensor
#             labels:             list of dictionaries
#     """
#     types = ["tensor"]
#     if torch.is_tensor(data):
#         data = get_tensor(data, inplace=inplace, msg=msg)
#         return [data], types

#     if isinstance(data, (tuple, list)):
#         _tensor = get_tensor(data[0], inplace=inplace, msg=msg)
#         assert torch.is_tensor(_tensor), 'img should be Tensor of (3,4,5) dims. Got {}'.format(type(data[0]))
#         data[0] = _tensor

#         if len(data) == 1:
#             return data, types

#         if config.VERBOSE:
#             _dsh = str(data[0].shape)
#             _dtgsh = ""
#             if len(data) > 1 and data[1] is not None:
#                 for i in range(len(data[1])):
#                     _dtgsh = _dtgsh +" "+str(data[1][i].shape)
#             print("  : F.unfold_tensors() \n\timg:%s, \n\ttgsh:%s"%(_dsh, _dtgsh))

#         for i, datum in enumerate(data[1:]):
#             _datum = is_tensor_list(datum, inplace=inplace)
#             if _datum:
#                 data[i] = _datum
#                 types += ["tensor_list"]
#             else:
#                 types += [datum.__class__.__name__]

#     return data, types

# def unfold_data(data, msg="", inplace=True):
#     """General function to handle transform inputs
#         Unfolds list of tensor, target tensors, labels
#         Checks that tensors are contiguous
#         Args:
#             msg     (str) if not contiuous, append to log
#             inplace (bool [True]) if False, makes a copy


#         Currently data inputs expected in format
#         tuple/ list
#             tensor:             image tensor
#             target_tensor_list: list of tensors of same dtype and device as tensor
#             labels:             list of dictionaries
#     """
#     if torch.is_tensor(data):
#         if not inplace:
#             data = data.clone().detach()
#         data = check_contiguous(data, verbose=config.DEBUG, msg=msg)
#         return [data, None, None]

#     if isinstance(data, (tuple, list)):
#         data = list(data)
#         if not torch.is_tensor(data[0]):
#             raise TypeError('img should be Tensor of (3,4,5) dims. Got {}'.format(type(data[0])))

#         if not inplace:
#             data[0] = data[0].clone().detach()
#         data[0] = check_contiguous(data[0], verbose=config.DEBUG, msg=msg)

#         # TODO FIX: this should not return extra Nones
#         if len(data) == 1:
#             return [data[0], None, None]

#         if config.VERBOSE:
#             _dsh = str(data[0].shape)
#             _dtgsh = ""
#             if len(data) > 1 and data[1] is not None:
#                 for i in range(len(data[1])):
#                     _dtgsh = _dtgsh +" "+str(data[1][i].shape)
#             print("  : F.unfold_data() \n\timg:%s, \n\ttgsh:%s"%(_dsh, _dtgsh))

#         target = check_tensor_list(data[1], dtype=data[0].dtype, device=data[0].device,
#                                    inplace=inplace, msg="check_annotation type")

#         if len(data) == 2:
#             return [data[0], target, None]

#         return [data[0], target, data[2]]

def inspect_data_sample(data):
    if isinstance(data, torch.Tensor):
        return 0
    if isinstance(data, (tuple, list)):
        return inspect_data_sample(data[0]) + 1
    return None

def _copy_tensor_options(src, dst):
    dst.to(dtype=src.dtype)
    dst.to(device=src.device)
    dst.requires_grad = src.requires_grad

def order_tensor(data, from_order, to_order):
    """
    this outght to be done with a net or object inheriting classes
    """
    dic = {
        ("yx_hw", "yx_yx"): yx_hw__yx_yx,
        ("xy_wh", "xy_xy"): yx_hw__yx_yx,
        ("yx_hw", "yxyx"): yx_hw__yxyx,
        ("xy_wh", "xyxy"): yx_hw__yxyx,
        ("yx_hw", "path"): yx_hw__path,
        ("xy_wh", "path"): yx_hw__path,
        ("yx_hw", "splitpath"): yx_hw__splitpath,
        ("xy_wh", "splitpath"): yx_hw__splitpath,
        ("yx_hw", "yxhwa"): yx_hw__yxhwa,
        ("xy_wh", "xywha"): yx_hw__yxhwa,
        ("xy_xy", "yx_yx"): yx_yx__xy_xy,
        ("yx_yx", "xy_xy"): yx_yx__xy_xy,
        ("yxyx", "xyxy"): yx_yx__xy_xy,
        ("xyxy", "yxyx"): yx_yx__xy_xy,
        ("xy_xy", "xyxy"): yx_yx__yxyx,
        ("yx_yx", "yxyx"): yx_yx__yxyx,
        ("yxyx", "yx_yx"): yxyx__yx_yx,
        ("xyxy", "xy_xy"): yxyx__yx_yx,
        ("yx_yx", "yx_hw"): yx_yx__yx_hw,
        ("xy_xy", "xy_wh"): yx_yx__yx_hw,
        ("yxyx", "path"): yxyx__path,
        ("xyxy", "path"): yxyx__path,
        ("yxyx", "splitpath"): yxyx__splitpath,
        ("xyxy", "splitpath"): yxyx__splitpath,
        ("splitpath", "yx_yx"): splitpath__yx_yx,
        ("splitpath", "xy_xy"): splitpath__yx_yx,
        ("path", "yxyx"): path__yxyx,
        ("path", "xyxy"): path__yxyx,
        ("yx_yx", "yy_xx"): yx_yx__yy_xx,
        ("xy_xy", "xx_yy"): yx_yx__yy_xx,
        ("yy_xx", "yx_yx"): yy_xx__yx_yx,
        ("xx_yy", "xy_xy"): yy_xx__yx_yx,
        ("yxhw", "yxhwa"): yxhw__yxhwa,
        ("xywh", "xywha"): yxhw__yxhwa,
        ("yxyx", "yxhwa"): yxyx__yxhwa,
        ("xyxy", "xywha"): yxyx__yxhwa,
        ("yx_yx", "yxhwa"): yx_yx__yxhwa,
        ("xy_xy", "xywha"): yx_yx__yxhwa,
        ("yxhwa", "yxhw"): yxhwa__yxhw,
        ("xywha", "xywh"): yxhwa__yxhw,
        ("yxhwa", "path"): yxhwa__path,
        ("xywha", "path"): yxhwa__path,
        ("yxhwa", "splitpath"): yxhwa__splitpath,
        ("xywha", "splitpath"): yxhwa__splitpath,

        ("xy_wh", "xx_yy"): yx_hw__yy_xx,
        ("yx_hw", "yy_xx"): yx_hw__yy_xx,
        ("xx_yy", "xy_wh"): yy_xx__yx_hw,
        ("yy_xx", "yx_hw"): yy_xx__yx_hw
        }
    if (from_order, to_order) in dic:
        data = dic[(from_order, to_order)](data)
    else:
        print("'%s' to '%s' conversion not suported, nothing converted, supported: %s"%
              (from_order, to_order, str(dic.keys())))
    return data

def yx_yx__xy_xy(data):
    """ or yxyx__xyxy
    io tensor shape (..., 2, 2)
    """
    #return flip(data, data.ndimension()-1)
    return data.flip(data.ndimension()-1)

def yx_yx__yxyx(data):
    """
    in tensor  (..., items, dims)
    out tensor (..., items*dims)
    """
    _sh = list(data.shape)
    _sh[-2] *= _sh[-1]
    _sh = _sh[:-1]
    return data.view(*_sh)

def yxyx__yx_yx(data, dims=2):
    """
    out tensor shape (n, items, dims)
    """
    _sh = list(data.shape)
    _it = _sh[-1]//dims
    _sh = _sh[:-1] + [_it, dims]
    return data.view(*_sh)

def yx_yx__yx_hw(data):
    """xy_xy__xy_wh
    io tensor shape (..., 2, 2)
    """
    _data = data.clone().detach()
    _data[:, 1].sub_(_data[:, 0])
    return _data

def yx_hw__yx_yx(data):
    """or xy_wh__xy_xy
    io tensor shape (n, 2, 2)
    """
    _data = data.clone().detach()
    _data[:, 1].add_(_data[:, 0])
    return _data

def yx_hw__yxyx(data):
    """or xy_hw__xyxy
    io tensor shape (n, 2, 2)
    """
    return yx_yx__yxyx(yx_hw__yx_yx(data))

def yx_hw__path(data):
    return yxyx__path(yx_yx__yxyx(yx_hw__yx_yx(data)))

def yx_hw__splitpath(data):
    return yx_hw__path(data).reshape(-1, 4, 2)

def yx_hw__yxhwa(data):
    return yxhw__yxhwa(yx_yx__yxyx(data))

def yxyx__path(data):
    return data[..., torch.LongTensor([0, 1, 2, 1, 2, 3, 0, 3])]

def yxyx__splitpath(data):
    return yxyx__path(data).reshape(-1, 4, 2)

def splitpath__yx_yx(data):
    return torch.stack((data.min(1)[0], data.max(1)[0]), 1)

def path__yxyx(data):
    return yx_yx__yxyx(torch.stack((data.min(1)[0], data.max(1)[0]), 1))

def yx_yx__yy_xx(data):
    """
    input data of shape  (n, items, dim)
    output data of shape (dim, n*items), items
    [[[y,x],[y1,x1]],[[y2,x2],[y3,x3]],...] to [[y,y1,y2,...],[x,x1,x2,...]]
    where
        dims = [x,y,...]
        items = [y,x],[y1,x1],...
        num = [[y,x],[y1,x1]],[[y2,x2],[y3,x3]],...
    for dim, items, num > 0
    eg. 3d
    [[[y,x,z],[y1,x1,z1]],[[y2,x2,z2],[y3,x3,z3]],...]
        to [[y,y1,y2,...],[x,x1,x2,...],[z,z1,z2,...]]

    """
    num, items, dims = data.shape
    return torch.cat([data[:, :, i].view(num, -1) for i in range(dims)]).contiguous(), items

def yy_xx__yx_yx(data, items):
    """
    inptut  data of shape (dim, n*items), items
    output data of shape  (n, items, dim)

    [[y,y1,y2,...],[x,x1,x2,...]] to [[[y,x],[y1,x1]],[[y2,x2],[y3,x3]],...]
    where
        dims = [x,y,...]
        items = [y,x],[y1,x1],...
        num = [[y,x],[y1,x1]],[[y2,x2],[y3,x3]],...
    for dim, items, num > 0
    """
    return torch.stack([datum.view(-1, items) for datum in data], dim=items).contiguous()

def yx_hw__yy_xx(data):
    return yx_yx__yy_xx(yx_hw__yx_yx(data))
    
def yy_xx__yx_hw(data, items):
    return yx_yx__yx_hw(yy_xx__yx_yx(data, items))

def yxhw__yxhwa(data, angle=0):
    """ bounding box to center half width, angle
    in tensor  (..., 4)
    out tensor (..., 5)
    """
    _dt = data.dtype
    _sh = list(data.shape)
    _d1 = data.view(-1, _sh[-1])
    if data.dtype not in (torch.half, torch.float, torch.double):
        print("angles require scalar type, converting to float")
        _dt = torch.float
        _d1 = _d1.to(dtype=torch.float)

    _sh[-1] += 1
    _data = torch.zeros(*_sh, dtype=_dt, device=data.device,
                        requires_grad=data.requires_grad).view(-1, _sh[-1])

    _data[:, 2:4].add_(_d1[:, 2:]/2)
    _data[:, :2].add_(_d1[:, :2] + _d1[:, 2:]/2)
    _data[:, 4] = angle
    return _data.view(_sh)

def yx_yx__yxhwa(data, angle=0):
    """ bounding box to center half width, angle
    in tensor  (..., 2, 2)
    out tensor (..., 5)
    """
    return yxhw__yxhwa(yx_yx__yxyx(yx_yx__yx_hw(data)), angle)

def yxyx__yxhwa(data, angle=0):
    """ bounding box to center half width, angle
    in tensor  (..., 4)
    out tensor (..., 5)
    """
    return yx_yx__yxhwa(yxyx__yx_yx(data), angle)

def yxhwa__yxhw(data):
    """ center(y,x) half(h,w), angle to bounding box
    in tensor  (..., 5)
    out tensor (..., 4)
    """
    _a = torch.fmod(data[..., -1], math.pi)/math.pi
    _a.mul_(-1).add_(0.5).mul_(2).abs_()
    _h = data[..., 2]*_a + data[..., 3]*(1 - _a)
    _w = data[..., 3]*_a + data[..., 2]*(1 - _a)

    _y = data[..., 0] - _h
    _x = data[..., 1] - _w

    _sh = list(data.shape)
    _sh[-1] -= 1

    _data = torch.zeros(*_sh, dtype=data.dtype, device=data.device,
                        requires_grad=data.requires_grad)
    _data[..., 0].add_(_y)
    _data[..., 1].add_(_x)
    _data[..., 2].add_(_h*2)
    _data[..., 3].add_(_w*2)

    return _data

def yxhwa__yx_hw(data):
    """ center(y,x) half(h,w), angle to bounding box
    in tensor  (..., 5)
    out tensor (..., 2, 2)
    """
    return yxyx__yx_yx(yxhwa__yxhw(data))

def yxhwa__yx_yx(data):
    """ center(y,x) half(h,w), angle to bounding box
    in tensor  (..., 5)
    out tensor (..., 2, 2)
    """
    return yx_hw__yx_yx(yxhwa__yx_hw(data))

def yxhwa__path(data):
    """ center(y,x) half(h,w), angle to path
    out tensor (..., 5)
    in tensor  (..., 8)
    """
    _dt = data.dtype
    _dv = data.device
    _gr = data.requires_grad
    _mirror = torch.tensor([[1],[-1.]], dtype=_dt, device=_dv, requires_grad=_gr)
    # angles ? angle = data[:,4]
    # > get rotations - per item rotation - tbd
    # shear
    _angle = data[0, 4].item()
    _mat = get_rotation(_angle, _dt, _dv, _gr).unsqueeze(0)

    _c = data[..., :2].t()
    _a = torch.bmm(_mat, data[:, 2:4].t().unsqueeze(0)).squeeze(0)
    _b = torch.bmm(_mat, (data[:, 2:4].t()* _mirror).unsqueeze(0)).squeeze(0)
    out = torch.cat(((_c-_a), (_c-_b), (_c+_a), (_c+_b))).t()
    _copy_tensor_options(data, out)
    return out

def yxhwa__splitpath(data):
    """ center(y,x) half(h,w), angle to spit path
    out tensor (..., 5)
    in tensor  (..., 4, 2)
    """
    return yxhwa__path(data).reshape(-1, 4, 2)

def shift_target(target, offset, mode=None, sign=1):
    """New Tensor, Added or Subtracted from Target Tensor
    """
    return shift_target_(target.clone().detach(), offset, mode, sign)

def shift_target_(target, offset, mode=None, sign=1):
    """adds or subtracts from target tensor
        in place
    """
    if mode is None:
        mode = config.BOXMODE
    elif isinstance(mode, str):
        mode = config.BoxMode(mode)
    assert mode in config.BoxMode, "mode has to be enum of Boxmode type, not '%s'"%type(mode)

    if isinstance(offset, (int, float)):
        offset = (offset, offset)

    if isinstance(offset, (tuple, list)):
        offset = torch.tensor(offset, device=target.device, dtype=target.dtype)

    offset = offset * sign

    if mode._name_[0] == "x":
        offset = offset.flip(0)

    if mode in (config.BoxMode.xywh, config.BoxMode.yxhw):
        target[:, 0].add_(offset)

    if mode in (config.BoxMode.yxyx, config.BoxMode.yxyx, config.BoxMode.ypath, config.BoxMode.xpath):
        target[:, :].add_(offset)

    if mode in (config.BoxMode.yxhwa, config.BoxMode.xywha):
        target[:, :2].add_(offset)

    return target


def get_rotation(angle, dtype, device, grad):
    """expects rotations in pis"""
    angle = _pi_to_rad(angle) * -1.0
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, -sin], [sin, cos]], dtype=dtype, device=device, requires_grad=grad)

def _deg_to_pi(angle): # math.pi*
    return (angle%360)/180
def _deg_to_rad(angle):
    return math.pi*_deg_to_pi(angle)
def _pi_to_rad(angle):
    return angle*math.pi
def _pi_to_deg(angle):
    return (angle*180)%360
def _rad_to_deg(angle):
    return _pi_to_deg(angle)/math.pi

#aliases

#flip dimensions
xy_xy__yx_yx = yx_yx__xy_xy
xyxy__yxyx = yx_yx__xy_xy
yxyx__xyxy = yx_yx__xy_xy
xy_xy__xyxy = yx_yx__yxyx
xyxy__xy_xy = yxyx__yx_yx

# absolute to/from relative
xy_xy__xy_wh = yx_yx__yx_hw
xy_wh__xy_xy = yx_hw__yx_yx
xy_wh__xyxy = yx_hw__yxyx

# aboslute to/from path
xyxy__path = yxyx__path
xyxy__splitpath = yxyx__splitpath
splitpath__xy_xy = splitpath__yx_yx
path__xyxy = path__yxyx

#to and from center size angle
xywh__xywha = yxhw__yxhwa
xyxy__xywha = yxyx__yxhwa
xy_xy__xywha = yx_yx__yxhwa

xywha__xywh = yxhwa__yxhw
xywha__xy_wh = yxhwa__yx_hw
xywha__path = yxhwa__path
xywha__splitpath = yxhwa__splitpath

xy_xy__xx_yy = yx_yx__yy_xx
xx_yy__xy_xy = yy_xx__yx_yx

xy_wh__path = yx_hw__path
xy_wh__splitpath: yx_hw__splitpath
xy_wh__xywha = yx_hw__yxhwa

xy_wh__xx_yy = yx_hw__yy_xx
xx_yy__xy_wh = yy_xx__yx_hw
