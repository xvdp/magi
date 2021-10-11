"""@xvdp
Tests open file:

1. transforms.Open() used as generic open is fault tolreant, as is intended to be used in the wild and as shortcut
2. functionalio and imageio are strict

pytest --duractions 0
pytest test_open.py::test_open_url_np
"""
from typing import Union
import os
import os.path as osp

import pytest
import numpy as np
import torch
from koreto import ObjDict
from magi import config
import magi.transforms as mt
from magi.utils.imageio import get_cache_name
from _test_common import make_fake_files, assert_msg, dict_str, source_url




# create fake files # add more types in _test_common._fake_files_specs()
FILES_INFO = make_fake_files()
def _get_file(index=None, **kwargs):
    """ when all files 'should' work, randomize
    to minimize tests and stochastically catch arbitrary assumed errors
    Args
        index   [None]
    kwargs in [name, channels, size, ext]
        if None, filter files by attribute
    """
    global FILES_INFO
    files_info = FILES_INFO.copy()
    if index is None:
        for k in kwargs:
            if k not in files_info[0]:
                continue
            files_info = [f for f in files_info if f[k] == kwargs[k]]
        index = np.random.randint(len(files_info))
    return files_info[index]

def _rnd_opt(key: str, choices: Union[list, tuple], no_option_option: bool=True) -> dict:
    """ get random option -> {}
    Args
        key                 (str) key to add or change
        choices             (list, tuple), set of possible choices
        no_option_option    (bool [True]) return empty dict 
    """
    out = {}
    num = len(choices) + int(no_option_option)
    choice = np.random.randint(num)
    if choice < len(choices):
        out[key] = choices[choice]
    return out

# Tests for Open(), should pass or warn, but not fail

# pylint: disable=no-member
# 1. Test Default as torch
def test_open_as_torch_with_defaults_all_files():
    """ commmon case, declare Open as mt.Open()
            pass filename: all files
            defaults: torch, float32, nograd
    """
    O = mt.Open()
    opts = defaults = ObjDict(O.__dict__)
    for fileinfo in FILES_INFO:
        img = O(fileinfo.name)
        print("opendict", dict_str(O.__dict__, "blue"))
        print("fileinfo", dict_str(fileinfo, "green"))
        assert_torch(img, opts, fileinfo, "mt.Open", defaults)

# 2. Test Default as numpy
def test_open_as_numpy_with_defaults():
    """ commmon case, declare Open as mt.Open()
            pass filename: signle file
            defaults: torch, float32, nograd
    """
    opts = ObjDict({'dtype': None, 'out_type': 'numpy', 'channels': 3, 'transforms': None})
    O = mt.Open(**opts)
    defaults = ObjDict(O.__dict__)

    fileinfo = _get_file()
    img = O(fileinfo.name)
    print("opendict", dict_str(O.__dict__, "blue"))
    print("fileinfo", dict_str(fileinfo, "green"))

    assert_basic(img, opts, fileinfo, "mt.Open", defaults)

# 3. Test channels default and options
def test_channels_None_1_3_4():
    """ channels:   default=3
        channels:   None: read file default
        channels:   [1,3,4]: expand or contract channels

    randomly open as numpy or torch
    """
    fileinfo_3 = _get_file(None, channels=3)
    fileinfo_1 = _get_file(None, channels=1)
    #
    # default to file channels
    opts = ObjDict(**{'channels': None}, **_rnd_opt("out_type", ["numpy", "torch"]))

    O = mt.Open(**opts)
    defaults = ObjDict(O.__dict__)

    # 1 channel
    img = O(fileinfo_1.name)
    assert_basic(img, opts, fileinfo_1, "mt.Open", defaults)
    # 3 channels
    img = O(fileinfo_3.name)
    assert_basic(img, opts, fileinfo_3, "mt.Open", defaults)

    # override channels on __call
    # to 1 channel
    opts = ObjDict(**{'channels': 1}, **_rnd_opt("out_type", ["numpy", "torch"]))
    O = mt.Open(**opts)
    defaults = ObjDict(O.__dict__)
    img = O(fileinfo_3.name, **opts)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo_3, "mt.Open", defaults)

    # to 3 channels
    opts = ObjDict({'channels': 3})
    img = O(fileinfo_1.name, **opts)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo_1, "mt.Open", defaults)

    # to 4 channels
    opts = ObjDict({'channels': 4}, **_rnd_opt("out_type", ["numpy", "torch"]))
    O = mt.Open(**opts)
    img = O(fileinfo_3.name)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo_3, "mt.Open", defaults)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not present in system")
def test_gpu():
    """
    """
    opts = ObjDict({'dtype': None, 'out_type': 'torch', 'device':'cuda'})
    fileinfo = _get_file()

    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

    # test on change
    opts = ObjDict({'dtype': None, 'out_type': 'torch', 'device':'cpu'})
    img = O(fileinfo.name, **opts)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not present in system")
def test_dtypes_torch():
    # float 16, 64 - cuda /cpu
    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float16", 'out_type': 'torch', 'device':'cuda'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float64", 'out_type': 'torch', 'device':'cuda'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float16", 'out_type': 'torch', 'device':'cpu'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float64", 'out_type': 'torch', 'device':'cpu'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)


def test_dtypes_numpy():

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float16", 'out_type': 'numpy'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo, "mt.Open", defaults)

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "float32", 'out_type': 'numpy'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo, "mt.Open", defaults)

    fileinfo = _get_file()
    opts = ObjDict({'dtype': "uint8", 'out_type': 'numpy'})
    O = mt.Open(**opts)
    img = O(fileinfo.name)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, fileinfo, "mt.Open", defaults)

def test_batch_torch():
    global FILES_INFO
    filesinfos = FILES_INFO.copy()
    opts = ObjDict({'dtype': "float32", 'out_type': 'torch'})
    O = mt.Open(**opts)
    batch = [filesinfos[0].name, filesinfos[0].name, filesinfos[0].name]
    img = O(batch)
    defaults = ObjDict(O.__dict__)
    assert_torch(img, opts, filesinfos[0], "mt.Open", defaults, batch_size=len(batch))

    opts = ObjDict({'dtype': "float64", 'out_type': 'torch', 'channels':None})
    batch = [filesinfos[0].name, filesinfos[1].name, filesinfos[2].name]
    img = O(batch, **opts)
    assert len(img) == len(batch) and isinstance(img, list), f"expected type list in opening unequal sizes, got {type(img)}, {len(img)}, {len(batch)}"

def test_batch_numpy():
    global FILES_INFO
    filesinfos = FILES_INFO.copy()
    opts = ObjDict({'dtype': "float16", 'out_type': 'numpy'})
    O = mt.Open(**opts)
    batch = [filesinfos[0].name, filesinfos[0].name, filesinfos[0].name]
    img = O(batch)
    defaults = ObjDict(O.__dict__)
    assert_basic(img, opts, filesinfos[0], "mt.Open", defaults, batch_size=len(batch))

    opts = ObjDict({'dtype': "uint8", 'out_type': 'numpy', 'channels':None})
    batch = [filesinfos[0].name, filesinfos[1].name, filesinfos[2].name]
    img = O(batch, **opts)
    assert len(img) == len(batch) and isinstance(img, list), f"expected type list in opening unequal sizes, got {type(img)}, {len(img)}, {len(batch)}"

def test_open_url_torch():

    url = source_url() # if not 200: skip test
    if url:
        fileinfo = ObjDict(name=url)
        opts = ObjDict({'dtype': "float32", 'out_type': 'torch', 'device':'cpu'})
        O = mt.Open(**opts)
        img = O(url)
        defaults = ObjDict(O.__dict__)
        assert_url_torch(img, opts, fileinfo, "mt.Open", defaults)

        cached = get_cache_name(url)
        assert osp.isfile(cached), f" file '{cached}' not found"

        # load cached file
        fileinfo = ObjDict(name=cached)
        defaults = ObjDict(O.__dict__)
        opts = ObjDict({'dtype': "float16", 'out_type':'torch', 'channels':1, 'device':'cuda'})

        img = O(cached, **opts)
        assert_url_torch(img, opts, fileinfo, "mt.Open", defaults)
        try:
            os.remove(cached)
        except:
            pass

def test_open_url_np():
    # load  url numpy
    url = [source_url(), source_url()]
    if all(url):
        opts = ObjDict({'dtype': "float32", 'out_type': 'numpy'})
        O = mt.Open(**opts)
        img = O(url)
        defaults = ObjDict(O.__dict__)
        fileinfo = ObjDict(name=url[0])
        assert_url(img[0], opts, fileinfo, "mt.Open", defaults)
        fileinfo = ObjDict(name=url[0])
        assert_url(img[1], opts, fileinfo, "mt.Open", defaults)

        cached = []
        for _u in url:
            _cached = get_cache_name(_u)
            assert osp.isfile(_cached), f" file '{_cached}' not found"
            cached.append(_cached)

        # check cached file

        opts = ObjDict({'dtype': "uint8", 'out_type':'numpy', 'channels':1})
        img = O(cached[0], **opts)
        print(img.dtype, img.shape)
        fileinfo = ObjDict(name=cached[0])
        defaults = ObjDict(O.__dict__)

        assert_url(img, opts, fileinfo, "mt.Open", defaults)

        for _cached in cached:
            try:
                os.remove(_cached)
            except:
                pass

#
## FAIL TESTS
# . change out type in __call__ expected to fail
@pytest.mark.xfail
def test_change_type_on_call_xfail():
    opts = ObjDict({'dtype': None, 'out_type': 'torch', 'channels': 3, 'transforms': None})
    O = mt.Open(**opts)
    defaults = ObjDict(O.__dict__)
    fileinfo = _get_file()
    img = O(fileinfo.name)
    img = O(fileinfo.name, out_type="numpy")

# on torch, open only accepts floating types, not uint
@pytest.mark.xfail
def test_pytorch_not_uint8_xfail():
    opts = ObjDict({'dtype': 'uint8', 'out_type': 'torch', 'channels': 3, 'transforms': None})
    O = mt.Open(**opts)
    defaults = ObjDict(O.__dict__)
    fileinfo = _get_file()
    img = O(fileinfo.name)
    assert_torch(img, opts, fileinfo, "mt.Open", defaults)

##
#
# assert formatting
#
def assert_channels(img, opt_dict, fileinfo, func, defaults):
    c_i = 1 if isinstance(img, torch.Tensor) else -1
    img_channels = img.shape[c_i]

    requested_channels = opt_dict.channels if "channels" in opt_dict else defaults.channels
    input_channels  = fileinfo.channels

    if requested_channels is not None:
        assert img_channels == requested_channels, assert_msg("requested channels", requested_channels, img_channels, opt_dict, fileinfo.name, func)
    else:
        assert img_channels  == input_channels, assert_msg("input channels", input_channels, img_channels, opt_dict,  fileinfo.name, func)

def assert_size(img, opt_dict, fileinfo, func, defaults, batch_size=None):
    tensor_size = tuple(img.shape[2:]) if isinstance(img, torch.Tensor) else tuple(img.shape[-3:-1])
    input_size = tuple(fileinfo.size[:2])
    assert tensor_size == input_size, assert_msg("input size", input_size, tensor_size, opt_dict, fileinfo.name, func)

    if batch_size is not None:
        assert len(img) == batch_size, assert_msg("batch size", batch_size, len(img), opt_dict, fileinfo.name, func) 

def assert_type(img, opt_dict, fileinfo, func, defaults):
    requested_type = opt_dict.out_type if "out_type" in opt_dict and opt_dict.out_type is not None else defaults.out_type
    found_type = None
    if isinstance(img, torch.Tensor):
        found_type = "torch"
    elif isinstance(img, np.ndarray):
        found_type = "numpy"

    assert found_type == requested_type, assert_msg("out type", requested_type, found_type, opt_dict, fileinfo.name, func)

def assert_dtype(img, opt_dict, fileinfo, func, defaults):

    requested_dtype = opt_dict.dtype if "dtype" in opt_dict and opt_dict.dtype is not None else config.DTYPE
    _is_floating = "float" in requested_dtype
    found_dtype = img.dtype


    if isinstance(img, torch.Tensor):
        requested_dtype = torch.__dict__[requested_dtype]
    elif isinstance(img, np.ndarray):
        requested_dtype = np.__dict__[requested_dtype]

    assert found_dtype == requested_dtype, assert_msg("dtype", requested_dtype, found_dtype, opt_dict, fileinfo.name, func)
    if _is_floating:
        assert img.max() <= 1.0, assert_msg(f"dtype {requested_dtype}, float cannot > 1", img.max(), 1.0, opt_dict, fileinfo.name, func)
    else:
        assert img.max() > 1, assert_msg(f"dtype {requested_dtype}, uint8 expected > 1", img.max(), 1.0, opt_dict, fileinfo.name, func)
        

def assert_grad(img, opt_dict, fileinfo, func, defaults):
    """torch only
        second check, incompatibility with INPLACE
    """
    if isinstance(img, torch.Tensor):
        found_grad = img.requires_grad
        if "grad" in opt_dict and opt_dict.grad is not None:
            requested_grad = opt_dict.grad
            assert found_grad == requested_grad, assert_msg("grad", requested_grad, found_grad, opt_dict, fileinfo.name, func)
        if found_grad:
            assert not config.INPLACE, assert_msg("grad is incompatible with Inplace", config.INPLACE, found_grad, opt_dict, fileinfo.name, func)

def assert_device(img, opt_dict, fileinfo, func, defaults):
    """ torch only
    """
    if isinstance(img, torch.Tensor):
        requested_device = torch.device(opt_dict.device) if "device" in opt_dict else torch.device(defaults.device)
        found_device = torch.device(img.device)

        if requested_device.index is None or found_device.index is None:
            device_index_eq = True
        else:
            device_index_eq = requested_device.index == found_device.index

        assert found_device.type == requested_device.type and device_index_eq, assert_msg("device", requested_device, found_device, opt_dict, fileinfo.name, func)


def assert_basic(img, opt_dict, fileinfo, func, defaults, batch_size=None):
    assert_type(img, opt_dict, fileinfo, func, defaults)
    assert_channels(img, opt_dict, fileinfo, func, defaults)
    assert_size(img, opt_dict, fileinfo, func, defaults, batch_size=batch_size)
    assert_dtype(img, opt_dict, fileinfo, func, defaults)

def assert_torch(img, opt_dict, fileinfo, func, defaults, batch_size=None):
    assert_basic(img, opt_dict, fileinfo, func, defaults, batch_size=batch_size)
    assert_grad(img, opt_dict, fileinfo, func, defaults)
    assert_device(img, opt_dict, fileinfo, func, defaults)


def assert_url(img, opt_dict, fileinfo, func, defaults, batch_size=None):
    assert_type(img, opt_dict, fileinfo, func, defaults)
    assert_dtype(img, opt_dict, fileinfo, func, defaults)

def assert_url_torch(img, opt_dict, fileinfo, func, defaults, batch_size=None):
    assert_url(img, opt_dict, fileinfo, func, defaults, batch_size=batch_size)
    assert_grad(img, opt_dict, fileinfo, func, defaults)
    assert_device(img, opt_dict, fileinfo, func, defaults)


"""


def test_outtype(open_tests):
    for _test in open_tests:
        assert_type(_test.tesnor, _test.opts, _test.fileinfo, _test.info)

def test_dtype(open_tests):
    for _test in open_tests:
        assert_dtype(_test.tesnor, _test.opts, _test.fileinfo, _test.info)

def test_grad(open_tests):
    for _test in open_tests:
        assert_grad(_test.tesnor, _test.opts, _test.fileinfo, _test.info)

def test_inplace(open_tests):
    for _test in open_tests:
        if isinstance(img, torch.Tensor) and opt_dict.inplace is not None:
            assert config.INPLACE == opt_dict.inplace, assert_msg("inplace_set", opt_dict.inplace, config.INPLACE, opt_dict, fileinfo.name, func)

"""
