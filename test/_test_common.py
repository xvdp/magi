""" @xvdp common to test"""
from typing import Any
import requests
import os
import os.path as osp
import numpy as np
from PIL import Image

from koreto import ObjDict, Col
import magi.transforms as at
from magi import config

TEST_DIR = "magi_test_data"

def _fake_files_specs(folder=TEST_DIR):
    """
        TODO: other image types.
        TODO: use tmp_folder structure instead? or not.
    """
    def _rndsize(smin=32,smax=2048):
        size = [np.random.randint(smin,smax)]
        if np.random.randint(2): # is square
            size += size
        else:
            size += [np.random.randint(smin, smax)]
        return size

    return [
        ObjDict(name=osp.join(folder, "fake_file3.jpg"), size=[*_rndsize(32, 2048), 3], channels=3, ext=".jpg"),
        ObjDict(name=osp.join(folder, "fake_file3.png"), size=[*_rndsize(32, 2048), 3], channels=3, ext=".png"),
        ObjDict(name=osp.join(folder, "fake_file1.jpg"), size=_rndsize(32, 2048), channels=1, ext=".jpg"),
    ]

def make_fake_files(seed=None, folder=TEST_DIR):
    """
    """
    os.makedirs(folder, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    fileinfos = _fake_files_specs(folder)
    for fifo in fileinfos:
        #print(fileinfo.name, "seed",  seed)
        if seed is None or not osp.isfile(fifo.name):
            #print(" ->", fileinfo.size)
            fake = np.random.randint(0,255, np.prod(fifo.size)).astype(np.uint8).reshape(fifo.size)
            Image.fromarray(fake).save(fifo.name)
    return fileinfos

def dict_str(in_dic: dict, strip: bool=True, color: str=None) -> str:
    """ retunrs stripped dict for logging
        Args
            input   (dict)
            strip   (bool [True])  removed None
    """
    if in_dic is not None:
        out = in_dic
        if strip:
            out = {k:v for k,v in out.items() if v is not None}
            if not out:
                return ""

        out = f"{out}".replace(" ", "")
        if color is not None and color[0].lower() in ('b', 'y', 'r', 'g'):
            col = {'b':Col.BB, 'y':Col.YB, 'r':Col.RB, 'g':Col.GB}
            out = f"(**{col[color[0].lower()]}{out}{Col.AU})"
        # f"(**{Col.BB}{out}{Col.AU})".replace(" ", "")

        return out
    return

def assert_msg(msg: str, val_request: Any, val_recieved: Any, opts: dict=None, name:str="", fun: str="") -> str:
    """ generic assert message
    Args
        msg         (str) assert message
        val_request, val_recieved  (Any) values to compare
        opts        (dict[None]) function parameters
        name        (str [""]) str argument if fun is a callable
        fun         (str [""]) function name
    Examples
        >>> assert_msg(msg="requested channels", val0=requested_channels, val1=tensor_channels, opt=_test.opt, name=_test.fileinfo.name)
        >>> assert_msg("requested channels", requested_channels, tensor_channels, _test.opt, _test.fileinfo.name, _test.fileinfo.info)
    """
    # if options passed
    _opts = dict_str(opts, color="blue")
    if name:
        name = f"{Col.AU}('{name}')"
    if fun and not name and not _opts:
        fun +="()"

    return f"{Col.YB}{msg} requested: {val_request}, got: {Col.RB}{val_recieved}; {Col.AU}{fun}{_opts}{name}"

def source_url():
    _urls = ["https://i.guim.co.uk/img/media/75aa2f1d235331c0d46dda31fdf072db39393fa0/0_0_4000_2667/master/4000.jpg?width=1920&quality=85&auto=format&fit=max&s=86f1596ec8a97cbda810be19531c971f",
            "https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/pia01492-main.jpg",
            "https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/iss065e334875.jpg",
            "https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/ta010359_lucy3-b-orbit-crop_0.png",
            "https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/landsat_art_greenland_1920x1200.jpeg",
            "https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/potw2137b.jpg"]
    while len(_urls):
        idx = np.random.randint(len(_urls))
        url = _urls.pop(idx)
        if requests.get(url).status_code == 200:
            return url

    assert False, "urls are down, skip url tests"