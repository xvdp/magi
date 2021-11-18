"""@xvdp
Display utilities
"""
from typing import Union, Optional
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
import torch
from koreto import Col
# from .np_util import np_validate_dtype#, to_xywh
from .imageio import increment_name
from .. import config

# pylint: disable = no-member

# pylint: disable = no-member
def show_tensor(x: Union[np.ndarray, torch.Tensor],
                targets: Union[None, np.ndarray, torch.Tensor, list, tuple],
                target_mode: Union[str, list, tuple] = 'xywh',
                figsize: Optional[tuple] = (10,10),
                subplot: Optional[tuple] = None,
                show: bool = True,
                save: Optional[str] = None,
                unfold_channels: bool = False,
                **kwargs) -> None:
    """ Shows tensor or ndarray with 2d annotation targets
    Args
        x           batch tensor NCHW, managed targets and channels only if x is Tensor
        targets     list, or tensor of 2d annotations
        target_mode str, list of modes  in config.BoxMode.__members__
        figsize     tuple [(10,10)], optional
        subplot     tuple [None] - can be passed to build a grid
        show        bool [True] - set to False to build a grid
        save        str [None] if passed saves image
        unfold_chanels  bool [False] grids channels
    **kwargs
        ncols   int [None], for gridding batch N, closest to square if None
        pad     int [1] for gridding batch N
        background  float [1.] pad color
        suptitle    str
        title       str
        adjust      dict, matplotlib adjust params
        ticks       unfinished, 

    """

    if isinstance(x, torch.Tensor):
        ncols = None if 'ncols' not in kwargs else kwargs['ncols']
        pad = 1 if 'pad' not in kwargs else kwargs['pad']
        background = 1. if 'background' not in kwargs else kwargs['background']
        x, targets = to_numpy_grid(x, targets, ncols, pad=pad, background=background,
                                  mode=target_mode, unfold_channels=unfold_channels)

    if figsize is not None:
        plt.figure(figsize=figsize)

        if 'suptitle' in kwargs:
            supfontsize = 1.5 if "supfontsize" not in kwargs else kwargs["supfontsize"]
            plt.suptitle(kwargs['suptitle'], fontsize=matplotlib.rcParams["font.size"]*supfontsize)

    if subplot is not None:
        plt.subplot(*subplot)

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    if targets is not None:
        colors = ['yellow', 'red', 'orange', 'black', 'gray', 'white', 'blue', 'green']
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(target_mode, (list, tuple)):
            target_mode = [target_mode]*len(targets)
        # print("\ndrawing targets")
        for i, target in enumerate(targets):
            # print(f" target {i}, type {type(target)}, mode {target_mode[i]}")
            draw_boxes(target, target_mode[i], color=colors[i%len(colors)])#, annot, labels, color, lwidth)

    plt.imshow(x, cmap='gray' if 'cmap' not in kwargs else kwargs['cmap'])

    if "ticks" in kwargs:
        plt.grid(kwargs['ticks'])

    if show:
        plt.tight_layout()
        if "adjust" in kwargs:
            plt.subplots_adjust(**kwargs["adjust"])

        if isinstance(save, str):
            savefig(save, **kwargs)

        plt.show()


def showhist(data, width=10):
    # pylint: disable=no-member
    plt.figure(figsize=(width, width/2))
    for _d in data:
        plt.plot(torch.histc(_d, bins=255).clone().detach().cpu().numpy())
    plt.grid()
    plt.show()


def to_numpy_grid(x: torch.Tensor,
                  targets: Union[None, list, tuple, torch.Tensor] = None,
                  ncols: Optional[int] = None,
                  pad: int = 1,
                  background: float = 1.,
                  mode: Union[str, list, tuple] = "xywh",
                  unfold_channels: bool = False) -> tuple:
    """ Convert tensor with position annotations to numpy grid
    """
    x = x.to("cpu").clone().detach()
    n, c, h, w = x.shape
    m = n
 
    # more than 3 channels
    if (c not in (1, 3) and c%3) or unfold_channels:
        x = x.view(n*c, 1, h, w)
        m = n*c
        if ncols is None and n == 1:
            ncols = 1 if w > h else 3
    elif c not in (1,3):
        x = torch.cat(x.split(c//3, dim=1))
        m = n*c//3

    x = x.permute(0,2,3,1).contiguous().numpy()
    if m == 1:
        return x.squeeze(0), squeeze_target(targets, mode, True)

    if ncols in (0, None):
        nrows, ncols = closest_square(m)
    else:
        nrows = m//ncols + int(bool(m%ncols))

    nrows = m//ncols + int(bool(m%ncols))
    img = np.ones((nrows*(h + pad) - pad, ncols*(w + pad) - pad, c)).astype(x.dtype) * background

    out_targets = []
    if isinstance(mode, str):
        mode = [mode]*len(x)
    for i, _x in enumerate(x):
        posx = i%ncols*(w + pad)
        posy = i//ncols*(h + pad)
        img[posy:posy+h, posx:posx+w, :] = _x
        j = i//(m//n)

        # TODO clean up / test
        if isinstance(targets, (list, tuple, torch.Tensor)) and len(targets) > i:

            translate = torch.tensor([posy,posx])
            _mode = mode[min(i, len(mode)-1)]
            out_targets += [squeeze_target(targets[i], _mode, False)]
            out_targets[i] = translate_target(out_targets[i], translate, mode=_mode, to_numpy=True)

    return img, out_targets


def draw_boxes(boxes, mode, annot=None, labels=None, color='yellow', lwidth=1):
    """ draw boxes on matplotlib axis28
    Args
        boxes   tensor
        mode    in BoxMode
    """
    if isinstance(mode, str):
        mode = [mode]*len(boxes)
    for i, box in enumerate(boxes):
        if len(box) == 0:
            continue
            
        _draw_box_of_mode(box, mode[i], color, lwidth)

def _draw_box_of_mode(box: np.ndarray,
                      mode: Union[str, config.BoxMode],
                      color: str = "yellow",
                      lwidth: float = 1.) -> np.ndarray:
    target = None

    mode = mode if isinstance(mode, str) else mode.name
    assert mode in config.BoxMode.__members__, f"mode {mode} not defined in BoxMode {config.BoxMode.__members__}"

    if (mode in ('xywha', 'yxhwa') and box.ndim > 1) or box.ndim > 2:
        for b in box:
            _draw_box_of_mode(b, mode, color, lwidth)

    else:
        if mode in ('yxhw', 'yxyx'):
            box[0] = box[0][::-1]
            box[1] = box[1][::-1]
            mode = mode[:2][::-1] + mode[2:][::-1]

        if mode == 'xyxy':
            box[1] -= box[0]
            mode = 'xywh'

        if mode == 'xywh':
            # Rectangle(xy, width, height, angle=0.0, **kwargs)
            target = Rectangle(box[0], *box[1], edgecolor=color, linewidth=lwidth, facecolor='none')  
        elif mode in ('xywha', 'yxhwa'):
            target = Ellipse(box[:2], box[2]*2, box[3]*2, angle=box[4]*-180/np.pi, edgecolor=color,
                            linewidth=lwidth, facecolor='none')
        elif mode in ('ypath', 'xpath'):
            target = Polygon(box, closed=True, edgecolor=color, linewidth=lwidth, facecolor='none')
        else:
            raise NotImplementedError(f"Mode {mode} not recognized")

        if target is not None:
            plt.gca().add_patch(target)

def closest_square(number):
    side = int(number**(1/2))
    if side**2 == number:
        return side, side
    if side * (side + 1) >= number:
        return side, side + 1
    return side + 1, side + 1

def squeeze_target(target, mode="xywh", to_numpy=True):
    """ squeeze target N,mindims
    """
    if isinstance(target, (tuple, list)) and len(target):
        if not isinstance(mode, (list, tuple)):
            mode = [mode]*len(target)
        for i in range(len(target)):
            target[i] = squeeze_target(target[i], mode=mode[i], to_numpy=to_numpy)

    elif torch.is_tensor(target):
        if isinstance(mode, (list, tuple)):
            mode = mode[0]
        # print("  squeezing target in mode", mode)
        target = target.clone().detach().view(-1, *target.shape[-2:])
        # target = target.clone().detach().view(-1, *target.shape[-(1 if 'path' in mode else 2):])
        if to_numpy:
            target = target.numpy()
    elif not isinstance(target, np.ndarray):
        target = []
    return target

def translate_target(target, translation, mode="xywh", sign=1, to_numpy=False):
    """adds or subtracts from target
    """
    
    if isinstance(target, (list, tuple)):
        mode = mode if isinstance(mode, (list, tuple)) else [mode]*len(target)
        out  = []
        for i in range(len(target)):
            out.append(translate_target(target[i], translation, mode[i], sign, to_numpy))
        return out

    if torch.is_tensor(target):
        # print("  translating", target.tolist(), "by", translation, "mode", mode)

        if mode[0] == 'x':
            translation = translation.flip(-1)
        translation = translation.mul(sign)

        out = target.clone().detach()

        if mode in ('xywh', 'yxhw'):
            translation = torch.stack((translation, torch.zeros(2)))
        elif mode in ('xywha', 'yxhwa'):
            translation = torch.cat((translation, torch.zeros(3)))

        out = out.add(translation)
        if to_numpy:
            out = out.numpy()
        return out

    elif not isinstance(target, np.ndarray):
        return []

    # print("  returning, unchanged")
    return target

def savefig(path, **kwargs):

    _kk = ["transparent", "frameon", "dpi", "facecolor", "edgecolor",
           "orientation", "pad_inches"]
    _kwargs = {k:kwargs[k] for k in _kk if k in kwargs}

    if "dpi" not in _kwargs:
        _kwargs["dpi"] = 300

    _, _ext = osp.splitext(path)
    if _ext.lower() not in ('.png', '.jpg', '.pdf'):
        path += ".png"

    if osp.isfile(path) and "increment" in kwargs:
        path = increment_name(path)
    print("saving image to <%s>"%path)
    plt.savefig(path, **_kwargs)



def _crop(img, boxes, crop, mode):
    if len(crop) < 4:
        print("need 4 values to crop, found %s"%str(crop))
    else:
        _x0, _y0, _x1, _y1 = crop
        if _x1 <= _x0 or _y1 <= _y0: # yhxw
            _x1 += _x0
            _y1 += _y0
        img = img[_y0:_y1, _x0:_x1]
        if boxes is not None:
            for i, _ in enumerate(boxes):
                if mode == config.BoxMode.xywh:
                    boxes[i][0, 0] -= _x0
                    boxes[i][0, 1] -= _y0
                elif mode in (config.BoxMode.yxhwa, config.BoxMode.xywha):
                    boxes[i][0] -= _x0
                    boxes[i][1] -= _y0
                elif mode in (config.BoxMode.ypath, config.BoxMode.xpath):
                    for j in range(len(boxes[i])):
                        boxes[i][j][0] -= _x0
                        boxes[i][j][1] -= _y0

                else:
                    assert False, "config mode not recognized %s"%str(mode)
    return img, boxes



def tickle(ax, kwargs, i=None):
    _sticks = False
    if "xticks" in kwargs:
        if not kwargs["xticks"]:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            set_xticks(ax, kwargs["xticks"], i)
        _sticks = True

    if "yticks" in kwargs:
        if not kwargs["yticks"]:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            set_yticks(ax, kwargs["yticks"], i)
        _sticks = True

    if not _sticks:
        if "ticks" in kwargs and not kwargs["ticks"]:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_major_formatter(plt.NullFormatter())

        if "noframe" in kwargs:
            ax.axis('off')
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])

def set_xticks(ax, xticks, i=None):

    if i is not None:
        xticks = xticks[i]

    if isinstance(xticks, (tuple, list, np.ndarray, torch.Tensor)):
        if isinstance(xticks[0], (tuple, list, np.ndarray, torch.Tensor)):
            _xticks = xticks[0]
            _xticklabels = xticks[1]
        else:
            _xticks = xticks
            _xticklabels = xticks
        ax.set_xticks(_xticks)
        ax.set_xticklabels(_xticklabels)
    elif not xticks:
        ax.set_xticks([])
        ax.set_xticklabels([])

def set_yticks(ax, yticks, i=None):

    if i is not None:
        yticks = yticks[i]
    if isinstance(yticks, (tuple, list, np.ndarray, torch.Tensor)):
        if isinstance(yticks[0], (tuple, list, np.ndarray, torch.Tensor)):
            _yticks = yticks[0]
            _yticklabels = yticks[1]
        else:
            _yticks = yticks
            _yticklabels = yticks
        ax.set_yticks(_yticks)
        ax.set_yticklabels(_yticklabels)
    elif not yticks:
        ax.set_yticks([])
        ax.set_yticklabels([])
