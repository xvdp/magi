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
import torch
from .. import config
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
