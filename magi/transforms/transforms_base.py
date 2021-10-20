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

    def __repr__(self, exclude_keys: Union[list, tuple]=None) -> str:
        """ utility, auto __repr__()
        Args
            exclude_keys    (list, tuple [None])
        """
        rep = self.__class__.__name__+"("
        for i, (key, value) in enumerate(self.__dict__.items()):
            if exclude_keys is not None and key in exclude_keys:
                continue
            value = value if not isinstance(value, str)  else f"'{value}'"
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
        return out, unused

class TransformAppearance(Transform):
    """ class of Transforms that do not change the size of tensor

    To make appearance transforms differentiable set
        config.INPLACE or self.inplace == True

    Args:
        inplace     (bool [config.INPLACE])
    #super().__init__(inplace=None)
    """
    __type__ = "Appearance"
    def __init__(self, inplace: bool=None) -> None:
        self.inplace = inplace if inplace is not None else config.INPLACE
