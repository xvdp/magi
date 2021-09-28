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
from . import functional_io as Fio
from .. import config

class Transform(object):
    """ base transforma class

    """
    def __repr__(self, exclude_keys=None):
        """ general make __repr__
        Args
            cls             (self)
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

    def update_kwargs(self, **kwargs):
        """
        updates instance attributes on__call__() over kwargs defined on __init__()
        returns unused attributes for further use, if required
        """
        out = self.__dict__.copy()
        unused = {}
        for k in kwargs:
            if k in out:
                out[k] = kwargs[k]
            else:
                unused[k] = kwargs[k]
        return out, unused

class AppearanceTransform(Transform):
    """ class of Transforms that do not change the size of tensor
    'in_place' by default unless config.INPLACE=False or
    inplace=False is passed as argument to Transform
    """
    __type__ = "Appearance"
    def __init__(self, inplace=config.INPLACE):
        self.inplace = inplace

#
# IO Transforms
#
class Open(Transform):
    """Open a filename as torch tensor

    Args:
        file_name   (string or list of strings): valid existing filename or filename list

        dtype       (str | torch.dtype [None])
            if None uses torch.get_default_dtype()
            if not None: changes torch.default_dtype

        device      (str ["cpu"]) | "cuda"
        grad        (bool [False])
        out_type:   (str ["torch"]) | 'numpy']
        transforms: (torchvision transformse) [None]

    Returns:
        Tensor, default float32, in range 0,1, default NCL, NCHW, or NCHWD depending on data type
    """
    __type__ = "IO"
    def __init__(self, dtype=None, device="cpu", grad=False, inplace=None, out_type="torch",
                 channels=None, transforms=None):

        self.dtype = config.set_dtype(dtype)
        if inplace is not None:
            config.set_inplace(inplace)

        self.device = device
        self.grad = grad

        self.out_type = out_type if out_type in ("torch", "numpy") else "torch"
        self.channels = channels
        self.transforms = transforms

    def __call__(self, file_name, **kwargs):
        """
        Args:
            file_name   (str, list),  valid file name(s)
            **kwargs    any argument from __init__, locally
        """
        args, _ = self.update_kwargs(**kwargs)

        return Fio.open_file(file_name, dtype=args["dtype"], device=args["device"],
                             grad=args["grad"], out_type=args["out_type"],
                             channels=args["channels"], transforms=args["transforms"])
