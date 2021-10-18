"""@xvdp
Features:

Item is intentded as compromise between:
    simplicity of untagged lists to data items commonly used in torch datasets
    precision of dict features on tensorflow
e.g. ref. https://github.com/tensorflow/datasets/tree/v4.4.0/tensorflow_datasets/core/features

dataset.__getitem__() can output a plain ListDict without tags as canonical ImageNet e.g.
    [image_tensor, class_id]
        ot iy can output any size list, or tag each element, eg.
    [image_tensor, image_id, class_id].{"tags":["image", "image_id", "class_id]}

ListDict() has no requirements, at its simplest is a list, or a list with dictionary, 

Item is a ListDict with keys which contents are lists  which if filled by datasets can inform dataloader collate_fn
    collate_fn, see .dataloader
    .tags()

"""
from typing import Any, TypeVar, Iterable, Union
from inspect import getmembers
from copy import deepcopy
import numpy as np
import torch
from ..utils import torch_dtype
from .list_util import is_int_iterable, list_flatten, list_modulo

_T = TypeVar('_T')

# pylint: disable=no-member
class ListDict(list):
    """ List with Dict

    Example
    >>> item = ListDic([torch.randn([1,3,45,45]), torch.tensor([[0,20],[0,20.]]), 1], tags=["image", "positions", "index"])

    >>> print(item[0].shape, item.tags[0])
    # [*] (1,3,45,45), "image"

    # supports assigning new keys
    >>> item.new_key = "micky mouse is dead he got kicked in the head"


    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.__dict__.update(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def __types__(self) -> list:
        """ return a list of types for the DictLIST
        """
        return [type(item) for item in self]

    def copy(self) -> Any:
        """
        Return a shallow copy of the ListDict
        """
        return ListDict(super().copy(), **self.__dict__.copy())

    def deepcopy(self) -> Any:
        return ListDict(super().copy(), **self.__dict__.copy())

    def clear(self) -> None:
        super().clear()
        self.__dict__.clear()

    def index(self, value, start=0, stop=9223372036854775807) -> int:
        """Return first index of value.
        Fix to list to allow indexing of torch and numpy elements
        Raises ValueError if the value is not present.
        """
        _is_tensor = isinstance(value, torch.Tensor)
        _is_ndarray = isinstance(value, np.ndarray)

        for i in range(start, min(len(self), stop)):
            if torch.is_tensor(self[i]) and _is_tensor and torch.equal(value, self[i]):
                return i
            if isinstance(self[i], np.ndarray) and _is_ndarray and np.all(np.equal(self[i], value)):
                return i
            elif value == self[i]:
                return i
        raise ValueError(f"{value} is not in list")

class Item(ListDict):
    """ Generic Feature Datastructure.
    Keeps data as list, addresses in a dict

    Stricter ListDict() requiring that public keys contain lists of same length as main list
        private keys may contain any data

    Does not enforce typing or require specific type keys.
    By convention however keys .meta and .dtype will be interpreted by magi augments and datasets
        .meta: data_2d (tbd ..._1d, _3d) / transforms treated as images
        .meta: positions_2d (tbd ..._1d, _3d) /transforms treated as positions.
        .dtype: if present specific dtypes will be applied to list elements on handling

        self -> list
        self.__dict__ -> dict
            self.<key> -> list(of len(self))  # public keys, logged with property self._keys
            self._<key> -> Any                # private keys, logged with property, self.keys

        public keys (self.<key>)
            self.keys -> self.__dict__.keys(), key[0] != "_"
            self.__dict__[key] -> list
            # enforcing len(self.__dict__[key]) == self.__len__()

        private keys (self._<key>)
            self.private_keys ->  self.__dict__.keys(), key[0] == "_"
            # length property self._<key> not enforced

    Item methods:
        .get(key, value) -> sub list of self items tagged with (key, value) pairs
        .to_torch(dtype, device, grad, [include=[]], [exclude=[]])
        .to(**kwargs) runs tensor.to(**kwargs) for all kwargs in data
        .keep(**kwargs) # deletes all entries not in key values # keep(None) keeps everything
        .keep(*args)    # deletes all in indices
        .deepcopy() | .deepclone()   # data is deepcopied, tensors are cloned and detached

    ListDict methods:
        .clear()

    list methods extended to handle keys:
        .append(value, key=keyval, ...)
        .insert(value, key=keyval, ...)
        .extend(values, key=keyvalues, ...)
        .remove(value)
        .pop(index)
        .reverse()
        .index()
        .copy() | .clone() # data is copied tensors are cloned

    list methods removed
        .sort() # instances in Item are not of same type generally.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Args
            *args       -> list(*args)
            **kwargs    -> {k0:list, k1:list, ...}
        Examples
            >>> d = Item([torch.randn([1,3,224,224]), 2, 1], tags=['image', 'image_id', 'class_id'])
            >>> d.keys  # -> ['tags']
            >>> d       # -> [torch.Tensor[....], 2, 1]

            # add new key, filling every element of list with some info
            >>> d.meta  # ['image.source...', 'id in dataset', 'class in wordnet']
            >>> d.keys  # -> ['tags', 'meta']

            # append new datum, assigning a value to every key
            >>> d.append([[0,100,200,200]], tags='position', meta='made up foo')
            >>> d       # -> [torch.Tensor[....], 2, 1, [[0,100,200,200]]]
            >>> d.tags  # -> ['image', 'image_id', 'class_id', 'position']
            >>> d.meta  # -> ['image.source...', 'id in dataset', 'class in wordnet', 'made up foo']

            # access elements with key, value:
            >>> d.get('tags', 'position') -> [ [[0,100,200,200]] ]
            >>> d.get('meta', 'made up foo') -> [ [[0,100,200,200]] ]

            # clone (item.copy() and item.clone(()) or deepclone (deepcopy(item) and item.clone().detach())
            >>> e = d.deepclone()
            >>> f = d.clone()

            # remove all elements except selected in keep, by index
            >>> e.keep(0,1) # or e.keep([0,1])
            >>> e       # -> [torch.Tensor[....], 2]      # list removed indices not in 0,1
            >>> e.tags  # -> ['image', 'image_id']   # idem for all keys

            # remove all elements except selected in keep, by keyvalue
            >>> f.keep(meta=['class in wordnet', 'made up foo']) ->
            >>> f       # -> [1, [[0,100,200,200]] ]

            # convert to torch
            >>> d.to_torch(dtype="float64", device="cuda") # dtype can be list or included as a key

            # remove keys, cast as list
            >>> list(d)
        """
        super().__init__(*args) # list
        self.__dict__.update(**kwargs) # dict, {key: list (len: args), ...}

        _larg = 0 if not args else len(*args)
        _slen = len(self)
        for key in self.keys:
            _len = len(self.__dict__[key])
            assert _len == _larg == _slen, f"len(Item.{key}:{_len}) should equal len(Item){_slen} and input args {_larg}"

    def __setattr__(self, name: str, value: list) -> None:
        self._assert_read_only(name)
        if name[0] != "_":
            assert isinstance(value, list) and len(self) == len(value), f"keyed list needs to match data(len), expected: {len(self)}, got {len(value)}"
        return super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        return super().__delattr__(name)

    def __delitem__(self, i: Union[int, slice]) -> None:
        """ Delete self[key], if self[key] is torch cuda, empty cache
        """
        _is_torch_cuda = isinstance(self[i], torch.Tensor) and self[i].device != "cpu"
        super().__delitem__(i)
        for key in self.keys:
            self.__dict__[key].__delitem__(i)
        if _is_torch_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _assert_read_only(self, name):
        if name in [m[0] for m in getmembers(self)]:
            raise AttributeError(f"'ListDict' object attribute '{name}' is read-only")

    @property
    def keys(self) -> list:
        """ Returns all public keys as a list
        """
        return [key for key in self.__dict__  if key[0] != "_"]

    @property
    def _keys(self) -> list:
        """ Returns all private keys as a list
        """
        return [key for key in self.__dict__  if key[0] == "_"]

    def get_indices(self, key: str, val: str) -> list:
        """ Returns indices of elements tagged by (key, value) pairs
        """
        assert key in self.keys, f"key '{key}' not found in {self.keys}"
        return [i for i in range(len(self.__dict__[key])) if self.__dict__[key][i] == val]

    def get(self, key: str, val: str) -> list:
        """ Returns sublist of elements tagged by (key, value) pairs
        Example
        if self has a 'tags' key, and self.tags has 'name' entries,
        >>> self.get("tags", "name") -> [items with .tags == 'name']
        """
        return [self[i] for i in self.get_indices(key, val)]

    def to_torch(self, dtype: Union[torch.dtype, list, tuple]=None, device: Union[torch.device, str]="cpu",
                 grad: bool=False, include: Union[list, tuple]=None, exclude: Union[list, tuple]=None, **kwargs):
        """ converts all numeric list items to torch except when dtype=[...] is passed or dtype key with value
        not in torch.__dict__, that does not translate to torch.dtype
        Args
            dtype       (str, torch.dtype, list [None]) | .__dict__['dtype'] if 'dtype' in .__dict__
            device      (str, torch.device ['cpu'])
            grad        [False]

            include   list of indices to convert to torch, if None, all
            exclude   list of indices to keep not as torch
            **kwargs    any valid for torch.as_tensor(**kwargs)
        """
        exclude = exclude if exclude is not None else []
        # fix dtype
        # dtype assigned in Item key "dtype"
        if dtype is None and "dtype" in self.__dict__:
            dtype = self.__dict__['dtype']
            exclude += [i for i in range(len(dtype)) if dtype[i] not in torch.__dict__]
        # convert to list
        elif dtype is None or isinstance(dtype, (str, torch.dtype)):
            dtype = [dtype for i in range(len(self))]

        # ensure dtype is valid torch.dtype, default to None: take no action
        dtype = torch_dtype(dtype, fault_tolerant=True)

        # filter indices
        include = sorted(list_modulo(include, len(self))) if include is not None else range(len(self))
        include = [i for i in include if i not in list_modulo(exclude, len(self))]

        for i in include:
            if isinstance(self[i], (int, float)):
                self[i] = [self[i]]
            try:
                _item = torch.as_tensor(self[i], dtype=dtype[i], device=device)
                self[i] = _item
                if self[i].requires_grad != grad:
                    self[i].requires_grad = grad
            except:
                pass

    def to(self, **kwargs):
        """ wrapper to torch.to() for all tensor elements
            (dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
            (device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
            (other, non_blocking=False, copy=False)
        https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
        """
        for i, _ in enumerate(self):
            if isinstance(self[i], torch.Tensor):
                self[i] = self[i].to(**kwargs)

    def keep(self, *args, **kwargs) -> None:
        """ Removes all public keys and items not tagged to keep
        overloads: indices or key,value pairs
            .keep(<key>=[<value>]) #<key> in self.keys
            .keep([list of indices])
        .. warning - can be slow if Item has large tensors
        """
        if args and args[0] is not None and is_int_iterable(*args):
            self._keep_by_index(*args)

        elif kwargs:
            if "index" in kwargs and is_int_iterable(kwargs["index"]):
                self._keep_by_index(kwargs["index"])
            else:
                self._keep_by_keyvals(**kwargs)

    # @overload # not available outside stub files
    def _keep_by_index(self, *args) -> None:
        """ """
        indices = list_flatten(args, unique=True)
        indices = list_modulo(indices, len(self))
        _remove = [i for i in range(len(self)) if i not in indices]
        _remove.reverse()
        for i in _remove:
            del self[i]

    # @overload
    def _keep_by_keyvals(self, **kwargs) -> None:
        """ """
        _kwargs = {key:kwargs[key] for key in kwargs if key in self.keys}
        index = []
        for key in _kwargs:
            if not isinstance(_kwargs[key], (list, tuple)):
                _kwargs[key] = [_kwargs[key]]
            for val in _kwargs[key]:
                index.extend([i for i in range(len(self.__dict__[key]))
                              if self.__dict__[key][i] == val])
        if index:
            self._keep_by_index(index)

    def _assert_values(self, **kwargs) -> None:
        """ Item imposes no type on key values
        this ought to be implmenented with a metaclass
        """
        pass

    def _assert_keys(self, msg, **kwargs) -> None:
        _nukeys = [key for key in kwargs if key not in self.keys]
        _miskeys = [key for key in self.keys if key not in kwargs]

        if _nukeys or _miskeys:
            _msg = ""
            if _miskeys:
                _msg = f"missing keys {_miskeys} required; "
            if _nukeys:
                _msg+= f"cannot add new keys {_nukeys} on .{msg}(), declare as self.<new_key>=<list of len(self)>"
            assert not _miskeys and not _nukeys, _msg
        self._assert_values(**kwargs)

    def append(self, obj: _T, **kwargs) -> None:
        """Appends to self and to lists for each public key, requires entries for each key
        Example
         if self.keys == ["tags", "info"]
        >>> self.append(1, tags="one", info="the first number")
        """
        self._assert_keys("append", **kwargs)
        super().append(obj)
        for key, val in kwargs.items():
            self.__dict__[key].append(val)

    def copy(self) -> Any:
        """Returns a copy of Item
        : seme as deepcopy or deepclone but does not detach tensors
        """
        outlist = []
        for i, item in enumerate(self):
            if isinstance(item, torch.Tensor):
                outlist.append(item.clone())
            else:
                outlist.append(deepcopy(item))
        return Item(outlist, **deepcopy(self.__dict__))
    clone = copy

    def deepcopy(self) -> Any:
        """ Returns a deep copy of Item
        tensors, are clone().detached()
        """
        outlist = []
        for i, item in enumerate(self):
            if isinstance(item, torch.Tensor):
                outlist.append(item.clone().detach())
            else:
                outlist.append(deepcopy(item))
        return Item(outlist, **deepcopy(self.__dict__))
    deepclone = deepcopy

    def extend(self, iterable: Iterable[_T], **kwargs) -> None:
        """ Extend to self and to lists for each public key, requires entries for each key
        Example
        if self.keys == ["tags", "info"]
        >>> self.extend([1, 3.14], tags=["one", "pi"], info=["int", "float"])
        """
        self._assert_keys("extend", **kwargs)
        super().extend(iterable)
        for key, val in kwargs.items():
            self.__dict__[key].extend(val)

    def insert(self, index: int, obj: int, **kwargs) -> None:
        """ Inserts into self and into lists for each public key, requires entries for each key
        """
        self._assert_keys("insert", **kwargs)
        super().insert(index, obj)
        for key, val in kwargs.items():
            self.__dict__[key].insert(index, val)

    def pop(self, index: int) -> _T:
        """Remove and return Item at index (default last).
        Raises IndexError if list is empty or index is out of range.
        """
        dict_component = {key:[self.__dict__[key].pop(index)] for key in self.keys}
        return Item([super().pop(index)], **dict_component)

    def remove(self, obj: _T) -> None:
        index = self.index(obj)
        self.__delitem__(index)

    def reverse(self) -> None:
        super().reverse()
        for key in self.keys:
            self.__dict__[key].reverse()

    def sort(self) -> None:
        """ Not Implemented"""
        raise NotImplementedError("Item() is not sortable")

    def new_like(self, data:list):
        """ Creates new Item with same dict as current"""
        if len(self.keys):
            assert len(data) == len(self.__dict__[self.keys[0]])
        return Item(data, **self.__dict__)

def tolist(item: Any) -> list:
    """ converts, casts or encapsulates items in lists
    """
    if isinstance(item, list):
        return item
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item.tolist()
    if isinstance(item, (tuple, Item)):
        return list(item)
    return [item]


"""
example features from tensorflow: wider

FeaturesDict({
    'faces': Sequence({
        'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
        'blur': tf.uint8,
        'expression': tf.bool,
        'illumination': tf.bool,
        'invalid': tf.bool,
        'occlusion': tf.uint8,
        'pose': tf.bool,
    }),
    'image': Image(shape=(None, None, 3), dtype=tf.uint8),
    'image/filename': Text(shape=(), dtype=tf.string),
})


"""