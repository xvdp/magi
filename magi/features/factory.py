"""@xvdp

"""
from .features import Item
from ..utils import is_torch_strdtype


### ItemFactory
class TypedItemBase:
    """ Item Provider
    dict elements only allowed to be str
    """
    def __init__(self, **kwargs):
        lens = [len(val) for key, val in kwargs.items() if key[0]!="_"]
        if lens:
            assert all([l == lens[0] for l in lens]), f"all keys must have same len, found {list(zip(kwargs.keys(), lens))}"
            for key, val in kwargs.items():
                assert all([isinstance(s, str) for s in val]), f"{key}=<> expected <str> list, found, {[type(s) for s in val]}"

        self.__dict__.update(**kwargs)

    def get(self, data: list=None) -> Item:
        _len = 0
        if self.__dict__:
            keys = [key for key in self.__dict__ if key[0] != "_"]
            _len = len(self.__dict__[keys[0]])
        if data is None:
            data = [None]*_len
        assert len(data) == _len or _len == 0, f"expected {_len} items, got {len(data)}"
        return Item(data, **self.__dict__)

class TypedItem(TypedItemBase):
    """ Item() maker that requires keys 'names', 'meta', 'dtype'
    with list of strings
        names   (list[str]) - any string
        meta    (list[str]) - in 'data_<1,2,3>d', 'positions_<1,2,3>d', 'name', 'path', 'id'
            informs agumentation how to handle item
        .. modifications to meta need to have augmentation handlers registered

        dtype   (list[str]) - torch dtype as str or 'str'
            informs Item().to_torch() how to convert
    """
    def __init__(self, names: list, meta: list, dtype: list):
        """
        """
        super().__init__(names=names, meta=meta, dtype=dtype)
        # allowed dtypes
        for _dt in dtype:
            assert is_torch_strdtype(_dt) or _dt == "str", f"'str' or dtype str, found {_dt}"

        # allowed data types: meta
        _dims = [1, 2, 3]
        _meta = [f'data_{d}d' for d in _dims]
        _meta += [f'postions_{d}d' for d in _dims]
        _meta += ['name', 'path', 'id']
        for _mt in meta:
            assert _mt in _meta, f"meta '{_mt}' not implemented expected {_meta}"
