"""@xvdp"""
import logging
import numpy as np
from koreto import Col

def warn_np_dtypes(dtype: str) -> str:
    """ torch tensors work onluy with floating_point types [float16, float32, float64]
        numpy arrays with ["uint8", "float32", "float64"]
    """
    _valid = ["uint8", "float16", "float32", "float64"]
    dtype = "float32" if dtype is None else dtype
    if dtype not in _valid:
        logging.warning(f"{Col.YB}numpy cannot handle {dtype}, use float32 instead {Col.AU}")
        dtype = "float32"
    return dtype

def np_validate_dtype(data, allowed=None):
    """converts to allowable types """
    if isinstance(data, (tuple, list)):
        for i, _d in enumerate(data):
            data[i] = np_validate_dtype(_d)
        return data

    assert isinstance(data, np.ndarray), "expected ndarray, found, '%s'"%type(data)
    allowed = [np.float32, np.float64] if allowed is None else allowed

    if data.dtype not in allowed:
        data = data.astype(allowed[0])
    return data