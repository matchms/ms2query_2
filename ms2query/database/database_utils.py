import io
import numpy as np
from typing import Optional, Union

_NPY_MAGIC = b"\x93NUMPY"

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    """
    Serialize a NumPy array (with dtype & shape) into a .npy payload for SQLite BLOB.
    """
    # make sure it's a regular ndarray with a stable memory layout
    arr = np.asarray(arr)
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        return f.getvalue()


def blob_to_array(b: Union[bytes, memoryview], dtype, copy: bool = True) -> np.ndarray:
    """
    Deserialize a SQLite BLOB into a NumPy array.

    Supports:
    - .npy payloads written by ndarray_to_blob (preferred; includes shape & dtype)
    - raw byte payloads (fallback), interpreted as a 1D array of 'dtype'

    Parameters
    ----------
    b : bytes | memoryview
        BLOB from SQLite.
    dtype : np.dtype or type
        Desired dtype. If the blob is .npy, we load with its native dtype
        and cast to `dtype` only if different.
    copy : bool
        If True, return a copy. If False and format allows, return a view.
    """
    if not b:
        return np.empty((0,), dtype=dtype)

    # SQLite may return memoryview; normalize to bytes for header check / np.load
    if isinstance(b, memoryview):
        b = b.tobytes()

    # Preferred path: .npy payload
    if isinstance(b, (bytes, bytearray)) and b.startswith(_NPY_MAGIC):
        arr = np.load(io.BytesIO(b), allow_pickle=False)
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype, copy=False)  # cast but don't force an extra copy
        if copy:
            arr = arr.copy()
        return arr

    # Fallback path: raw bytes -> 1D array view
    # (Only valid if you *originally* stored arr.tobytes(); no shape info here.)
    arr = np.frombuffer(b, dtype=dtype)
    return arr.copy() if copy else arr
