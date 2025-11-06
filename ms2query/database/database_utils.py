import io
import numpy as np


# ------------------------ helper functions ----------------------

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array (with dtype/shape) into bytes for SQLite BLOB."""
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        return f.getvalue()


def blob_to_array(b: bytes, dtype, copy=True) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=dtype)
    if copy:
        return np.frombuffer(b, dtype=dtype).copy()
    return np.frombuffer(b, dtype=dtype)
