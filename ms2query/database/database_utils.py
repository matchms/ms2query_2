import io
from typing import Optional, Union
import numpy as np


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



# =========================
# Fingerprint helpers
# =========================

def encode_sparse_fp(bits: Optional[np.ndarray], counts: Optional[np.ndarray]) -> tuple[bytes, bytes]:
    """Store bits as uint32 indices, counts as int32

    Parameters
    ----------
    bits : array-like of uint32 bit indices
    counts : array-like of int32 counts

    Returns (bits_blob, counts_blob). Accepts None -> empty blobs."""
    if bits is None:
        b = b""
    else:
        arr = np.asarray(bits)
        if arr.dtype != np.uint32:
            arr = arr.astype(np.uint32, copy=False)
        b = arr.tobytes(order="C")
    if counts is None:
        c = b""
    else:
        arrc = np.asarray(counts)
        if arrc.dtype != np.int32 and arrc.dtype != np.uint32 and arrc.dtype != np.uint16 and arrc.dtype != np.uint8:
            arrc = arrc.astype(np.int32, copy=False)
        c = arrc.tobytes(order="C")
    return b, c


def decode_sparse_fp(bits_blob: bytes, counts_blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_sparse_fp.

    Parameters
    ----------
    bits_blob : BLOB bytes of uint32 bit indices
    counts_blob : BLOB bytes of int32 counts

    Returns (bits_uint32, counts_int32). Empty blobs -> empty arrays.
    """
    bits = np.frombuffer(bits_blob, dtype=np.uint32).copy() if bits_blob else np.zeros(0, dtype=np.uint32)
    # Guess signedness: store as int32 by default
    counts = np.frombuffer(counts_blob, dtype=np.int32).copy() if counts_blob else np.zeros(0, dtype=np.int32)
    return bits, counts


def encode_dense_fp(vec: Optional[np.ndarray]) -> bytes:
    """Encode a dense vector as float32 bytes. None -> empty blob."""
    if vec is None:
        return b""
    arr = np.asarray(vec)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr.ravel().tobytes(order="C")


def decode_dense_fp(blob: bytes, dtype: str = "float32") -> np.ndarray:
    """Decode dense vector from blob with the given dtype (default float32)."""
    if not blob:
        return np.zeros(0, dtype=np.float32 if dtype == "float32" else np.dtype(dtype))
    return np.frombuffer(blob, dtype=np.dtype(dtype)).copy()
