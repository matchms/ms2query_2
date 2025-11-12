import json
import os
import sqlite3
import numpy as np
import pytest
import scipy.sparse as sp
from ms2query.database.ann_vector_index import EmbeddingIndex, csr_row_from_tuple, l1_norms_csr, tuples_to_csr


def _mk_unit_vecs(*rows):
    X = np.asarray(rows, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def test_build_index_and_query_dense():
    X = _mk_unit_vecs([1,0,0], [0,1,0], [0,0,1])
    ids = ["a","b","c"]
    idx = EmbeddingIndex(dim=3)
    idx.build_index(X, ids, assume_normalized=True)
    # Query close to [1,0,0]
    q = _mk_unit_vecs([0.9, 0.1, 0.0])[0]
    res = idx.query(q, k=2)
    assert [r[0] for r in res] == ["a", "b"]
    assert res[0][1] > res[1][1]  # similarity desc

def test_build_index_normalizes_when_requested():
    X = np.array([[2.0,0,0],[0,2.0,0]], dtype=np.float32)
    ids = ["x","y"]
    idx = EmbeddingIndex(dim=3)
    idx.build_index(X, ids, assume_normalized=False)
    q = np.array([1.0,0,0], dtype=np.float32)
    out = idx.query(q, k=1, assume_normalized=False)
    assert out[0][0] == "x"
    assert 0.99 <= out[0][1] <= 1.0

def test_query_errors_and_dim_check():
    idx = EmbeddingIndex(dim=3)
    with pytest.raises(RuntimeError):
        idx.query(np.zeros(3, np.float32))
    idx.build_index(np.eye(3, dtype=np.float32), ["a","b","c"])
    with pytest.raises(ValueError, match="dim=3"):
        idx.query(np.zeros(4, np.float32))

def test_save_and_load_roundtrip_dense(tmp_path):
    X = _mk_unit_vecs([1,0,0],[0,1,0],[0,0,1])
    ids = ["a","b","c"]
    idx = EmbeddingIndex(dim=3)
    idx.build_index(X, ids)
    prefix = os.path.join(tmp_path, "emb")
    idx.save_index(prefix)

    # New instance loads back
    idx2 = EmbeddingIndex()
    idx2.load_index(prefix)

    # Query should still work
    res = idx2.query(np.array([1.0,0,0], dtype=np.float32), k=1)
    assert res[0][0] == "a"
    # meta persisted
    with open(prefix + ".meta.json") as f:
        meta = json.load(f)
    assert meta["space"] == "cosinesimil"

@pytest.mark.parametrize("batch_rows", [1, 2, 3])
def test_build_index_from_sqlite_streams_and_orders(batch_rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE embeddings(spec_id TEXT, vec BLOB, d INTEGER)")
    # Add 3 vectors of dim 3
    conn.executemany( "INSERT INTO embeddings(spec_id, vec, d) VALUES (?,?,?)",
        [
            ("id_1", np.array([1.0, 0.0, 0.0], np.float32).tobytes(), 3),
            ("id_2", np.array([1.0, 1.0, 0.0], np.float32).tobytes(), 3),
            ("id_3", np.array([0.0, 1.0, 1.0], np.float32).tobytes(), 3),
        ],
    )
    idx = EmbeddingIndex(dim=3)
    n = idx.build_index_from_sqlite(conn, embeddings_table="embeddings", batch_rows=batch_rows, l2_normalize=True)
    assert n == 3
    # Should be ordered by spec_id ascending ("id_1","id_2")
    out = idx.query(np.array([1.0, 0.0, 0.0], np.float32), k=2)
    assert [o[0] for o in out] == ["id_1", "id_2"]

def test_build_index_from_sqlite_errors():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE embeddings(spec_id TEXT, vec BLOB, d INTEGER)")
    # Mixed dimensions should error
    v1 = np.array([1.0, 0.0], np.float32).tobytes()
    v2 = np.array([0.0, 1.0, 0.0], np.float32).tobytes()
    conn.executemany(
        "INSERT INTO embeddings(spec_id, vec, d) VALUES (?,?,?)",
        [
            ("x", v1, 2),
            ("y", v2, 3),
        ],
    )
    idx = EmbeddingIndex()
    with pytest.raises(ValueError, match="Mixed dimensions"):
        idx.build_index_from_sqlite(conn, embeddings_table="embeddings")

    # Empty table should error
    conn2 = sqlite3.connect(":memory:")
    conn2.execute("CREATE TABLE embeddings(spec_id TEXT, vec BLOB, d INTEGER)")
    with pytest.raises(ValueError, match="No rows"):
        idx.build_index_from_sqlite(conn2, embeddings_table="embeddings")


def test_tuples_to_csr_basic():
    tuples = [
        (np.array([3, 1, 1], dtype=np.int32), np.array([0.5, 2.0, 3.0], dtype=np.float32)),
        (np.array([], dtype=np.int32), np.array([], dtype=np.float32)),
        (np.array([0, 4], dtype=np.int32), np.array([1.0, 1.0], dtype=np.float32)),
    ]
    csr = tuples_to_csr(tuples, dim=5)
    assert csr.shape == (3, 5)
    # Row 0 should have indices [1,3] with values [5.0,0.5] after coalescing duplicates
    r0 = csr[0].toarray().ravel()
    np.testing.assert_allclose(r0, [0.0, 5.0, 0.0, 0.5, 0.0], rtol=0, atol=1e-8)

    # Row 1 empty
    assert csr[1].nnz == 0

    # Row 2
    r2 = csr[2].toarray().ravel()
    np.testing.assert_allclose(r2, [1.0, 0, 0, 0, 1.0])

def test_tuples_to_csr_errors_when_index_out_of_bounds():
    tuples = [
        (np.array([0, 6], dtype=np.int32), np.array([1.0, 2.0], dtype=np.float32)),
    ]
    with pytest.raises(ValueError, match=">= dim"):
        tuples_to_csr(tuples, dim=5)

def test_csr_row_from_tuple_coalesces_and_validates():
    idxs = np.array([2, 2, 0], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    row = csr_row_from_tuple((idxs, vals), dim=4)
    assert row.shape == (1, 4)
    # After coalescing index 2: total 3.0
    np.testing.assert_allclose(row.toarray(), [[3.0, 0.0, 3.0, 0.0]])

    with pytest.raises(ValueError, match="Query index"):
        csr_row_from_tuple((np.array([5]), np.array([1.0], np.float32)), dim=5)

def test_l1_norms_csr():
    X = sp.csr_matrix(
        np.array([[1.0, 2.0, 0.0], [0.0, 0.5, 0.5], [3.0, 0.0, 1.0]], dtype=np.float32)
    )
    norms = l1_norms_csr(X)
    np.testing.assert_allclose(norms, [3.0, 1.0, 4.0])
    assert norms.dtype == np.float64
