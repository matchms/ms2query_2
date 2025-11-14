import sqlite3
from pathlib import Path
import numpy as np
import pytest
from ms2query.data_processing import compute_morgan_fingerprints
from ms2query.database.compound_database import (
    CompoundDatabase,
)


# -------------------------
# Helpers
# -------------------------

def make_tmp_db(tmp_path: Path, name: str = "test.sqlite") -> str:
    p = tmp_path / name
    if p.exists():
        p.unlink()
    return str(p)

# Example InChIKeys
IK_FULL_1 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IK_FULL_2 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-O"  # same first14, different suffix
IK_FULL_3 = "BQJCRHHNABKAKU-KBQPJGBKSA-N"
IK14_1 = "BSYNRYMUTXBXSQ"
IK14_3 = "BQJCRHHNABKAKU"

# -------------------------
# Utilities
# -------------------------

def create_min_spectral_table(sqlite_path: str, rows):
    """Create a minimal spectra table (spec_id, inchikey) and insert rows."""
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS spectra(
            spec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inchikey TEXT
        );
    """)
    cur.executemany("INSERT INTO spectra(inchikey) VALUES (?)", [(r,) for r in rows])
    con.commit()
    con.close()

# -------------------------
# Tests: low-level utilities
# -------------------------

def test_compute_fingerprints_contract():
    # API now expects list input in either smiles=... or inchis=...
    smiles = ["CCO", "C1=CC=CC=C1", None]  # last one will be ignored by our call below
    # Call only with valid smiles strings
    fps = compute_morgan_fingerprints(
        smiles=[s for s in smiles if s is not None],
        inchis=None, sparse=True, count=True, radius=9, progress_bar=False)
    assert isinstance(fps, list)
    assert len(fps) == 2
    for fp in fps:
        # Optional[Tuple[np.ndarray, np.ndarray]]
        assert fp is None or (isinstance(fp, tuple) and len(fp) == 2)
        if fp is not None:
            bits, counts = fp
            assert isinstance(bits, np.ndarray) and bits.dtype == np.uint32
            assert isinstance(counts, np.ndarray)
            # counts are usually integer-like (could be float if you later scale)
            assert counts.ndim == 1

# -------------------------
# Tests: CompoundDatabase (no FP at upsert, backfill later)
# -------------------------

def test_compound_upsert_and_get_and_backfill(tmp_path):
    db_path = make_tmp_db(tmp_path)
    cdb = CompoundDatabase(db_path)

    # Upsert (no fingerprints written at this step)
    cid = cdb.upsert_compound(
        smiles="C(CO)O",
        inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        inchikey=IK_FULL_3,
        classyfire_class="Alcohols",
        classyfire_superclass="Organic compounds",
    )
    assert cid == IK14_3

    # Metadata-only getter
    meta = cdb.get_compound(cid)
    assert meta is not None
    assert set(meta.keys()) == {"comp_id","smiles","inchi","inchikey","classyfire_class","classyfire_superclass"}
    assert meta["inchikey"] == IK_FULL_3

    # No fingerprint yet
    assert cdb.get_fingerprint(cid) is None

    # Compute fingerprints for all missing (should fill this one)
    stats = cdb.compute_fingerprints_missing(batch_size=100, use_progress_bar=False)
    assert stats["attempted"] >= 1
    assert stats["updated"] >= 1

    # Now fingerprint should be present
    fp = cdb.get_fingerprint(cid)
    assert fp is not None
    bits, counts = fp
    assert bits.dtype == np.uint32
    assert counts.ndim == 1

    cdb.close()

def test_compound_upsert_many_and_batch_getters(tmp_path):
    db_path = make_tmp_db(tmp_path)
    cdb = CompoundDatabase(db_path)

    comp_ids = cdb.upsert_many([
        {"smiles": "CCO",         "inchi": None,  "inchikey": IK_FULL_1, "classyfire_class": "A"},  # ethanol (valid)
        {"smiles": "c1ccccc1",    "inchi": None,  "inchikey": IK_FULL_2, "classyfire_class": "B"},  # benzene (valid)
        {"smiles": None,          "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", 
         "inchikey": IK_FULL_3, "classyfire_class": "C"},
    ])
    assert set(comp_ids) == {IK14_1, IK14_3}

    # Batch metadata (order preserved)
    df = cdb.get_compounds([IK14_3, IK14_1, "NOPE0000000000"])
    assert list(df["comp_id"]) == [IK14_3, IK14_1]  # missing omitted
    assert set(["smiles","inchi","inchikey","classyfire_class","classyfire_superclass"]).issubset(df.columns)

    # No fingerprints yet
    fps = cdb.get_fingerprints([IK14_3, IK14_1, "NOPE0000000000"])
    assert fps[0] is None and fps[1] is None and fps[2] is None

    # Backfill (will compute for rows with smiles OR inchi)
    stats = cdb.compute_fingerprints_missing(batch_size=100, use_progress_bar=False)
    assert stats["attempted"] >= 2
    fps = cdb.get_fingerprints([IK14_3, IK14_1, "NOPE0000000000"])
    assert fps[0] is not None and fps[1] is not None and fps[2] is None

    cdb.close()

@pytest.mark.parametrize("count,sparse", [(True, True), (False, True), (True, False), (False, False)])
def test_compute_fingerprints_method(count, sparse):
    # Test the CompoundDatabase.compute_fingerprints_missing method directly
    db_path = ":memory:"
    cdb = CompoundDatabase(
        db_path,
        fingerprint_radius=3,
        fingerprint_nbits=1024,
        fingerprint_count=count,
        fingerprint_sparse=sparse,
        )

    smiles_lst = ["CCO", "c1ccccc1"]
    # Upsert compounds without fingerprints
    comp_ids = cdb.upsert_many([
        {"smiles": smiles_lst[0], "inchi": None, "inchikey": IK_FULL_1, "classyfire_class": "A"},
        {"smiles": smiles_lst[1], "inchi": None, "inchikey": IK_FULL_3, "classyfire_class": "B"},
    ])
    assert set(comp_ids) == {IK14_1, IK14_3}

    # Initially no fingerprints
    fps_initial = cdb.get_fingerprints(comp_ids)
    assert all(fp is None for fp in fps_initial)

    # Compute fingerprints for missing
    stats = cdb.compute_fingerprints_missing(batch_size=2, use_progress_bar=False)
    assert stats["attempted"] == 2
    assert stats["updated"] == 2

    # Now fingerprints should be present
    fps_after = cdb.get_fingerprints(comp_ids)
    assert all(fp is not None for fp in fps_after if fp is not None)

    assert cdb.get_fingerprint_settings()["count"] == count
    assert cdb.get_fingerprint_settings()["nbits"] == 1024
    assert cdb.get_fingerprint_settings()["radius"] == 3
    # compute fingerprints directly with class method
    fps_directly = cdb.compute_fingerprints(smiles=smiles_lst)
    assert len(fps_directly) == 2  # one None skipped!


    if sparse and count:
        assert fps_directly[0][0].shape == fps_after[0][0].shape
        assert np.allclose(fps_directly[0][0][0], fps_after[0][0][0])  # bits
        assert np.allclose(fps_directly[0][0][1], fps_after[0][0][1])  # counts
    elif sparse:
        assert np.allclose(fps_directly[0][0], fps_after[0][0])  # bits
    else:
        assert np.allclose(fps_directly[0], fps_after[0])
    cdb.close()
    