# tests/test_compounds_and_mapping.py
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ms2query.data_processing import compute_morgan_fingerprints
from ms2query.database import CompoundDatabase
from ms2query.database.spec_to_compound_mapper import (
    SpecToCompoundMap,
    get_unique_compounds_from_spectraldb,
    map_from_spectraldb_metadata,
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
IK_FULL_1 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # glucose
IK_FULL_2 = "BSYNRYMUTXBXSQ-UHFFFAOYSA-O"  # same first14, different suffix
IK_FULL_3 = "BQJCRHHNABKAKU-KBQPJGBKSA-N"  # ethanol
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
# Tests: SpecToCompoundMap + integration
# -------------------------

def test_mapping_and_compound_creation(tmp_path):
    db_path = make_tmp_db(tmp_path)

    # Create minimal spectra table (3 rows; one is NULL inchikey)
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_2, None])

    # Run mapping (same db hosts compounds + mapping)
    n_mapped, n_new = map_from_spectraldb_metadata(db_path)
    assert n_mapped == 2                      # two spectra had inchikeys
    assert n_new in (1, 2)                    # at least one unique comp

    # Validate mapping contents
    mapper = SpecToCompoundMap(db_path)
    df_map = mapper.get_comp_id_for_specs([1, 2, 3])
    assert set(df_map.columns) == {"spec_id", "comp_id"}
    # spec_id 3 has no inchikey -> may be missing
    assert set(df_map["spec_id"]) <= {1, 2, 3}
    # comp_ids are 14 chars
    assert all(len(c) == 14 for c in df_map["comp_id"])
    mapper.close()

    # Validate compounds exist (metadata only, no FPs yet)
    cdb = CompoundDatabase(db_path)
    dfc = cdb.sql_query("SELECT comp_id, inchikey FROM compounds")
    assert not dfc.empty
    assert all(len(cid) == 14 for cid in dfc["comp_id"])
    # FPs should be empty prior to backfill
    empty = cdb.sql_query("SELECT COUNT(*) AS n FROM compounds WHERE COALESCE(LENGTH(fingerprint_bits),0)=0")
    assert empty["n"].iloc[0] >= 1
    cdb.close()

def test_mapper_link_and_get(tmp_path):
    db_path = make_tmp_db(tmp_path)

    # need compounds table; mapping works independently
    cdb = CompoundDatabase(db_path)
    cdb.upsert_compound(inchikey=IK_FULL_1)  # ensure a compound exists
    cdb.close()

    mapper = SpecToCompoundMap(db_path)
    mapper.link(123, IK14_1)
    mapper.link_many([(124, IK14_1), (125, IK14_1)])

    ids = mapper.get_specs_for_comp(IK14_1)
    assert set(ids) == {123, 124, 125}

    df = mapper.get_comp_id_for_specs([122, 123, 124, 125])
    assert set(df.columns) == {"spec_id", "comp_id"}
    assert set(df["spec_id"]) == {123, 124, 125}

    mapper.close()

# -------------------------
# Tests: get_unique_compounds_from_spectraldb
# -------------------------

def test_get_unique_compounds_basic(tmp_path):
    db_path = make_tmp_db(tmp_path)
    # spectra: two with same IK14, one different, one NULL
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_2, IK_FULL_3, None])

    uniq = get_unique_compounds_from_spectraldb(db_path)
    # Column order follows the current function: inchikey14, inchikey, n_spectra
    assert list(uniq.columns[:3]) == ["inchikey14", "n_spectra", "inchikey"]
    assert set(uniq["inchikey14"]) == {IK14_1, IK14_3}
    counts = dict(zip(uniq["inchikey14"], uniq["n_spectra"]))
    assert counts[IK14_1] == 2
    assert counts[IK14_3] == 1

def test_get_unique_compounds_with_external_merge(tmp_path):
    db_path = make_tmp_db(tmp_path)
    create_min_spectral_table(db_path, [IK_FULL_1, IK_FULL_3])

    external = pd.DataFrame({
        "inchikey14": [IK14_1, "NOPE0000000000"],
        "my_tag": ["hit", "miss"],
        "score": [0.9, 0.1],
    })
    uniq = get_unique_compounds_from_spectraldb(db_path, external_meta=external)
    # Should have merged columns
    assert "my_tag" in uniq.columns and "score" in uniq.columns
    # Only IK14_1 should have my_tag filled
    row = uniq.loc[uniq["inchikey14"] == IK14_1].iloc[0]
    assert row["my_tag"] == "hit"
    assert pytest.approx(row["score"]) == 0.9
