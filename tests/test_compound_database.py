import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ms2query.data_processing import compute_morgan_fingerprints, inchikey14_from_full
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


def test_overwrite_metadata_from_dataframe_basic_and_mapping(tmp_path):
    db_path = tmp_path / "compounds.sqlite"
    cdb = CompoundDatabase(str(db_path))

    # Wide DF with aliases + extras; includes:
    # - valid 14-char keys via 'nchikey'
    # - one invalid key (too short) -> skipped
    # - one duplicate comp_id -> keep last
    df = pd.DataFrame({
        "nchikey": ["AAAQFGUYHFJNHI", "AABFWJDLCCDJJN", "SHORTKEY", "AABFWJDLCCDJJN"],
        "smiles":  ["S1", "S2", "S_bad", "S2_override"],
        "cf_class": ["C1", "C2", "C_bad", "C2_override"],
        "cf_superclass": ["SC1", "SC2", "SC_bad", "SC2_override"],
        "mass": [423.146, 324.126, 0.0, 999.0],  # extra column to be ignored
    })

    stats = cdb.overwrite_metadata_from_dataframe(
        df,
        column_mapper={  # map aliases -> expected names
            "comp_id": "nchikey",
            "smiles": "smiles",
            "classyfire_class": "cf_class",
            "classyfire_superclass": "cf_superclass",
        }
    )

    # Rows: 4 incoming, 1 invalid (SHORTKEY) -> skipped=1
    # Valid comp_ids: AAAQFGUYHFJNHI, AABFWJDLCCDJJN (duplicate -> keep last) => written=2
    assert stats["rows"] == 4
    assert stats["skipped"] == 1
    assert stats["valid"] == 2
    assert stats["written"] == 2

    # Check DB content
    df_db = pd.read_sql_query("SELECT comp_id, smiles, classyfire_class, classyfire_superclass, inchikey, inchi FROM compounds", cdb._conn)
    assert set(df_db["comp_id"]) == {"AAAQFGUYHFJNHI", "AABFWJDLCCDJJN"}

    # Row without full inchikey provided -> stored as NULL. Inchi not provided -> NULL
    assert df_db.loc[df_db["comp_id"] == "AAAQFGUYHFJNHI", "inchikey"].iloc[0] in (None, np.nan, "")
    assert df_db.loc[df_db["comp_id"] == "AAAQFGUYHFJNHI", "inchi"].iloc[0] in (None, np.nan, "")

    # “keep last” behavior for duplicate comp_id
    r = df_db.set_index("comp_id").loc["AABFWJDLCCDJJN"]
    assert r["smiles"] == "S2_override"
    assert r["classyfire_class"] == "C2_override"
    assert r["classyfire_superclass"] == "SC2_override"

    # Settings table is intact and readable
    settings = cdb.get_fingerprint_settings()
    assert {"nbits", "radius", "sparse", "count", "dtype"} <= set(settings.keys())

    cdb.close()


def test_overwrite_metadata_from_dataframe_derive_comp_id_and_true_replace(tmp_path):
    db_path = tmp_path / "compounds.sqlite"
    cdb = CompoundDatabase(str(db_path))

    # First load: only full InChIKeys (custom column name), comp_id must be derived
    df1 = pd.DataFrame({
        "IK_FULL": [
            "BQJCRHHNABKAKU-KBQPJGBKSA-N",
            "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        ],
        "smiles": ["CCO", "O=C=O"],
        "inchi": ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "InChI=1S/CO2/c2-1-3"],
        "cf_class": ["Alcohols", "Carbon oxides"],
        "cf_superclass": ["Organooxygen compounds", "Inorganic compounds"],
    })

    stats1 = cdb.overwrite_metadata_from_dataframe(
        df1,
        column_mapper={
            "inchikey": "IK_FULL",                # derive comp_id from full IK
            "smiles": "smiles",
            "inchi": "inchi",
            "classyfire_class": "cf_class",
            "classyfire_superclass": "cf_superclass",
        }
    )
    assert stats1["written"] == 2
    df_db1 = pd.read_sql_query("SELECT comp_id, inchikey, smiles FROM compounds ORDER BY comp_id", cdb._conn)
    # comp_id equals inchikey14_from_full(inchikey)
    for _, row in df_db1.iterrows():
        assert row["comp_id"] == inchikey14_from_full(row["inchikey"])

    # Second load: replace with a different set -> previous rows must disappear
    df2 = pd.DataFrame({
        "IK_FULL": ["AAOVKJBEBIDNHE-UHFFFAOYSA-N"],
        "smiles": ["CC(=O)O"],
        "cf_class": ["Carboxylic acids"],
        "cf_superclass": ["Organooxygen compounds"],
    })
    stats2 = cdb.overwrite_metadata_from_dataframe(
        df2,
        column_mapper={
            "inchikey": "IK_FULL",
            "smiles": "smiles",
            "classyfire_class": "cf_class",
            "classyfire_superclass": "cf_superclass",
        }
    )
    assert stats2["written"] == 1
    df_db2 = pd.read_sql_query("SELECT comp_id, inchikey, smiles FROM compounds", cdb._conn)
    assert len(df_db2) == 1
    assert df_db2.iloc[0]["comp_id"] == inchikey14_from_full(df_db2.iloc[0]["inchikey"])
    assert set(df_db2["smiles"]) == {"CC(=O)O"}  # previous rows gone (true replace)

    cdb.close()
