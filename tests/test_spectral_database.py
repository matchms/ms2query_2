import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from ms2query.database.spectral_database import SpectralDatabase


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "spectra.sqlite"
    db = SpectralDatabase(str(db_path))
    yield db
    db.close()


def make_spectrum(mz, intens, **metadata):
    return Spectrum(
        mz=np.asarray(mz, dtype="float"),
        intensities=np.asarray(intens, dtype="float"),
        metadata=metadata,
    )


@pytest.fixture
def spectra_small():
    s1 = make_spectrum(
        [5, 110, 220, 330, 399, 440],
        [10, 10, 1, 10, 20, 100],
        precursor_mz=240.0,
        ionmode="positive",
    )
    s2 = make_spectrum(
        [50.5, 75.3, 125.0],
        [100, 20, 10],
        precursor_mz=123.4,
        name="test-2",
        instrument_type="Orbitrap",
        collision_energy="[20.0, 30.0, 60.0]"
    )
    s3 = make_spectrum(
        [101, 202, 303, 404],
        [1, 2, 3, 4],
        precursor_mz=404.1,
        inchikey="ABCD-IK",
        collision_energy=35.0,
        adduct="[M+H]+",
    )
    return [s1, s2, s3]


def test_add_and_retrieve_single(tmp_db):
    s = make_spectrum(
        [5, 110, 220, 330, 399, 440],
        [10, 10, 1, 10, 20, 100],
        precursor_mz=240.0,
    )
    ids = tmp_db.add_spectra([s])
    assert isinstance(ids, list) and len(ids) == 1
    sid = ids[0]
    assert isinstance(sid, str)
    assert sid == s.spectrum_hash(), "Returned spec_id should match spectrum hash"

    out = tmp_db.get_spectra_by_ids([sid])
    assert len(out) == 1
    sp_out = out[0]

    # values equal (allow float tolerance), dtype is float32 in storage
    assert np.allclose(sp_out.mz, np.array([5, 110, 220, 330, 399, 440], dtype=np.float32))
    assert np.allclose(sp_out.intensities, np.array([10, 10, 1, 10, 20, 100], dtype=np.float32))
    assert sp_out.mz.dtype == np.float32
    assert sp_out.intensities.dtype == np.float32

    # metadata contains what we stored + spec_id
    assert sp_out.metadata.get("precursor_mz") == pytest.approx(240.0)
    assert sp_out.metadata.get("spec_id") == sid


def test_add_and_retrieve_multiple_order_preserved(tmp_db, spectra_small):
    ids = tmp_db.add_spectra(spectra_small)
    assert len(ids) == 3

    # request in a permuted order; results should follow request order
    req = [ids[2], ids[0], ids[1]]
    out = tmp_db.get_spectra_by_ids(req)
    assert [sp.metadata["spec_id"] for sp in out] == req

    # spot-check one item’s content
    s2 = out[1]  # corresponds to ids[0]
    assert s2.metadata["precursor_mz"] == pytest.approx(240.0)


def test_get_fragments_by_ids(tmp_db, spectra_small):
    ids = tmp_db.add_spectra(spectra_small)
    req = [ids[1], ids[1], 9999999, ids[0]]  # includes duplicate + missing
    # Implementation skips missing IDs and preserves order for the present ones
    frags = tmp_db.get_fragments_by_ids(req)
    # We expect two results (the duplicate is returned twice; missing is skipped)
    assert len(frags) == 3
    (mz_a, in_a), (mz_b, in_b), (mz_c, in_c) = frags

    # dtype should be float32
    for arr in (mz_a, in_a, mz_b, in_b, mz_c, in_c):
        assert arr.dtype == np.float32

    # shape matches inputs for those spectra
    assert mz_a.shape[0] == spectra_small[1].mz.shape[0]
    assert mz_c.shape[0] == spectra_small[0].mz.shape[0]


def test_get_metadata_by_ids_df(tmp_db, spectra_small):
    ids = tmp_db.add_spectra(spectra_small)
    df = tmp_db.get_metadata_by_ids([ids[2], ids[0], ids[1]])

    # Expected columns: spec_id + configured metadata fields
    expected_cols = ["spec_id"] + tmp_db.metadata_fields
    assert list(df.columns) == expected_cols

    # Three rows, in the requested order
    assert df.shape[0] == 3
    assert df.loc[0, "spec_id"] == ids[2]
    assert df.loc[1, "spec_id"] == ids[0]

    # Stored values present / normalized
    assert df.loc[0, "inchikey"] == "ABCD-IK"  # came from spectrum_3
    assert df.loc[1, "precursor_mz"] == pytest.approx(240.0)  # came from spectrum_1
    assert df.loc[2, "collision_energy"] == "[20.0, 30.0, 60.0]"

    # Missing fields become None
    assert pd.isna(df.loc[1, "inchikey"]) or df.loc[1, "inchikey"] is None


def test_sql_query_simple(tmp_db, spectra_small):
    _ = tmp_db.add_spectra(spectra_small)
    df = tmp_db.sql_query("SELECT COUNT(*) AS n FROM spectra")
    assert df.iloc[0]["n"] == 3
    # Query some metadata back
    df2 = tmp_db.sql_query("SELECT spec_id, precursor_mz FROM spectra ORDER BY spec_id")
    assert set(df2.columns) == {"spec_id", "precursor_mz"}
    assert len(df2) == 3


def test_missing_ids_handling(tmp_db, spectra_small):
    ids = tmp_db.add_spectra(spectra_small)
    req = [999999, ids[1]]
    out_spectra = tmp_db.get_spectra_by_ids(req)
    out_meta = tmp_db.get_metadata_by_ids(req)
    out_frags = tmp_db.get_fragments_by_ids(req)

    # Implementation skips missing IDs but preserves order of the ones that exist
    assert [s.metadata["spec_id"] for s in out_spectra] == [ids[1]]

    # get_metadata_by_ids now returns one row per requested ID
    assert out_meta.shape[0] == 2
    assert list(out_meta["spec_id"]) == req

    # First row corresponds to missing ID: all metadata fields should be None/NaN
    missing_row = out_meta.iloc[0]
    assert missing_row["spec_id"] == req[0]
    assert missing_row[tmp_db.metadata_fields].isna().all()

    # Exactly one row has *any* metadata filled (the real spec_id)
    non_empty_rows = out_meta.dropna(how="all", subset=tmp_db.metadata_fields)
    assert non_empty_rows.shape[0] == 1
    assert non_empty_rows.iloc[0]["spec_id"] == ids[1]

    # get_fragments_by_ids still skips missing IDs
    assert len(out_frags) == 1


def test_get_metadata_by_ids_all_ids_included_even_if_same_compound(tmp_db):
    # Two spectra with different peaks but same compound-level metadata
    s1 = make_spectrum(
        [100, 200, 300],
        [10, 20, 30],
        precursor_mz=250.0,
        ionmode="positive",
        inchikey="SAME-IK",
        smiles="C",
        name="compound-1",
    )
    s2 = make_spectrum(
        [110, 210, 310],
        [5, 15, 25],
        precursor_mz=260.0,
        ionmode="positive",
        inchikey="SAME-IK",  # same compound
        smiles="C",
        name="compound-1",
    )

    ids = tmp_db.add_spectra([s1, s2])

    # Request both spec_ids; we expect two rows, one per ID, same order
    df = tmp_db.get_metadata_by_ids(ids)

    expected_cols = ["spec_id"] + tmp_db.metadata_fields
    assert list(df.columns) == expected_cols
    assert df.shape[0] == 2
    assert list(df["spec_id"]) == ids

    # Both rows should carry the same compound-level metadata (same inchikey/smiles)
    assert df.loc[0, "inchikey"] == "SAME-IK"
    assert df.loc[1, "inchikey"] == "SAME-IK"
    assert df.loc[0, "smiles"] == "C"
    assert df.loc[1, "smiles"] == "C"

    # If name is stored, it should be consistent across rows (but may be None)
    assert df.loc[0, "name"] == df.loc[1, "name"]

    # The precursor m/z values differ per spectrum and should be preserved per ID
    assert df.loc[0, "precursor_mz"] == pytest.approx(250.0)
    assert df.loc[1, "precursor_mz"] == pytest.approx(260.0)


def test_add_duplicates_are_ignored_and_ids_repeat(tmp_db, spectra_small):
    # First insert
    ids_first = tmp_db.add_spectra(spectra_small)
    assert len(ids_first) == 3
    # Count rows after first insert
    n1 = tmp_db.sql_query("SELECT COUNT(*) AS n FROM spectra").iloc[0]["n"]
    assert n1 == 3

    # Insert the exact same spectra again
    ids_second = tmp_db.add_spectra(spectra_small)
    assert len(ids_second) == 3

    # Returned IDs must be identical to the first time (hashes are deterministic)
    assert ids_second == ids_first

    # Row count must still be 3 (duplicates were ignored)
    n2 = tmp_db.sql_query("SELECT COUNT(*) AS n FROM spectra").iloc[0]["n"]
    assert n2 == 3

    # ids() should list each unique spec_id exactly once
    all_ids = set(tmp_db.ids())
    assert all_ids == set(ids_first) == set(ids_second)

    # Verify retrieval still works with repeated IDs in the request
    out = tmp_db.get_spectra_by_ids([ids_first[0], ids_first[0], "nonexistent"])
    assert [s.metadata["spec_id"] for s in out] == [ids_first[0], ids_first[0]]
