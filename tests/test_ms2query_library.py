import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from matchms.importing import load_spectra
from ms2query import MS2QueryDatabase, MS2QueryLibrary
from ms2query.library_io import create_new_library, load_created_library


TEST_COMP_ID = "ZBSGKPYXQINNGF"   # known from your snippet
EXPECTED_METADATA_SHAPE = (5, 11)
EXPECTED_METADATA_FIELDS = [
    "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
    "charge", "instrument_type", "adduct", "collision_energy",
]
EMB_PREFIX = "embedding_index"
SQLITE_NAME = "ms2query_library.sqlite"
MANIFEST_NAME = "ms2query_manifest.json"


def _data_dir() -> Path:
    return Path(__file__).parent / "test_data"


def _paths():
    data_dir = _data_dir()
    spectra_file = data_dir / "10_spectra.mgf"
    model_path = data_dir / "ms2deepscore_testmodel_v1.pt"
    assert spectra_file.exists(), f"Missing test spectra file: {spectra_file}"
    assert model_path.exists(), f"Missing test model file: {model_path}"
    return spectra_file, model_path


def create_lib_from_test_files(tmp_path: Path) -> MS2QueryLibrary:
    """Create a small library from test files and return the loaded MS2QueryLibrary."""
    spectra_file, model_path = _paths()
    outdir = tmp_path / "ms2query_out"
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the library (keep HNSW params small for speed)
    lib_created = create_new_library(
        spectra_files=[str(spectra_file)],
        annotation_files=[],
        output_folder=str(outdir),
        model_path=str(model_path),
        build_embedding_index=True,
        embedding_index_params={"M": 8, "ef_construction": 50, "post_init_ef": 50, "batch_rows": 100_000},
        compute_embeddings_batch_rows=256,
    )
    # Sanity
    assert isinstance(lib_created, MS2QueryLibrary)
    assert isinstance(lib_created.db, MS2QueryDatabase)

    # Load from disk to mirror real workflow
    lib_loaded = load_created_library(str(outdir))
    assert isinstance(lib_loaded, MS2QueryLibrary)
    return lib_loaded


# --------------------------------------------------------------------
# End-to-end smoke test (also re-used by the unit tests below)
# --------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_create_and_load_smoke(tmp_path: Path):
    lib = create_lib_from_test_files(tmp_path)

    outdir = tmp_path / "ms2query_out"
    db_path = outdir / SQLITE_NAME
    assert db_path.exists(), "MS2Query database file was not created."

    # Manifest present & contains basic keys
    manifest_path = outdir / MANIFEST_NAME
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("sqlite_path") == SQLITE_NAME

    # DB content checks
    ms2query_db = lib.db
    meta_df = ms2query_db.metadata_by_comp_id(TEST_COMP_ID)
    assert tuple(meta_df.shape) == EXPECTED_METADATA_SHAPE
    for field in EXPECTED_METADATA_FIELDS:
        assert field in ms2query_db.metadata_fields
        assert field in meta_df.columns

    # ANN artifacts (nmslib base + .dat OR legacy .nmslib)
    emb_base = outdir / EMB_PREFIX
    two_file_ok = emb_base.exists() and (emb_base.with_suffix(".dat")).exists()
    legacy_ok = (emb_base.with_suffix(".nmslib")).exists()
    assert two_file_ok or legacy_ok, "Embedding index files missing."


# --------------------------------------------------------------------
# Unit tests for MS2QueryLibrary methods
# --------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_process_and_compute_embeddings(tmp_path: Path):
    lib = create_lib_from_test_files(tmp_path)
    spectra_path, _ = _paths()
    spectra = list(load_spectra(spectra_path))
    assert len(spectra) > 0

    # process_spectra passthrough for now
    processed = lib.process_spectra(spectra)
    assert isinstance(processed, list)
    assert len(processed) == len(spectra)

    # compute_embeddings returns (n, d) float32, d > 0
    E = lib.compute_embeddings(spectra[:3])  # small batch
    assert isinstance(E, np.ndarray)
    assert E.dtype == np.float32
    assert E.ndim == 2 and E.shape[0] == 3 and E.shape[1] > 0

    # L2 normalization sanity (norm ~ 1)
    norms = np.linalg.norm(E, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_query_embedding_index_returns_dataframe(tmp_path: Path):
    lib = create_lib_from_test_files(tmp_path)
    spectra_path, _ = _paths()
    spectra = list(load_spectra(spectra_path))
    q = spectra[0]

    df = lib.query_embedding_index(q, k=3, ef=40, return_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert set(["query_ix", "rank", "spec_id", "score"]).issubset(df.columns)
    assert df.shape[0] >= 1
    assert isinstance(df["spec_id"].iloc[0], (str, np.str_))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_query_spectra_by_spectra_and_compounds(tmp_path: Path):
    lib = create_lib_from_test_files(tmp_path)
    spectra_path, _ = _paths()
    spectra = list(load_spectra(spectra_path))

    # spectra-by-spectra (DataFrame)
    df_s = lib.query_spectra_by_spectra(spectra[:2], k_spectra=5, ef=40)
    assert isinstance(df_s, pd.DataFrame)
    assert set(["query_ix", "rank", "spec_id", "score"]).issubset(df_s.columns)
    assert df_s["query_ix"].nunique() == 2

    # compounds-by-spectra (top-k compounds per query)
    df_c = lib.query_compounds_by_spectra(spectra[:3], k_spectra=20, k_compounds=5, ef=40)
    # Expect columns from metadata + query_ix/rank/score present after merge
    required_cols = set(["query_ix", "spec_id", "score", "inchikey"]).union(EXPECTED_METADATA_FIELDS)
    assert required_cols.issubset(df_c.columns)
    # per query, at most k_compounds rows
    assert (df_c.groupby("query_ix").size() <= 5).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_query_by_spec_ids_uses_db_embeddings(tmp_path: Path):
    lib = create_lib_from_test_files(tmp_path)
    # grab a few ids from the DB
    spec_ids = lib.db.ref_sdb.ids()[:2]
    assert len(spec_ids) >= 1

    df = lib.query_by_spec_ids(spec_ids, k=4, ef=40, return_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert set(["query_ix", "rank", "spec_id", "score"]).issubset(df.columns)
    # same number of query_ix values as requested ids
    assert df["query_ix"].nunique() == len(spec_ids)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_error_paths_missing_model_or_index(tmp_path: Path):
    # Fresh DB only (no index attached)
    lib = create_lib_from_test_files(tmp_path)

    # Remove index to test guard rails
    lib.embedding_index = None
    spectra_path, _ = _paths()
    spectra = list(load_spectra(spectra_path))
    with pytest.raises(RuntimeError, match="EmbeddingIndex is not set"):
        lib.query_embedding_index(spectra[0], k=3)

    # New instance without model_path → compute_embeddings should raise when invoked
    lib2 = MS2QueryLibrary(db=lib.db, embedding_index=None, model_path=None)
    with pytest.raises(RuntimeError, match="model_path is not set"):
        lib2.compute_embeddings(spectra[:1])
