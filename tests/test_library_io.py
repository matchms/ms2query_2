# tests/test_library_workflow.py
from pathlib import Path
import numpy as np
import pytest
from matchms.importing import load_spectra
from ms2query import MS2QueryDatabase
from ms2query.library_io import create_new_library, load_created_library


TEST_COMP_ID = "ZBSGKPYXQINNGF"   # expected InChIKey14 present in the test data
EXPECTED_METADATA_SHAPE = (5, 12)
EXPECTED_METADATA_FIELDS = [
    "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
    "charge", "instrument_type", "adduct", "collision_energy",
]


def _data_dir() -> Path:
    return Path(__file__).parent / "test_data"


def _paths():
    data_dir = _data_dir()
    spectra_file = data_dir / "10_spectra.mgf"
    model_path = data_dir / "ms2deepscore_testmodel_v1.pt"
    assert spectra_file.exists(), f"Missing test spectra file: {spectra_file}"
    assert model_path.exists(), f"Missing test model file: {model_path}"
    return spectra_file, model_path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_create_and_load_library(tmp_path: Path):
    spectra_file, model_path = _paths()

    # ---------- Create ----------
    outdir = tmp_path / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    lib = create_new_library(
        spectra_files=[str(spectra_file)],
        annotation_files=[],                 # currently unused in the workflow
        output_folder=str(outdir),
        model_path=str(model_path),
        # Keep the index small/fast for CI:
        embedding_index_params={"M": 8, "ef_construction": 50, "post_init_ef": 50, "batch_rows": 100_000},
    )
    # basic sanity
    assert lib is not None
    assert isinstance(lib.db, MS2QueryDatabase)

    # SQLite must exist
    db_path = outdir / "ms2query_library.sqlite"
    assert db_path.exists(), "MS2Query database file was not created."

    # ---------- DB content checks ----------
    ms2query_db = lib.db

    # Metadata query by compound id (expected shape from your snippet)
    df_meta = ms2query_db.metadata_by_comp_ids([TEST_COMP_ID])
    assert tuple(df_meta.shape) == EXPECTED_METADATA_SHAPE

    # Metadata fields presence both in db wrapper and in returned dataframe
    md_fields = ms2query_db.metadata_fields
    for f in EXPECTED_METADATA_FIELDS:
        assert f in md_fields, f"Metadata field '{f}' is missing in the database."
        assert f in df_meta.columns, f"Metadata field '{f}' missing in metadata_by_comp_id result."

    # ---------- Embedding index artifacts ----------
    # The workflow saves with a base prefix (no extension). NMSLIB writes two files: <prefix> and <prefix>.dat
    emb_prefix = outdir / "embedding_index"
    pair_exists = emb_prefix.exists() and (emb_prefix.with_suffix(".dat")).exists()
    assert pair_exists, "EmbeddingIndex files not found."

    # ---------- Load ----------
    lib2 = load_created_library(str(outdir))
    assert lib2 is not None
    assert isinstance(lib2.db, MS2QueryDatabase)

    # ---------- Tiny embedding-index query ----------
    # Take one spectrum from the same file and try a small top-k query
    spectra = list(load_spectra(spectra_file))
    assert len(spectra) > 0, "No spectra parsed from test file."

    # queries should return non-empty hits with spec_ids
    results = lib2.query_embedding_index(spectra[0], k=3, return_dataframe=False)
    assert isinstance(results, list)
    assert len(results) == 1

    hits = results[0]
    assert len(hits) == 3, "k was set to 3, but returned different number of hits."
    assert set(hits[0].keys()) >= {"rank", "spec_id", "score"}
    assert isinstance(hits[0]["spec_id"], (str, np.str_))
