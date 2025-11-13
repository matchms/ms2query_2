import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from matchms.importing import load_spectra
from matchms.similarity import CosineGreedy
from tqdm import tqdm
from ms2query import MS2QueryDatabase, MS2QueryLibrary
from ms2query.data_processing.merging_utils import cluster_block, get_merged_spectra
from ms2query.database import EmbeddingIndex
from ms2query.database.spectra_merging import _split_by_mode_charge


# ------------------------ defaults & helpers ------------------------

_MANIFEST_NAME = "ms2query_manifest.json"
_SQLITE_NAME   = "ms2query_library.sqlite"
_EMB_TABLE     = "embeddings"
_EMB_INDEX_BASENAME = "embedding_index"     # will create embedding_index.{nmslib,ids.npy,meta.json}


def _handle_default_settings(settings: dict) -> dict:
    """Ensure all necessary settings have default values if not provided."""
    defaults = {
        "spectrum_sum_normalization_for_embedding": True,
        "mz_tol": 0.01,
        "min_frac": 0.25,
        "cosine_thr": 0.95,
        "intensity_power": 0.5,
    }
    cfg = dict(defaults)
    cfg.update(settings or {})
    return cfg

def _print_progress(message: str):
    """Small progress printer."""
    def get_spec_ascii(size=15):
        return "".join(np.random.choice(["'", "|"], size=size))
    print(get_spec_ascii() + f" >> MS2Query >> {message}")

def _merge_spectra(all_spectra, **settings):
    """Merge spectra of the same compound (InChIKey14) that are highly similar."""
    inchikeys14 = np.array([s.get("inchikey")[:14] if s.get("inchikey") else "" for s in all_spectra])

    merged_spectra = []
    for inchikey in tqdm(np.unique(inchikeys14), desc="Merging spectra by InChIKey14 and cosine similarity..."):
        idx = np.where(inchikeys14 == inchikey)[0]
        spectra = [all_spectra[i] for i in idx]

        # Split by (ionmode, charge) to avoid over-merging
        groups = _split_by_mode_charge(spectra)

        for (_, _), idxs in groups.items():
            block_spectra = [spectra[i] for i in idxs]

            # Cluster within block
            clusters_local, _ = cluster_block(
                block_spectra,
                sim_score=CosineGreedy(intensity_power=settings["intensity_power"]),
                threshold=settings["cosine_thr"]
            )

            # Merge spectra
            merged_block = get_merged_spectra(
                block_spectra, clusters_local,
                mz_tol=settings["mz_tol"], min_frac=settings["min_frac"]
            )

            merged_spectra.extend(merged_block)
    return merged_spectra


# ------------------------------ create ------------------------------

def create_new_library(
    spectra_files: List[str],
    annotation_files: Optional[List[str]],
    output_folder: str,
    *,
    model_path: str,
    additional_compound_file: Optional[str] = None,
    build_embedding_index: bool = True,
    embedding_index_params: Optional[dict] = None,
    compute_embeddings_batch_rows: int = 4096,
    **settings,
) -> MS2QueryLibrary:
    """
    Create a new MS2Query library (SQLite + indices) from spectra (MGF/etc.) and annotations.

    Parameters
    ----------
    spectra_files : list of paths
        Spectral data files to ingest with matchms.
    annotation_files : list of paths (currently unused placeholder; kept for compatibility)
    output_folder : str
        Target folder for SQLite + indices + manifest.
    model_path : str
        Path to MS2DeepScore .pt model used to compute embeddings.
    additional_compound_file : str, optional
        CSV/TSV file with additional compounds (inchikey/smiles/etc.). No fingerprints assumed here.
    build_embedding_index : bool
        Whether to build the nmslib cosine HNSW index over embeddings.
    embedding_index_params : dict
        Params for HNSW: {'M': int, 'ef_construction': int, 'post_init_ef': int, 'batch_rows': int}
    compute_embeddings_batch_rows : int
        Batch size for writing embeddings into SQLite.
    settings : dict
        Merge settings for collapsing near-identical spectra of the same compound.

    Returns
    -------
    MS2QueryLibrary
        Ready-to-use library object (DB + EmbeddingIndex attached if built).
    """
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = _handle_default_settings(settings)
    _print_progress("Loading spectra from files...")

    all_spectra = []
    for spectra_file in tqdm(spectra_files, desc="Loading spectra files"):
        all_spectra.extend(list(load_spectra(spectra_file)))

    _print_progress(f"Loaded {len(all_spectra)} spectra.")

    # Process pipeline hook (kept minimal; merging relies on matchms utils you provided)
    merged_spectra = _merge_spectra(all_spectra, **settings)
    _print_progress(f"After merging, {len(merged_spectra)} spectra remain.")

    # -----------------------------
    # Create database + ingest spectra/compounds
    # -----------------------------
    db_path = out_dir / _SQLITE_NAME
    ms2query_db = MS2QueryDatabase(sqlite_path=str(db_path))
    creation_stats = ms2query_db.create_from_spectra(merged_spectra)
    _print_progress(f"Created MS2Query database at {db_path}.")
    _print_progress(f"Inserted {creation_stats['n_inserted_spectra']} spectra.")
    _print_progress(f"Mapped {creation_stats['n_mapped']} spectra to compounds; "
                    f"created {creation_stats['n_new_compounds']} new compounds.")

    if additional_compound_file is not None:
        if not additional_compound_file.lower().endswith((".csv", ".tsv", ".txt")):
            raise ValueError("Additional compound file must be CSV/TSV/TXT.")
        _print_progress("Adding additional compound database...")
        # Light reader that handles CSV/TSV by extension; customize as needed
        if additional_compound_file.lower().endswith(".csv"):
            df_compounds = pd.read_csv(additional_compound_file)
        else:
            df_compounds = pd.read_csv(additional_compound_file, sep="\t")
        ms2query_db.add_second_compound_database(df_compounds)
        _print_progress("Additional compound database added.")

    # -----------------------------
    # Embeddings (SQLite) + EmbeddingIndex
    # -----------------------------
    # Compute & store MS2DeepScore embeddings in the reference SpectralDatabase
    _print_progress("Computing & writing embeddings to SQLite ...")
    n_new = ms2query_db.ref_sdb.compute_embeddings_to_sqlite(
        model_path=model_path,
        embeddings_table=_EMB_TABLE,
        batch_rows=compute_embeddings_batch_rows,
        only_missing=True,
    )
    _print_progress(f"Wrote {n_new} new embeddings (table '{_EMB_TABLE}').")

    # Create central library object (and pass db, model and parameters!)
    lib = MS2QueryLibrary(
        db=ms2query_db, model_path=model_path,
        _spectrum_sum_normalization_for_embedding=settings["spectrum_sum_normalization_for_embedding"]
        )

    if build_embedding_index:
        _print_progress("Building EmbeddingIndex (nmslib/HNSW cosine) ...")
        params = {
            "M": 16,
            "ef_construction": 200,
            "post_init_ef": 200,
            "batch_rows": 100_000,  # streaming from SQLite
        }
        if embedding_index_params:
            params.update(embedding_index_params)

        emb_index = EmbeddingIndex()  # dim will be inferred from DB
        # Stream embeddings from SQLite --> HNSW
        n_vecs = emb_index.build_index_from_sqlite(
            ms2query_db.ref_sdb.connection,
            embeddings_table=_EMB_TABLE,
            where_sql=None,                       # or e.g. "WHERE d=500"
            batch_rows=params["batch_rows"],
            M=params["M"],
            ef_construction=params["ef_construction"],
            post_init_ef=params["post_init_ef"],
            l2_normalize=True,
        )
        _print_progress(f"Indexed {n_vecs} embedding vectors.")
        emb_prefix = str(out_dir / _EMB_INDEX_BASENAME)
        emb_index.save_index(emb_prefix)
        _print_progress(f"Saved EmbeddingIndex files with prefix: {emb_prefix}")
        lib.set_embedding_index(emb_index)

    # -----------------------------
    # Manifest
    # -----------------------------
    manifest = {
        "format_version": 1,
        "sqlite_path": str(db_path.name),
        "embedding_table": _EMB_TABLE,
        "model_path": model_path,                    # stored for convenience; not copied
        "embedding_index_prefix": _EMB_INDEX_BASENAME if build_embedding_index else None,
        "settings": settings,
    }
    with open(out_dir / _MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _print_progress(f"Wrote manifest: {out_dir / _MANIFEST_NAME}")

    _print_progress("Library creation complete.")
    return lib


# ------------------------------- load -------------------------------

def load_created_library(folder: str) -> MS2QueryLibrary:
    """
    Load a previously created MS2Query library (SQLite + indices + manifest).

    Parameters
    ----------
    folder : str
        Path to the output folder produced by create_new_library().

    Returns
    -------
    MS2QueryLibrary
        Connected database with indices loaded (if present).
    """
    folder = str(folder)
    out_dir = Path(folder)
    manifest_path = out_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    db_path = out_dir / manifest.get("sqlite_path", _SQLITE_NAME)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_path}")

    # Build DB wrapper
    ms2query_db = MS2QueryDatabase(sqlite_path=str(db_path))

    # Create central library object
    lib = MS2QueryLibrary(
        db=ms2query_db,
        model_path=manifest.get("model_path")  # may be None; you can still query with precomputed embeddings-by-id
    )

    # Load EmbeddingIndex if present
    emb_prefix = manifest.get("embedding_index_prefix")
    if emb_prefix:
        emb_index = EmbeddingIndex()
        emb_index.load_index(str(out_dir / emb_prefix))
        lib.set_embedding_index(emb_index)

    # (Optional) Load fingerprint index here if/when you add it later.

    return lib
