import numpy as np
from matchms import Spectrum
from ms2deepscore.models import compute_embedding_array


def normalize_spectrum_sum(spectrum: Spectrum) -> Spectrum:
    """Return a spectrum with intensities normalized to sum=1 (if possible)."""
    mz = spectrum.peaks.mz
    intens = spectrum.peaks.intensities
    tot = intens.sum()
    if tot > 0:
        intens = intens / tot

    # Build a shallow copy with normalized peaks but same metadata
    md = dict(spectrum.metadata) if hasattr(spectrum, "metadata") else {}
    return Spectrum(mz=mz, intensities=intens, metadata=md)


def compute_spectra_embeddings(
    model,
    spectra: list[Spectrum],
    normalize_spectrum: bool = True,  # TODO: this should probably be consistent with the actual model training?
    normalize_embeddings_L2: bool = True,
    ) -> np.ndarray:
    """Compute MS2DeepScore embeddings for arbitrary query spectra.

    Important: spectral processsing should be consistent throught the application to ensure consistent embeddings!
    """
    # preprocess spectra
    if normalize_spectrum:
        spectra = [normalize_spectrum_sum(s) for s in spectra]

    embeddings = compute_embedding_array(model, spectra).astype(np.float32, copy=False)

    # L2 normalize (EmbeddingIndex assumes/benefits from cosine-normalized vectors)
    if normalize_embeddings_L2:
        n = np.linalg.norm(embeddings, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        embeddings = embeddings / n
    return embeddings
