from matchms import Spectrum


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
