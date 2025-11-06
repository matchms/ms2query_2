from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def inchikey14_from_full(inchikey: str) -> Optional[str]:
    """Return the first 14 characters (inchikey14). Robust to hyphens/malformed keys."""
    if not inchikey:
        return None
    s = str(inchikey).strip().upper()
    if "-" in s:
        return s.split("-", 1)[0][:14]
    return s[:14] if len(s) >= 14 else None



def compute_morgan_fingerprints(
        smiles: Optional[str] = None,
        inchis: Optional[str] = None,
        sparse: bool = True,
        count: bool = True,
        radius: int = 9,
        progress_bar: bool = True,
        ) -> np.ndarray:
    """
    Compute a molecular fingerprint from SMILES or InChI.

    Parameters
    ----------
    smiles : str or None
        SMILES string to compute the fingerprint from.
    inchis : str or None
        InChI strings to compute the fingerprint from (used if smiles is None).
    sparse : bool
        If True, compute sparse fingerprint (indices/counts); else dense bit vector.
    count : bool
        If True, compute count-based fingerprint; else binary fingerprint.
    radius : int
        Radius for Morgan fingerprint. Default 9.
    progress_bar : bool
        Whether to show a progress bar during computation. Default True.
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=4096)

    if inchis and not smiles:
        # convert inchis to smiles
        smiles = []
        for inchi in inchis:
            try:
                mol = Chem.MolFromInchi(inchi)
                smi = Chem.MolToSmiles(mol) if mol is not None else None
                smiles.append(smi)
            except Exception as e:
                print(f"Error converting InChI to SMILES for {inchi}: {e}")
                smiles.append(None)
    elif not smiles and not inchis:
        raise ValueError("Either smiles or inchis must be provided.")
    return compute_fingerprints_from_smiles(
        smiles, 
        fpgen,
        count=count,
        sparse=sparse,
        progress_bar=progress_bar,
    )
