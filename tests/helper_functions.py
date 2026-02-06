import os
from pathlib import Path
import numpy as np
from matchms.Spectrum import Spectrum
from ms2deepscore.models import load_model


TEST_RESOURCES_PATH = Path(__file__).parent / "test_data"


def create_test_spectra(
    number_of_spectra_per_inchikey=3,
    inchikey_inchi_pairs=None,
    nr_of_inchikeys=3,
) -> list[Spectrum]:
    if inchikey_inchi_pairs is None:
        inchikey_inchi_pairs = get_inchikey_inchi_pairs(nr_of_inchikeys)
    spectra = []
    for i, inchikey_inchi_tuple in enumerate(inchikey_inchi_pairs):
        inchikey, inchi, smiles, compound_name = inchikey_inchi_tuple
        for j in range(number_of_spectra_per_inchikey):
            spectra.append(
                Spectrum(
                    mz=np.array([100 + i * 10.0, 500 + i * 1.0]),
                    intensities=np.array([1.0, 1.0 / (j + 1)]),
                    metadata={
                        "precursor_mz": 111.1 + i * 10,
                        "inchikey": inchikey,
                        "inchi": inchi,
                        "smiles": smiles,
                        "compound_name": compound_name,
                    },
                )
            )
    return spectra


def ms2deepscore_model():
    return load_model(os.path.join(TEST_RESOURCES_PATH, "ms2deepscore_testmodel_v1.pt"))


def get_inchikey_inchi_pairs(number_of_pairs):
    """Returns inchikey_inchi_pairs"""
    inchikey_inchi_pairs = (
        (
            "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Caffeine",
        ),
        (
            "ZPUCINDJVBIVPJ-LJISPDSOSA-N",
            "InChI=1S/C17H21NO4/c1-18-12-8-9-13(18)15(17(20)21-2)14(10-12)22-16(19)11-6-4-3-5-7-11/h3-7,12-15H,8-10H2,1-2H3/t12-,13+,14-,15+/m0/s1",
            "CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC",
            "Cocaine",
        ),
        (
            "RZVAJINKPMORJF-UHFFFAOYSA-N",
            "InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)",
            "CC(=O)NC1=CC=C(C=C1)O",
            "Paracetemol",
        ),
        (
            "JGSARLDLIJGVTE-MBNYWOFBSA-N",
            "InChI=1S/C16H18N2O4S/c1-16(2)12(15(21)22)18-13(20)11(14(18)23-16)17-10(19)8-9-6-4-3-5-7-9/h3-7,11-12,14H,8H2,1-2H3,(H,17,19)(H,21,22)/t11-,12+,14-/m1/s1",
            "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
            "Penicillin",
        ),
        (
            "WQZGKKKJIJFFOK-GASJEMHNSA-N",
            "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1",
            "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
            "Glucose",
        ),
        (
            "MWOOGOJBHIARFG-UHFFFAOYSA-N",
            "InChI=1S/C8H8O3/c1-11-8-4-6(5-9)2-3-7(8)10/h2-5,10H,1H3",
            "COC1=C(C=CC(=C1)C=O)O",
            "vanillin"
        ),
        (
            "ROHFNLRQFUQHCH-YFKPBYRVSA-N",
            "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t5-/m0/s1",
            "CC(C)C[C@@H](C(=O)O)N",
            "L-Leucine"
        )
    )
    if number_of_pairs > len(inchikey_inchi_pairs):
        raise ValueError("Not enough example compounds, add some in conftest")
    return inchikey_inchi_pairs[:number_of_pairs]
