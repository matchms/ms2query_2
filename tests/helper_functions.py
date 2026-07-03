import os
import string
from pathlib import Path
import numpy as np
from matchms.Spectrum import Spectrum
from ms2deepscore.models import load_model
from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.Fingerprints import Fingerprints


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
            "vanillin",
        ),
        (
            "ROHFNLRQFUQHCH-YFKPBYRVSA-N",
            "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t5-/m0/s1",
            "CC(C)C[C@@H](C(=O)O)N",
            "L-Leucine",
        ),
        (
            "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Aspirin",
        ),
        (
            "WHUUTDBJXJRKMK-VKHMYHEASA-N",
            "InChI=1S/C5H9NO4/c6-3(5(9)10)1-2-4(7)8/h3H,1-2,6H2,(H,7,8)(H,9,10)/t3-/m0/s1",
            "C(CC(=O)O)[C@@H](C(=O)O)N",
            "L-Glutamic acid",
        ),
        (
            "ZKHQWZAMYRWXGA-KQYNXXCUSA-N",
            "InChI=1S/C10H14N5O7P/c11-8-5-9(13-2-12-8)15(3-14-5)10-7(17)6(16)4(22-10)1-21-23(18,19)20/h2-4,6-7,10,16-17H,1H2,(H2,11,12,13)(H2,18,19,20)/t4-,6-,7-,10-/m1/s1",
            "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N",
            "AMP",
        ),
        (
            "GVJHHUAWPYXKBD-UHFFFAOYSA-N",
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "CCO",
            "Ethanol",
        ),
        (
            "IKHGUXGNUITLKF-XPULMUKRSA-N",
            "InChI=1S/C9H13NO3/c1-6(11)13-8-4-2-7(3-5-8)9(10)12/h2-6,11H,1H3,(H2,10,12)/t6-/m0/s1",
            "C[C@@H](C1=CC=C(C=C1)C(=O)N)O",
            "Salbutamol",
        ),
    )
    if number_of_pairs > len(inchikey_inchi_pairs):
        raise ValueError("Not enough example compounds, add some in conftest")
    return inchikey_inchi_pairs[:number_of_pairs]


def get_dummy_inchikeys(nr_of_inchikeys):
    """Creates dummy inchikeys like AAAAAAAAAAAAA, AAAAAAAAAAAAB etc."""
    letters = string.ascii_uppercase
    base = len(letters)
    list_of_inchikeys = []
    counter = 0
    for i in range(nr_of_inchikeys):
        n = counter
        code = []
        for _ in range(14):
            n, rem = divmod(n, base)
            code.append(letters[rem])
        list_of_inchikeys.append("".join(reversed(code)))
        counter += 1
    return list_of_inchikeys


def create_fingerprints(nbits, nr_of_fingerprints):
    """Creates dummy fingerprints"""
    fingerprints = []
    for i in range(nr_of_fingerprints):
        fingerprint = []
        for bit in range(nbits):
            if bit % (i + 1) == 0:
                fingerprint.append(0)
            else:
                fingerprint.append(1)
        fingerprints.append(fingerprint)
    return np.array(fingerprints)


def make_test_fingerprints(nbits=30, nr_of_inchikeys=20):
    inchikeys = get_dummy_inchikeys(nr_of_inchikeys=nr_of_inchikeys)

    fingerprints = create_fingerprints(nbits, nr_of_inchikeys)
    fingerprints = Fingerprints(fingerprints, tuple(inchikeys), fingerprint_type="daylight")
    return fingerprints


def get_library_and_test_spectra_not_identical() -> tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
    model = ms2deepscore_model()
    spectra = create_test_spectra(number_of_spectra_per_inchikey=3, nr_of_inchikeys=3)
    query_spectra = []
    lib_spectra = []
    for i, spectrum in enumerate(spectra):
        if i % 3 == 0:
            query_spectra.append(spectrum)
        else:
            lib_spectra.append(spectrum)
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(lib_spectra)
    test_spectra = AnnotatedSpectrumSet.create_spectrum_set(query_spectra)
    library_spectra.add_embeddings(model)
    test_spectra.add_embeddings(model)
    return library_spectra, test_spectra


def get_library_and_test_spectra_exactly_matching() -> tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
    model = ms2deepscore_model()
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(
        create_test_spectra(number_of_spectra_per_inchikey=3, nr_of_inchikeys=3)
    )
    test_spectra = AnnotatedSpectrumSet.create_spectrum_set(
        create_test_spectra(number_of_spectra_per_inchikey=1, nr_of_inchikeys=3)
    )
    library_spectra.add_embeddings(model)
    test_spectra.add_embeddings(model)
    return library_spectra, test_spectra
