# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

import os
from pytest import fixture
from trixs.spectra.spectrum_io import parse_spectrum
from pymatgen.core.spectrum import Spectrum
from trixs.spectra.core import XAS_Spectrum

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, 'test_files')


def test_parse_spectrum():
    """
    Tests that the parse spectrum function can import a simple
    2D CSV file
    :return:
    """

    speca = parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_a.txt'), skiprows=1)
    assert isinstance(speca, Spectrum)
    assert len(speca) == 417
    assert speca.x[0] == 7570
    assert speca.x[-1] == 8277.18
    assert speca.y[0] == 0.00267023
    assert speca.y[-1] == 0.9949

    specb = parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_b.txt'), skiprows=1)
    assert isinstance(specb, Spectrum)
    assert len(specb) == 417


# TODO Grab a random tests spectrum

@fixture
def random_spectrum():
    raise NotImplementedError


def test_parse_spectrum_json():
    """
    Tests that the parse spectrum function can import a simple
    2D CSV file
    :return:
    """

    specc = parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_c.txt'), kind='json')

    # print(specc.full_spectrum)

    # TODO FINISH THIS
    assert isinstance(specc, XAS_Spectrum)
