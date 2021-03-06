# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

"""
Unit tests for spectrum analyze methods
"""

import os
from trixs.spectra.spectrum_io import parse_spectrum
from trixs.spectra.spectrum_compare import compare_spectrum
from math import isclose
from pytest import fixture

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, 'test_files')

# TODO Roll this into future tests
@fixture
def sample_spectrum():
    return parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_a.txt'), skiprows=1)


def test_spectrum_comparison_basic():
    """
    Compares a spectrum to itself to verify the similarity measures
    are running
    :return:
    """
    specA = parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_a.txt'), skiprows=1)

    specB = parse_spectrum(os.path.join(TEST_FILE_DIR, 'sample_spectrum_b.txt'), skiprows=1)

    pers_aa = compare_spectrum(specA, specA, 'pearson')
    eucl_aa = compare_spectrum(specA, specA, 'euclidean')
    cos_aa = compare_spectrum(specA, specA, 'cosine')
    ruz_aa = compare_spectrum(specA, specA, 'ruzicka')

    pers_bb = compare_spectrum(specB, specB, 'pearson')
    eucl_bb = compare_spectrum(specB, specB, 'euclidean')
    cos_bb = compare_spectrum(specB, specB, 'cosine')
    ruz_bb = compare_spectrum(specB, specB, 'ruzicka')

    assert (pers_aa == 1.0)
    assert (eucl_aa == 0.0)
    assert (ruz_aa == 1.0)
    assert isclose(cos_aa ,1.0)

    assert (pers_bb == 1.0)
    assert (eucl_bb == 0.0)
    assert (ruz_bb == 1.0)
    assert isclose(cos_bb,1.0)
