"""
Copyright 2018-2020 Toyota Resarch Institute. All rights reserved.
Use of this source code is governed by an Apache 2.0
license that can be found in the LICENSE file.
"""
import pytest
import numpy as np
from trixs.spectra.spectrum_io import parse_spectrum

from trixs.spectra.spectrum_normalize import normalize_sum, normalize_l2, normalize_max, \
    normalize_minmax, normalize_z, normalize_0left_1right

from math import isclose


@pytest.fixture
def sample_spectrum():
    yield parse_spectrum('test_files/sample_spectrum_a.txt', skiprows=1)


def test_sum_normalization(sample_spectrum):
    """
    Test sum normalization method
    :param sample_spectrum:
    :return:
    """

    spec1 = normalize_sum(sample_spectrum, in_place=False)

    assert (spec1 is not sample_spectrum)
    assert (np.sum(spec1.y) == 1.0)

    normalize_sum(sample_spectrum, in_place=True, value=3)

    assert (isclose(np.sum(sample_spectrum.y), 3.0))


def test_max_normalization(sample_spectrum):
    """
    Test max normalization method
    :param sample_spectrum:
    :return:
    """
    spec1 = normalize_max(sample_spectrum)
    normalize_max(sample_spectrum, in_place=True)
    assert (spec1 is not sample_spectrum)
    assert max(spec1.y) == 1.0
    assert isclose(np.max(sample_spectrum.y), 1.0)


def test_l2_normalization(sample_spectrum):
    """
    Test L2 normalization method
    :param sample_spectrum:
    :return:
    """

    l2norm = np.linalg.norm(sample_spectrum.y, ord=2)
    spec1 = normalize_l2(sample_spectrum, in_place=False)

    assert np.isclose(l2norm * spec1.y, sample_spectrum.y).all()
    sumsq = np.sqrt(sum([y ** 2 for y in spec1.y]))
    assert isclose(sumsq, 1.0)

    normalize_l2(sample_spectrum, in_place=True)
    assert (spec1 is not sample_spectrum)

    sumsq = np.sqrt(sum([y ** 2 for y in sample_spectrum.y]))
    assert isclose(sumsq, 1.0)


def test_z_normalization(sample_spectrum):
    """
    Test the z-normalization
    :param sample_spectrum:
    :return:
    """

    # Reset sample spectrum values with a normal distribution
    sample_spectrum.y = np.random.normal(size=len(sample_spectrum.y))

    spec1 = normalize_z(sample_spectrum)
    normalize_z(sample_spectrum, in_place=True)

    assert spec1 is not sample_spectrum

    assert -0.1 <= np.mean(spec1.y) <= 0.1
    assert 0.9 <= np.std(spec1.y) <= 1.1

    assert -0.1 <= np.mean(sample_spectrum.y) <= 0.1
    assert 0.9 <= np.std(sample_spectrum.y) <= 1.1


def test_minmax_normalization(sample_spectrum):
    """
    Test the minmax normalization

    :param sample_spectrum:
    :return:
    """

    spec1 = normalize_minmax(sample_spectrum)
    normalize_minmax(sample_spectrum, in_place=True)

    for spec in [spec1, sample_spectrum]:
        assert isclose(np.min(spec.y), 0.0)
        assert isclose(np.max(spec.y), 1.0)

def test_0to1_normalization(sample_spectrum):


    spec1 = normalize_0left_1right(sample_spectrum)

    normalize_0left_1right(sample_spectrum, in_place=True)

    for spec in [spec1, sample_spectrum]:
        assert isclose(spec.y[0], 0.0)
        assert isclose(np.mean(spec.y[-5:]), 1.0)


    pass