"""
Copyright 2018-2020 Toyota Resarch Institute. All rights reserved.
Use of this source code is governed by an Apache 2.0
license that can be found in the LICENSE file.
"""
# coding: utf-8

import pytest
from trixs.spectra.spectrum_io import parse_spectrum
from pymatgen.core.spectrum import Spectrum
from trixs.spectra.core import XAS_Spectrum


def test_parse_spectrum():
    """
    Tests that the parse spectrum function can import a simple
    2D CSV file
    :return:
    """

    speca = parse_spectrum('test_files/sample_spectrum_a.txt', skiprows=1)
    assert isinstance(speca, Spectrum)
    assert len(speca) == 417
    assert speca.x[0] == 7570
    assert speca.x[-1] == 8277.18
    assert speca.y[0] == 0.00267023
    assert speca.y[-1] == 0.9949

    specb = parse_spectrum('test_files/sample_spectrum_b.txt', skiprows=1)
    assert isinstance(specb, Spectrum)
    assert len(specb) == 417


# TODO Grab a random test spectrum

@pytest.fixture
def random_spectrum():
    raise NotImplementedError


def test_parse_spectrum_json():
    """
    Tests that the parse spectrum function can import a simple
    2D CSV file
    :return:
    """

    specc = parse_spectrum('test_files/sample_spectrum_c.txt', kind='json')

    # print(specc.full_spectrum)

    # TODO FINISH THIS
    assert isinstance(specc, XAS_Spectrum)
