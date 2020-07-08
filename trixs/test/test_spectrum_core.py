# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

"""
Defines a new XAS Spectrum object built on top of Pymatgen's
Spectrum object.
"""

import numpy as np
from pymatgen.core.structure import Structure
from trixs.spectra.core import XAS_Spectrum, XAS_Collation
from trixs.spectra.spectrum_io import parse_spectrum
from copy import deepcopy
from numpy import eye
from pytest import fixture, raises
from unittest import TestCase
from json import loads, dumps


@fixture
def fake_structure():
    lattice = eye(3)
    species = ['H']
    coords = np.array([[0, 0, 0]])

    yield Structure(lattice, species, coords)


@fixture
def fake_spectrum(fake_structure):
    x = np.random.uniform(size=100)
    y = np.random.uniform(size=100)

    return XAS_Spectrum(x, y, structure=fake_structure,
                        absorbing_site=0)


def test_instantiate_XAS_spectra(fake_structure):
    x = np.random.uniform(size=100)
    y = np.random.uniform(size=100)

    absorbing_site = 0

    spec = XAS_Spectrum(x, y, fake_structure, absorbing_site)

    assert isinstance(spec, XAS_Spectrum)


def test_XAS_full_spec_attributes():
    x = np.random.uniform(size=100)
    y = np.random.uniform(size=100)

    structure = Structure.from_file('./test_files/Cu_structure.cif')

    absorbing_site = 0

    full_spectrum = np.random.uniform(size=(100, 6))

    spec = XAS_Spectrum(x, y, structure, absorbing_site, full_spectrum=full_spectrum)

    assert isinstance(spec, XAS_Spectrum)

    assert np.array_equal(spec.E, full_spectrum[:, 0])
    assert np.array_equal(spec.Enorm, full_spectrum[:, 1])
    assert np.array_equal(spec.k, full_spectrum[:, 2])
    assert np.array_equal(spec.mu, full_spectrum[:, 3])
    assert np.array_equal(spec.mu0, full_spectrum[:, 4])
    assert np.array_equal(spec.chi, full_spectrum[:, 5])
    assert spec.abs_idx == 0
    assert isinstance(spec.as_dict(), dict)


def test_exceptions(fake_spectrum):

    with raises(ValueError):
        fake_spectrum.E()
    with raises(ValueError):
        fake_spectrum.mu()
    with raises(ValueError):
        fake_spectrum.Enorm()
    with raises(ValueError):
        fake_spectrum.mu0()
    with raises(ValueError):
        fake_spectrum.k()
    with raises(ValueError):
        fake_spectrum.chi()
    with raises(ValueError):
        fake_spectrum.shifted_Enorm(shift=0)

    with raises(NotImplementedError):
        fake_spectrum.normalize('zappa')



def test_load_from_doc_and_object():

    with open('./test_files/sample_spectrum_e.txt','r') as f:
        data = loads(f.readline())

    spec1 = XAS_Spectrum.from_atomate_document(data)
    spec2 = XAS_Spectrum.load_from_object(data)

    line = dumps(data)
    spec3 = XAS_Spectrum.load_from_object(line)


    for spec in [spec1, spec2, spec3]:
        assert isinstance(spec,XAS_Spectrum)
        assert spec.has_full_spectrum()
        assert spec.E[0] == 8334.08
        assert spec.Enorm[0] == -9.293
        assert spec.k[0] == -0.8
        assert spec.mu[0] == 0.0519168
        assert spec.mu0[0] == 0.0795718
        assert spec.chi[0] == -0.027655

        assert len(spec.E) == 100
        assert len(spec.Enorm) == 100
        assert len(spec.mu) == 100
        assert len(spec.mu0) == 100
        assert len(spec.k) == 100
        assert len(spec.chi) == 100


    enorm = spec1.Enorm
    sub_enorm = np.add(enorm,1)
    assert np.isclose(sub_enorm,spec.shifted_Enorm(1)).all()



def test_XAS_gradient():
    X = np.linspace(0,1,500)
    Y = np.sin(X)

    Yprime = np.cos(X)

    spec = XAS_Spectrum(X,Y)
    dy = spec.dy
    assert np.isclose(dy, Yprime,atol=.001).all()


def test_XAS_Spectrum_methods(fake_spectrum):
    assert isinstance(str(fake_spectrum), str)
    assert 'H' in str(fake_spectrum)
    assert isinstance(fake_spectrum.as_dict(), dict)
    assert isinstance(fake_spectrum.as_str(),str)
    assert fake_spectrum.has_full_spectrum() is False

    X = np.linspace(0,1,100)
    Y = np.sin(X)

    assert np.isclose(np.add(fake_spectrum.x, 10), fake_spectrum.shifted_x(10)).all()

    fake_spectrum_2 = XAS_Spectrum(X,Y)

    assert fake_spectrum_2.get_peak_idx() == 99

    fake_spectrum_2.normalize('sum')
    assert np.isclose(np.sum(fake_spectrum_2.y), 1.0)

    fake_spectrum_2.normalize('max')
    assert np.isclose(np.max(fake_spectrum_2.y), 1.0)

    # Reset and shift up
    fake_spectrum_2 = XAS_Spectrum(X,Y+.2)
    fake_spectrum_2.normalize('minmax')
    assert np.isclose(np.min(fake_spectrum_2.y), 0)
    assert np.isclose(np.max(fake_spectrum_2.y), 1.0)

    fake_spectrum_2.normalize('l2')
    assert np.isclose(np.linalg.norm(fake_spectrum_2.y, ord=2),1.0)



def test_XAS_shift():
    spec1 = parse_spectrum('test_files/sample_spectrum_c.txt', kind='json')

    spec2 = deepcopy(spec1)

    spec1.x = np.subtract(spec1.x, 5.0)
    assert (abs(5.0 + spec1.get_shift_alignment(spec2, fidelity=20, iterations=3)[0]) < .1)


def test_projection():
    x = np.linspace(0, np.pi / 2, 100)
    y = np.sin(x)

    spec = XAS_Spectrum(x, y)

    # Test that interpolation method works

    new_xvals = np.linspace(0, np.pi / 2, 300)
    new_yvals = spec.project_to_x_range(new_xvals)
    assert (max(np.abs(y - new_yvals[::3])) < 1e-2)

    new_xvals = np.linspace(-np.pi / 2, np.pi, 300)
    new_yvals = spec.project_to_x_range(new_xvals)
    true_yvals = np.sin(new_xvals)

    # Test that it pads to 0 on the left edge
    assert np.isclose(0.0, np.abs(new_yvals[:33])).all()
    # Test that it extrapolates correctly for a few points
    assert (np.isclose(new_yvals[200:205], true_yvals[200:205])).all()


def test_broaden_spectrum():
    x = np.linspace(0, 1, 100)
    y = np.sin(x)
    spec = XAS_Spectrum(x, y)
    assert x[-1]-x[0] == 1.0

    spec.broaden_spectrum_mult(0)
    assert spec.x[-1]-spec.x[0] == 1.0


    spec.broaden_spectrum_mult(.05)
    assert np.isclose(spec.x[-1] - spec.x[0], 1.05)

    assert spec.x[-1] == 1.025
    assert spec.x[0] ==-.025

    x = np.linspace(0, 1, 100)
    y = np.sin(x)
    spec = XAS_Spectrum(x, y)
    spec.broaden_spectrum_mult(-.05)

    assert np.isclose(spec.x[-1] - spec.x[0], .95)
    assert spec.x[-1] ==.975
    assert spec.x[0] ==.025


def test_sanity_check():
    X = np.linspace(0, 1, 100)
    Y = np.linspace(.5, -.5, 100)
    bad_spec = XAS_Spectrum(x=X, y=Y)
    assert bad_spec.sanity_check() is False

    good_spec = XAS_Spectrum(x=X, y=X)
    assert good_spec.sanity_check() is True

    Y = np.linspace(-0.001,0.001, 100)

    bad_spec = XAS_Spectrum(x=X,y=Y)
    assert bad_spec.sanity_check() is False



def test_simple_XAS_collation(fake_structure):
    col = XAS_Collation(fake_structure)
    assert isinstance(XAS_Collation(fake_structure), XAS_Collation)

    all_false = [col.has_mp_spectra(),
                 col.has_mp_bader(),
                 col.has_oqmd_bader(),
                 col.has_bader(),
                 col.has_feff_spectra(),
                 col.has_features(),
                 col.has_spectra()]
    assert not all(all_false)

def test_full_XAS_collation(fake_structure):
    col = XAS_Collation(fake_structure,
                        mp_id='frank',
                        oqmd_id='zappa',
                        mp_spectra=[fake_spectrum],
                        feff_spectra=[fake_spectrum],
                        icsd_ids=[1,2,3],
                        mp_bader=[1],
                        oqmd_bader=[1],
                        coordination_numbers=[1])

    all_true = [col.has_mp_spectra(),
                 col.has_mp_bader(),
                 col.has_oqmd_bader(),
                 col.has_bader(),
                 col.has_feff_spectra(),
                 col.has_spectra(),
                 col.has_features()]

    assert  all(all_true)
