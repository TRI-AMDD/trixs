# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

from math import isclose

import numpy as np

from trixs.spectra.spectrum_featurize import polynomialize_by_idx,\
                                        gauge_polynomial_error


def test_polynomialize_by_idx_simple():

    X = np.linspace(-.1,.1,100)
    Y = np.sin(X)

    N = 1

    polys_quad = polynomialize_by_idx(X=X,Y=Y,N=N,deg=2,label_type='size')


    assert len(polys_quad)==N

    coefs = polys_quad[0].coef

    assert np.isclose(coefs[0], 0.0)
    assert np.isclose(coefs[1], .1,atol=1e-2)
    assert np.isclose(coefs[2], 0.0)

    assert 'deg' in polys_quad[0].label
    assert 'step_size' in polys_quad[0].label


    polys_cub = polynomialize_by_idx(X=X,Y=Y,N=N,deg=3,label_type='frac')

    coefs = polys_cub[0].coef
    assert len(coefs) == 4
    assert 'fraction_size' in polys_cub[0].label


def test_polynomialize_by_idx_complex():

    X = np.linspace(0,1,100)
    Y = np.array([(x-.5)**2 for x in X])

    polys = polynomialize_by_idx(X,Y,N=2,deg=2)

    coefs1 = polys[0].coef
    coefs2 = polys[1].coef

    assert np.isclose(coefs1[0], coefs2[0])
    assert np.isclose(coefs1[1], -coefs2[1])
    assert np.isclose(coefs1[2], coefs2[2])

    assert np.isclose(polys[0](X[:50]), Y[:50]).all()
    assert np.isclose(polys[1](X[50:]), Y[50:]).all()

    abs_ers = gauge_polynomial_error(X[:50],Y[:50],polys[0],error='abs')
    assert np.isclose(abs_ers, np.zeros(50)).all()

    sq_ers = gauge_polynomial_error(X[:50],Y[:50],polys[0],error='squared')
    assert np.isclose(sq_ers, np.zeros(50)).all()

