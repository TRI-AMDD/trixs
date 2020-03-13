# coding: utf-8
"""
Featurize spectra by fitting into a set of polynomials or other techniques
"""

from math import sqrt, log
from scipy.interpolate import interp1d

import numpy as np
from pymatgen.core.spectrum import Spectrum
import math
from trixs.spectra.core import XAS_Spectrum


class XAS_Polynomial(object):

    def __init__(self, coeffs, x=None, y=None):
        """

        :param coeffs: Coefficient array
        :param x: Store domain
        :param y: Store range
        """
        self.coeffs = coeffs
        self.x = x
        self.y = y


def gauge_polynomial_error(X, Y, poly, error='abs'):
    diff = Y - poly(X)

    if error.lower() == 'abs':
        errors = np.sum(np.abs(diff))
    if error.lower() == 'percent':
        errors = np.sum(np.abs((Y-poly(X))/Y))
    if error.lower() == 'squared':
        errors = np.power(diff, 2)

    return errors


def polynomialize_by_idx(X, Y, N, deg=2, label_type='size', **kwargs):
    """

    :param X: X domain
    :param Y:  Y domain
    :param N: Number of splits over domain to make
    :param deg: degree of polynomial to use
    :param label_type: Choose from two different kinds of coefficient label
    :return:
    """
    n_x = len(X)
    step = n_x // N

    domain_splits = list(range(0, n_x, step))
    if domain_splits[-1] != n_x - 1:
        domain_splits.append(n_x)

    polynomials = []

    # Perform fits
    for i in range(N):
        l = domain_splits[i]
        r = domain_splits[i + 1]
        x = X[l:r]
        y = Y[l:r]


        cur_poly, full_data = np.polynomial.Polynomial.fit(x, y, deg=deg, full=True, **kwargs)
        cur_poly.full_data = full_data

        cur_poly.x = x
        cur_poly.y = y
        polynomials.append(cur_poly)

        if label_type == 'frac':
            cur_poly.label = "deg:{},fraction_size:{},chunk:{}".format(deg, N, i)
        else:
            cur_poly.label = "deg:{},step_size:{},chunk:{}".format(deg, r - l, i)

    return polynomials

