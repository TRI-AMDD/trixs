from math import isclose

import numpy as np

from trixs.spectra.spectrum_compare import compare_spectrum, compare_euclidean, compare_pearson
from trixs.spectra.spectrum_io import parse_spectrum

from trixs.spectra.spectrum_compare import get_all_comparisons




def test_compare_euclidean():
    """
    Compare the euclidean distance between two lines.
    :return:
    """

    X1 = np.linspace(0, 1, 1000)
    X2 = np.zeros(1000)

    assert compare_euclidean(X1, X1) == 0.0

    assert compare_euclidean(X2, X2) == 0.0

    assert isclose(compare_euclidean(X1, X2), np.sqrt(np.sum([x ** 2 for x in X1])))


def test_run_comparison_function():
    a = parse_spectrum('./test_files/sample_spectrum_a.txt', skiprows=1)
    b = parse_spectrum("./test_files/sample_spectrum_b.txt", skiprows=1)

    compare_spectrum(a, b, method='pearson')


def test_compare_pearson():
    X1 = np.linspace(0, 1, 100)
    X2 = np.linspace(0, 2, 100)

    assert compare_pearson(X1, X1) == 1.0
    assert compare_pearson(X1, X2) == 1.0
    assert compare_pearson(X2, X2) == 1.0

    X3 = np.linspace(0, -1, 100)

    assert compare_pearson(X1, X3) == -1.0


def test_run_comparison_function_alt_x():
    # a = parse_spectrum('./test_files/sample_spectrum_a.txt', skiprows=1)
    # b = parse_spectrum("./test_files/sample_spectrum_b.txt", skiprows=1)

    c = parse_spectrum('./test_files/sample_spectrum_c.txt', kind='json')
    d = parse_spectrum('./test_files/sample_spectrum_d.txt', kind='json')

    compare_spectrum(c, c, method='pearson', alt_x='Enorm')

    assert compare_spectrum(c, c, method='pearson', alt_x='Enorm') == 1
    assert 0 < compare_spectrum(c, d, method='pearson', alt_x='Enorm') < 1
    assert compare_spectrum(d, c, method='pearson', alt_x='Enorm') == \
           compare_spectrum(d, c, method='pearson', alt_x='Enorm')


def test_all_comparisons():
    a = parse_spectrum('./test_files/sample_spectrum_a.txt', skiprows=1)
    b = parse_spectrum("./test_files/sample_spectrum_b.txt", skiprows=1)

    all_comparisons = get_all_comparisons()

    for key in all_comparisons.keys():
        _ = compare_spectrum(a, b, method=key)
