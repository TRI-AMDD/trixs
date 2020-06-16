# coding: utf-8
"""
Contains scripts which help to process spectra from experiment to make it easily comparable.

Some routines below draw from the methods section of
Zheng, Mathew, Chen et al, NPJ Comp. Mat. 4, 12 (2018)
( https://doi.org/10.1038/s41524-018-0067-x )
Other implementations can be found in the Veidt software code by Materials
Virtual Lab : https://github.com/materialsvirtuallab/veidt

Methods below are based upon implementations in particular from
https://github.com/materialsvirtuallab/veidt/blob/master/veidt/elsie
/spectra_similarity.py .

Author: Steven Torrisi
Toyota Research Institute 2019
"""

from math import sqrt, log

import numpy as np
from pymatgen.core.spectrum import Spectrum
from scipy.interpolate import interp1d


class Spectral_Comparer(object):

    def __init__(self):
        raise NotImplementedError


def determine_overlap_spectra(x1, y1, x2, y2, only_in_overlap=True):
    """
    Given two x and y spectra, 
    :param only_in_overlap:
    :param x1:
    :param x2: 
    :param y1: 
    :param y2: 
    :return: 
    """

    xmin1, xmax1 = min(x1), max(x1)
    xmin2, xmax2 = min(x2), max(x2)

    overlap_range = (max([xmin1, xmin2]), min([xmax1, xmax2]))

    overlap_energies = []

    for x in list(x1) + list(x2):
        if overlap_range[0] <= x <= overlap_range[1]:
            overlap_energies.append(x)

    if len(overlap_energies) == 0:
        raise ValueError(
            "No overlap energies detected between spectra with min/max energies of "
            "{},{} vs {},{}. ".format(xmin1, xmax1, xmin2, xmax2))

    # Prune any repeats by turning into set then sort
    overlap_energies = sorted(list(set(overlap_energies)))

    # Interpolation has issues when sampling from energy outside of the x-range
    # of each spectrum; prune energies which lie outside
    if overlap_energies[0] != x1[0] and overlap_energies[0] != x2[0]:
        overlap_energies.pop(0)
    if overlap_energies[-1] != x1[-1] and overlap_energies[-1] != x2[-1]:
        overlap_energies.pop(-1)

    y1_func = interp1d(x1, y1)
    y2_func = interp1d(x2, y2)

    y1_interp = y1_func(overlap_energies)
    y2_interp = y2_func(overlap_energies)

    if only_in_overlap:
        Y1 = y1_interp
        Y2 = y2_interp
    else:
        raise NotImplementedError

    return Y1, Y2


def compare_spectrum(spec1: Spectrum, spec2: Spectrum, method: str,
                     only_in_overlap=True,
                     alt_x='', alt_y='', shift_1x: float = 0,
                     shift_2x: float = 0):
    """
    Compare two spectra and return a measure of similarity.

    :param shift_1x:
    :param shift_2x:
    :param alt_x:
    :param alt_y:
    :param only_in_overlap:
    :param spec1:
    :param spec2:
    :param method:
    :return:
    :rtype:
    """

    if alt_x.lower() == 'enorm':
        s1x = spec1.Enorm
        s2x = spec2.Enorm
    else:
        s1x = spec1.x
        s2x = spec2.x

    if alt_y.lower() == 'mu0':
        s1y = spec1.mu0
        s2y = spec2.mu0

    elif alt_y.lower() == 'chi':
        s1y = spec1.chi
        s2y = spec2.chi
    else:
        s1y = spec1.y
        s2y = spec2.y

    if shift_1x or shift_2x:
        s1x = np.subtract(s1x, shift_1x)
        s2x = np.subtract(s2x, shift_2x)

    Y1, Y2 = determine_overlap_spectra(s1x, s1y, s2x, s2y, only_in_overlap)

    func = get_all_comparisons().get(method.lower(), None)

    if method.lower() == 'all':
        metrics = sorted(list(get_all_comparisons().keys()))

        return [get_all_comparisons()[key](Y1, Y2) for key in metrics], metrics


    elif func is None:
        raise ValueError("Warning! The method specified was not in the list of"
                         "valid comparison metrics.")
    else:
        return func(Y1, Y2)


def compare_pearson(X, Y):
    mux = np.mean(X)
    muy = np.mean(Y)

    # Compute first moments

    Mom1 = [x - mux for x in X]
    Mom2 = [y - muy for y in Y]

    numerator = sum([mo1 * mo2 for mo1, mo2 in zip(Mom1, Mom2)])
    denominator = sqrt(
        sum([mo1 ** 2 for mo1 in Mom1]) * sum([mo2 ** 2 for mo2 in Mom2]))

    return numerator / denominator


def compare_euclidean(X, Y):
    return sqrt(sum(np.power((X - Y), 2)))


# The following are drawn from
# https://apps.dtic.mil/dtic/tr/fulltext/u2/1026967.pdf

# Minkowski Family

def compare_city_block(X, Y):
    return sum(abs(X - Y))


def compare_city_block_avg(X, Y):
    assert len(X) == len(Y)
    return sum(abs(X - Y)) / len(X)


def compare_chebyshev(X, Y):
    return max(abs(X - Y))


# L1 Family

def compare_sorensen(X, Y):
    num = sum(abs(X - Y))
    denom = sum((X + Y))
    return num / denom


def compare_soergel(X, Y):
    num = sum(abs(X - Y))
    denom = sum([max(x, y) for x, y in zip(X, Y)])
    return num / denom


def compare_kulczynski(X, Y):
    num = sum(abs(X - Y))
    denom = sum([min(x, y) for x, y in zip(X, Y)])
    return num / denom


def compare_canberra(X, Y):
    return sum([abs(x - y) / (x + y) for x, y in zip(X, Y)])


def compare_lorentzian(X, Y):
    return sum(np.log([1 + abs(x - y) for x, y in zip(X, Y)]))


# Intersection Family

def compare_intersection(X, Y):
    return np.sum(np.abs(X - Y)) / 2


def compare_wave_hedges(X, Y):
    return sum([abs(x - y) / max(x, y) for x, y in zip(X, Y)])


def compare_czekanowski(X, Y):
    num = sum([max(x, y) for x, y in zip(X, Y)])
    denom = sum([x + y for x, y in zip(X, Y)])
    return num / denom


def compare_ruzicka(X, Y):
    numerator = sum([min([x, y]) for x, y in zip(X, Y)])
    denominator = sum([max([x, y]) for x, y in zip(X, Y)])
    return numerator / denominator


# Inner Product Family

def compare_ip(X, Y):
    return np.dot(X, Y)


def compare_harmonic_mean(X, Y):
    num = 2 * np.dot(X, Y)
    denom = sum([x + y for x, y in zip(X, Y)])
    return num / denom


def compare_cosine(X, Y):
    numerator = sum([x * y for x, y in zip(X, Y)])
    denominator = sqrt(sum([x ** 2 for x in X])) * sqrt(
        sum([y ** 2 for y in Y]))
    return numerator / denominator


def compare_humar_hassebrook(X, Y):
    numerator = np.dot(X, Y)
    denominator = sum([x ** 2 + y ** 2 - x * y for x, y in zip(X, Y)])
    return numerator / denominator


def compare_dice(X, Y):
    return 2 * np.dot(X, Y) / sum([x ** 2 + y ** 2 for x, y in zip(X, Y)])


# Fidelity Family

def compare_fidelity(X, Y):
    return np.sum([sqrt(abs(x * y)) for x, y in zip(X, Y)])


def compare_bhattacharyya(X, Y):
    return -np.log(compare_fidelity(X, Y))


def dist_squared_chord(X, Y):
    return sum([(sqrt(abs(x)) - sqrt(abs(y))) ** 2 for x, y in zip(X, Y)])


def compare_hellinger(X, Y):
    return sqrt(2 * dist_squared_chord(X, Y))


def compare_matusita(X, Y):
    return sqrt(dist_squared_chord(X, Y))


def compare_squared_chord(X, Y):
    return 2 * dist_squared_chord(X, Y) - 1


# Chi Squared Family

def compare_squared_euclidean(X, Y):
    return np.sum(np.power(X - Y, 2))


def compare_squared_chisq(X, Y):
    return sum([(x - y) ** 2 / (x + y) for x, y in zip(X, Y)])


def compare_divergence(X, Y):
    return 2 * sum([((x - y) / (x + y)) ** 2 for x, y in zip(X, Y)])


def compare_clark(X, Y):
    return sqrt(sum([(abs(x - y) / (x + y)) ** 2 for x, y in zip(X, Y)]))


def compare_additive(X, Y):
    return sum([(x - y) ** 2 * (x + y) / (x * y) for x, y in zip(X, Y)])


# Shannon Entropy Family

def compare_kullback_leibler(X, Y):
    return sum([x * log(abs(x / y)) for x, y in zip(X, Y)])


def compare_jeffreys(X, Y):
    return sum([(x - y) * log(abs(x / y)) for x, y in zip(X, Y)])


def compare_jensen_difference(X, Y):
    return sum([(x * log(abs(x)) + y * log(abs(y))) / 2 - (x + y) / (
                2 * log((abs(x + y)) / 2)) for x, y in zip(X, Y)])


# Combination Family

def compare_taneja(X, Y):
    return sum(
        [(x + y) / (2 * log((abs(x + y)) / (2 * sqrt(abs(x * y))))) for x, y in
         zip(X, Y)])


def compare_kumar_johnson(X, Y):
    # Added tiny numerical stability value
    return sum(
        [(x ** 2 - y ** 2) ** 2 / (2 * x * y + 1E-5) ** 1.5 for x, y in
         zip(X, Y)])


def compare_l1_linf(X, Y):
    max_diff = max([abs(x - y) for x, y in zip(X, Y)])
    return .5 * sum([abs(x - y) + max_diff for x, y in zip(X, Y)])


# Vicissitude Family
# Not yet implemented: https://apps.dtic.mil/dtic/tr/fulltext/u2/1026967.pdf


def compare_max_index_diff(X, Y):
    return np.amax(X) - np.amax(Y)


def get_all_comparisons():
    return {'pearson': compare_pearson,
            'euclidean': compare_euclidean,
            'city_block': compare_city_block,
            'city_block_avg': compare_city_block_avg,
            'chebyshev': compare_chebyshev,
            'sorenson': compare_sorensen,
            'soergel': compare_soergel,
            'kulczynski': compare_kulczynski,
            'canberra': compare_canberra,
            'lorentzian': compare_lorentzian,
            'intersection': compare_intersection,
            'wave_hedges': compare_wave_hedges,
            'czekanowski': compare_czekanowski,
            'ruzicka': compare_ruzicka,
            'ip': compare_ip,
            'harmonic_mean': compare_harmonic_mean,
            'cosine': compare_cosine,
            'humar_hassebrook': compare_humar_hassebrook,
            'dice': compare_dice,
            'fidelity': compare_fidelity,
            'bhattacharrya': compare_bhattacharyya,
            'squared_chord': compare_squared_chord,
            'hellinger': compare_hellinger,
            'matusita': compare_matusita,
            'squared_euclidean': compare_squared_euclidean,
            'squared_chisq': compare_squared_chisq,
            'divergence': compare_divergence,
            'clark': compare_clark,
            'additive': compare_additive,
            'kullbeck_leibler': compare_kullback_leibler,
            'jeffreys': compare_jeffreys,
            'jensen_difference': compare_jensen_difference,
            'taneja': compare_taneja,
            'kumar_johnson': compare_kumar_johnson,
            'l1_linf': compare_l1_linf,
            'max_index': compare_max_index_diff}
