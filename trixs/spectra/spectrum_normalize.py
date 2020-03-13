# coding: utf-8
"""
Scripts to obtain spectra data from the Materials Project database or other sources.

Normalizes a spectrum by a variety of different means.
Norm and minmax use means defined by
Zheng, Mathew, Chen et al, NPJ Comp. Mat. 2018.

Author: Steven Torrisi
Toyota Research Institute 2019
"""

from pymatgen.core.spectrum import Spectrum
import numpy as np
from copy import deepcopy


def normalize_sum(spectrum: Spectrum, value=1.0, in_place: bool = False):
    total = np.sum(spectrum.y)

    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y /= total
    spectrum.y *= value
    return spectrum


def normalize_max(spectrum: Spectrum, in_place: bool = False):
    maximum = np.max(spectrum.y)

    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y /= maximum
    return spectrum


def normalize_z_max(spectrum: Spectrum, in_place: bool = False):
    mu = np.mean(spectrum.y)
    sig = np.std(spectrum.y)
    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y = np.array([(y - mu) / sig for y in spectrum.y])
    spectrum.y /= np.max(spectrum.y)

    return spectrum


def normalize_l2(spectrum: Spectrum, in_place: bool = False):
    sum_norm = np.linalg.norm(spectrum.y, ord=2)

    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y /= sum_norm
    return spectrum


def normalize_minmax(spectrum: Spectrum, in_place: bool = False):
    spec_max, spec_min = max(spectrum.y), min(spectrum.y)

    mins_array = spec_min * np.ones(shape=spectrum.y.shape, dtype='float')

    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y -= mins_array
    spectrum.y /= (spec_max - spec_min)

    return spectrum


def normalize_z(spectrum: Spectrum, in_place: bool = False):
    """
    Normalize by treating the spectrum as a normal distribution and finding the
    'z-score' associated with each point
    :param spectrum:
    :param in_place:
    :return:
    """

    mu = np.mean(spectrum.y)
    sig = np.std(spectrum.y)

    if not in_place:
        spectrum = spectrum.copy()

    spectrum.y = np.array([(y - mu) / sig for y in spectrum.y])

    return spectrum


def normalize_0left_1right(spectrum: Spectrum, in_place: bool = False):
    """
    Normalize by setting the input value to 0 and the final value to about 1.
    :param spectrum: 
    :param in_place: 
    :return: 
    """

    to_sub = spectrum.y[0]

    if not in_place:
        spectrum = spectrum.copy

    spectrum.y -= to_sub

    spectrum_final_avg = np.mean[spectrum.y[-5:]]

    spectrum.y /= spectrum_final_avg

    return spectrum