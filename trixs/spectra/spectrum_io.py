# coding: utf-8

from pymatgen.core.spectrum import Spectrum
from pymatgen.io.feff.outputs import Xmu
from numpy import loadtxt

from json import loads
from trixs.spectra.core import XAS_Spectrum


def parse_spectrum(output: str, feff_inp: str = '', kind='2dcsv', *args, **kwargs):
    """

    Open a spectrum file with x as the first column
    and y as the second

    :param feff_inp:
    :param output:
    :param kind: default, 2-dimensional comma separated values
    :return:
    """

    if kind == '2dcsv':
        values = loadtxt(fname=output, *args, **kwargs)

        return XAS_Spectrum(x=values[:, 1], y=values[:, 0])

    if kind == '2dcsv_spec':
        values = loadtxt(fname=output, *args, **kwargs)

        return XAS_Spectrum(x=values[:, 0], y=values[:, 1])

    if kind == 'feff_mu':
        xmu = Xmu.from_file(output, feff_inp)

        return XAS_Spectrum(x=xmu.energies, y=xmu.mu)

    if kind == 'json':

        line = open(output, 'r').readline().strip()

        xas = loads(line)

        return XAS_Spectrum.from_dict(xas)

    else:
        raise NotImplementedError("Type of data file not found")


def load_XYs(file: str, as_str: bool = True):
    """
    Open up a set of json files which represent XY pairs of spectra
    :param file:
    :param as_str: Store as string instead of dictionary (cheaper memory wise)
    :return:
    """

    collected_spectra = []
    with open(file, 'r') as f:
        cur_str = f.readline()
        while cur_str:
            collected_spectra.append(cur_str if as_str else loads(cur_str))
    return collected_spectra

