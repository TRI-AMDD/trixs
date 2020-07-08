# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

"""
Scripts to plot Spectra using pymatgen methods.
"""

import matplotlib.pyplot as plt
from pymatgen.core.spectrum import Spectrum
from pymatgen.io.feff.outputs import Xmu
from pymatgen.vis.plotters import SpectrumPlotter

from trixs.spectra.spectrum_compare import compare_spectrum


def comparison_plot(inp1='feff1.inp', out1='xmu1.dat',
                    out2='xmu2.dat', inp2='feff2.inp'):
    data1 = Xmu.from_file(out1, inp1)
    data2 = Xmu.from_file(out2, inp2)

    if data1.calc != data2.calc:
        raise TypeError("Spectra are of different "
                        "types: {} and {}".format(data1.calc, data2.calc))

    if data1.calc == 'XANES':
        spec1 = Spectrum(data1.energies, data1.mu)
        spec2 = Spectrum(data2.energies, data2.mu)

    if data1.calc == "EXAFS":
        spec1 = Spectrum(data1.energies, data1.mu0)
        spec2 = Spectrum(data2.energies, data2.mu0)

    plotter = SpectrumPlotter()

    plotter.add_spectrum(spec1, label=spec1.header['TITLE'])
    plotter.add_spectrum(spec2, label=spec2.header['TITLE'])

    cur_plot = plotter.get_plot()

    for comparator in ['pearson', 'euclidean', 'cosine', 'ruzicka']:
        comparison_value = compare_spectrum(spec1, spec2, method=comparator)

        cur_plot.plot([], [], ' ', label='{}:{}'.format(comparator,
                                                        comparison_value))

    plt.legend()
    plt.show()
