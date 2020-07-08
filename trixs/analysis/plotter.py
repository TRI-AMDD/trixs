# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as mplcm
import matplotlib.patches as mpatches
from typing import List
import matplotlib.pyplot as plt


class SpectrumPlotter(object):

    def __init__(self):
        raise NotImplementedError


def plot_spectra_by_formulae(spectra: List, alt_x='', alt_y='',
                             exclude_elts: set = set([]),
                             title=''):
    ## Determine which spectra to plot
    to_plot = []
    exclusion_set = set(exclude_elts)
    for spec in spectra:

        cur_struc = spec.structure
        cur_elts = set([str(elt) for elt in spec.structure.species])
        if exclusion_set.intersection(cur_elts):
            continue

        to_plot.append(spec)

    formulae = list(set([x.structure.formula for x in to_plot]))

    # print('Presently represented formulae:',formulae)
    N_colors = len(formulae)
    formulae = sorted(formulae, key=lambda x: len(x))

    cm = plt.get_cmap('hsv')

    cNorm = colors.Normalize(vmin=0, vmax=N_colors)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    thecolors = [scalarMap.to_rgba(i) for i in range(N_colors)]
    color_dict = {formula: thecolors[n] for n, formula in enumerate(formulae)}

    plt.figure(figsize=(8, 6), dpi=300)
    for spec in to_plot:
        spec.normalize()
        if alt_x.lower() == 'enorm':
            X = spec.Enorm
        else:
            X = spec.x
        if alt_y.lower() == 'chi':
            Y = spec.chi
        elif alt_y.lower() == 'mu0':
            Y = spec.mu0
        else:
            Y = spec.y

        plt.plot(X, Y, label=spec.structure.formula,
                 color=color_dict[spec.structure.formula], lw=1)

    if title == '':
        title = spectra[0].absorbing_element
    plt.title(title)

    patches = []
    for formula in formulae:
        patches.append(mpatches.Patch(color=color_dict[formula], label=formula))
    plt.legend(handles=patches)
    plt.show()
