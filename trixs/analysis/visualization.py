# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

from pymatgen.io.ase import AseAtomsAdaptor
from ase.build.supercells import make_supercell
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from nglview import show_ase
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from trixs.spectra.spectrum_compare import compare_spectrum


def visualize_feff_supercell(XASSpectrum, supercell: int = 0):
    # Turns to ASE, colors the absorbing atom white, and then generates a supercell and returns

    structure = XASSpectrum.structure
    abs_at_idx = XASSpectrum.absorbing_site

    sa = SpacegroupAnalyzer(XASSpectrum.structure)
    equivs = sa.get_symmetry_dataset()['equivalent_atoms']
    eq_at_idx = equivs[abs_at_idx]
    N = len(structure.species)
    equiv_idxs = [n for n in range(N) if equivs[n] == eq_at_idx]

    ase_struc = AseAtomsAdaptor().get_atoms(structure)

    for idx in equiv_idxs:
        ase_struc.numbers[idx] = 1

    if supercell:
        ase_struc = make_supercell(ase_struc, supercell * np.eye(3))
    elif len(ase_struc.numbers) < 6:
        ase_struc = make_supercell(ase_struc, 3 * np.eye(3))
    elif len(ase_struc.numbers) < 12:
        ase_struc = make_supercell(ase_struc, 2 * np.eye(3))
    print("Absorbing atom element: {}".format(structure.species[abs_at_idx]))
    return show_ase(ase_struc)


def generate_distance_matrix(spectra, method='pearson', imshow_kwargs={},
                             compare_kwargs={}):
    N = len(spectra)
    distance_matrix = np.empty(shape=(N, N))

    for i, spec1 in enumerate(tqdm(spectra)):
        for j, spec2 in enumerate(spectra):
            if i == j:
                distance_matrix[i][j] = 1.0
            elif j < i:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = compare_spectrum(spec1, spec2, method=method, **compare_kwargs)

    plt.imshow(distance_matrix, **imshow_kwargs)

    plt.colorbar()
    plt.title('{} similarity for {} spectra'.format(method, spectra[0].absorbing_element))
    plt.show()
