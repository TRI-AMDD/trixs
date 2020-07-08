# Copyright 2019-2020 Toyota Research Institute. All rights reserved.

"""
Defines a new XAS Spectrum object built on top of Pymatgen's
Spectrum object.
"""

from typing import Sequence

import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.core.structure import Structure
from trixs.spectra.spectrum_normalize import normalize_minmax, normalize_sum, \
    normalize_z, normalize_max, normalize_l2, normalize_z_max
from trixs.spectra.spectrum_compare import compare_spectrum
from trixs.spectra.util import NumpyEncoder
from json import dumps, loads

from scipy.interpolate import interp1d

from monty.json import MSONable


class XAS_Spectrum(Spectrum):

    def __init__(self, x: Sequence[float], y: Sequence[float],
                 structure: Structure = None,
                 absorbing_site: int = None,
                 absorbing_element: str = None,
                 edge: str = 'K', kind: str = 'XANES',
                 full_spectrum: np.array = None,
                 metadata=None, *args,
                 **kwargs):
        """
        XAS Spectrum object. Contains info about the associated structure,
        absorbing atom, and spectroscopy parameters. A metadata dictionary
        can contain other relevant information, like the MP ID.

        Can pass in x and y if only those are known
        (as they are for MP XAS spectra).
        For atomate spectra, for which all E, E0, k, etc info exists, pass in
        as 'full spectra', in which case x and y will be assigned to E and Mu.

        :param x:
        :param y:
        :param structure:
        :param absorbing_site:
        :param absorbing_element:
        :param edge:
        :param kind:
        :param full_spectrum:
        :param metadata:
        :param args:
        :param kwargs:
        """

        super().__init__(x, y, structure, absorbing_site, edge, kind,
                         *args, **kwargs)

        if structure is not None:
            self.structure = structure
            self.absorbing_site = absorbing_site

        if absorbing_element is None and structure is not None:
            self.absorbing_element = str(structure.species[absorbing_site])
        else:
            self.absorbing_element = absorbing_element

        self.edge = edge
        self.kind = kind

        self.full_spectrum = np.array(full_spectrum) \
            if full_spectrum is not None else None
        self.metadata = metadata

        # Derivative spectra is established when needed
        self._deriv = None

    def __str__(self):
        return f'{self.kind} for {self.structure.formula} ' \
               f'absorbing at {self.absorbing_element}({self.absorbing_site})'

    def normalize(self, method='sum', value=1):
        """
        Normalize the spectrum in-place via the specified method.
        :param method:
        :param value:
        :return:
        """

        if method.lower() == 'sum':
            normalize_sum(self, in_place=True)
        elif method.lower() == 'l2':
            normalize_l2(self, in_place=True)
        elif method.lower() == 'minmax' or method.lower() == 'min_max':
            normalize_minmax(self, in_place=True)
        elif method.lower() == 'z':
            normalize_z(self, in_place=True)
        elif method.lower() == 'max':
            normalize_max(self, in_place=True)
        elif method.lower() == 'zmax':
            normalize_z_max(self, in_place=True)

        else:
            raise NotImplementedError("Normalization type not found.")

    @property
    def dy(self):

        if self._deriv is None:
            self._deriv = np.gradient(self.y, self.x)

        return self._deriv

    @property
    def E(self):
        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 0]

    @property
    def Enorm(self):

        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 1]

    @property
    def k(self):
        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 2]

    @property
    def mu(self):
        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 3]

    @property
    def mu0(self):
        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 4]

    @property
    def chi(self):
        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        return self.full_spectrum[:, 5]

    @property
    def abs_idx(self):
        return self.absorbing_site

    @staticmethod
    def from_atomate_document(document):
        """
        Produces an XAS_Spectrum object from a Document object
        from atomate's fefftoDB object.
        :param document:
        :return:
        """
        spectrum = np.array(document['spectrum'])
        if isinstance(document['structure'], dict):
            structure = Structure.from_dict(document['structure'])
        else:
            structure = document['structure']
        abs_at_idx = document['absorbing_atom']
        # Extract E and Mu
        if 2 in spectrum.shape:
            x = spectrum[0, :]
            y = spectrum[1, :]
        else:
            x = spectrum[:, 0]
            y = spectrum[:, 3]

        spec = XAS_Spectrum(x, y, structure, abs_at_idx,
                            full_spectrum=spectrum)

        spec.metadata = document.get('metadata')

        return spec

    def as_dict(self):

        thedict = super().as_dict()
        return loads(dumps(thedict, cls=NumpyEncoder))

    def as_str(self):
        return dumps(self.as_dict())

    def shifted_Enorm(self, shift):

        if self.full_spectrum is None:
            raise ValueError("No full spectrum available")
        else:
            return np.add(self.Enorm, shift)

    def shifted_x(self, shift):

        return np.add(self.x, shift)

    def has_full_spectrum(self):
        if self.full_spectrum is not None:
            return True
        else:
            return False

    def get_shift_alignment(self, target_spectrum, shift_frac: float = .3,
                            fidelity: int = 20, method='pearson',
                            iterations: int = 5, narrow_factor=0.1, **kwargs):
        """
        Return the x-value by which to shift the first spectrum in order
        to maximize the pearson correlation coefficient with the
        provided spectrum.

        :param method:
        :param iterations:
        :param narrow_factor:
        :param target_spectrum:
        :param shift_frac:
        :param fidelity:
        :param kwargs:
        :return:
        """

        window_size = max(self.x) - min(self.x)

        shift_range = shift_frac * window_size

        prev_shift = 0
        # TODO allow for alternate x values, like E_norm

        for _ in range(iterations):
            shifts = np.linspace(prev_shift - shift_range,
                                 prev_shift + shift_range, fidelity)

            correlations = [compare_spectrum(self, target_spectrum,
                                             method=method,
                                             shift_1x=shift, **kwargs)
                            for shift in shifts]

            best_correlation = max(correlations)
            best_shift = shifts[correlations.index(best_correlation)]

            shift_range = window_size * narrow_factor
            window_size = shift_range
            prev_shift = best_shift

        return best_shift, best_correlation

    def get_peak_idx(self):
        """
        Returns the index along y that the maximum value occurs,
        which we assume to be the edge.
        :return:
        """

        return np.where(self.y == np.max(self.y))[0][0]

    def project_to_x_range(self, proj_x: np.ndarray, alt_x: str=None,
                           alt_y: str=None,
                           pad: bool = True):
        """
        Projects the value of the spectrum onto the values of proj_x using
        cubic interpolation, padding with 0 on the left
        and extrapolating on the right

        :param pad: Pad missing values with 0s. Otherwise extrapolates.
        :param proj_x: The domain of x values to project the spectrum onto
        :param alt_x: Alternate X domain to use
        :param alt_y: Alternate Y values to use
        :return:
        """

        # Get alternate x or y values

        if isinstance(alt_x, str):
            if alt_x.lower() == 'enorm':
                X = self.Enorm
            elif alt_x.lower() == 'k':
                X = self.k
        elif alt_x is None:
            X = self.x
        else:
            X = alt_x

        if isinstance(alt_y, str):

            if alt_y.lower() == 'mu0':
                Y = self.mu0
            elif alt_y.lower() == 'chi':
                Y = self.chi
        elif alt_y is None:
            Y = self.y
        else:
            Y = alt_y

        xmin = min(X)

        # Pad with zeroes on the left

        if pad:
            x_pad = np.array([proj_x[i] for i in range(len(proj_x)) if
                              proj_x[i] < xmin])
            y_pad = np.array([0.0] * len(x_pad))

            # Return interpolation and extrapolate to right if necessary
            func = interp1d(np.concatenate((x_pad, X)),
                            np.concatenate((y_pad, Y)),
                            kind='cubic', fill_value='extrapolate',
                            assume_sorted=True)
        else:
            func = interp1d(X, Y,
                            kind='cubic', fill_value='extrapolate',
                            assume_sorted=True)
        # Occasionally the interpolation returns negative values
        # with the left zero-padding, so manually ceil them to 0
        return np.maximum(func(proj_x.round(8)), 0)

    @staticmethod
    def load_from_object(spectrum):
        """
        Turn either a XAS Spectrum formatted as a string
        into an XAS spectrum object, or a dictionary into an
        XAS Spectrum object (depending on if it was generated by a
        to_dict method or if it is a raw atomate document).
        Does nothing if is already an XAS Spectrum object.

        A convenience method to make interacting with stored XAS
        spectra easier (e.g. load in directly from a line in a file).

        :param spectrum:
        :return:
        """
        # Turn str into dict
        if isinstance(spectrum, str):
            spectrum = loads(spectrum)
        # Turn dict into object
        if isinstance(spectrum, dict):
            if not spectrum.get('x') and spectrum.get('_id'):
                spectrum = XAS_Spectrum.from_atomate_document(spectrum)
            else:
                spectrum['x'] = spectrum['spectrum'][0]
                spectrum['y'] = spectrum['spectrum'][1]
                spectrum = XAS_Spectrum.from_dict(spectrum)
        return spectrum

    def sanity_check(self):
        """
        Ensures that the spectrum satisifies two sanity checks; namely,
        that there are no negative values, and that the spectrum
        is not near 0 (sometimes occurs for failed calculations).
        :return:
        """

        # No negative spectra
        if min(self.y) < -.01:
            return False
        # Catch strange observed failure mode where spectrum is near-0
        # throughout domain
        if np.abs(np.mean(self.y)) < .01:
            return False

        return True

    def broaden_spectrum_mult(self, factor: float):
        """
        Returns a broadened form of the spectrum.

        :param factor: 0 means no change, .05 means 5 percent broadening, etc.
        :return:
        """
        current_domain = (min(self.x), max(self.x))

        L = current_domain[1] - current_domain[0]
        new_domain = (current_domain[0] - factor / 2 * L, current_domain[
            1] + factor / 2 * L)

        self.x = np.linspace(new_domain[0], new_domain[1], 100)

    def broaden_spectrum_const(self, factor: float):
        """
        Returns a broadened form of the spectrum.

        :param factor: 0 means no change, 5 means 5 eV broadening, etc.
        :return:
        """
        current_domain = (min(self.x), max(self.x))

        new_domain = (current_domain[0] - factor / 2, current_domain[
            1] + factor / 2)

        self.x = np.linspace(new_domain[0], new_domain[1], 100)


def _trim_non_alpha(string):
    """
    Used to prune non-alpha numeric characters from the elements
    of bader-charged structures"""
    return ''.join([x for x in string if x.isalpha()])


class XAS_Collation(MSONable):
    def __init__(self, structure: Structure,
                 mp_id=None, oqmd_id=None,
                 icsd_ids=None,
                 mp_spectra: list = None,
                 feff_spectra: list = None,
                 mp_bader: list = None,
                 oqmd_bader: list = None,
                 coordination_numbers=None):
        """
        Object to group together a structure along with associated
        MP and OQMD properties; here we define MP spectra
        as those computed in Kiran Mathew's work
        and FEFF spectra as those computed by us.
        MP oxy and OQMD oxy refer to oxidation numbers
        extracted from Bader charges of the respective databases.

        :param structure:
        :param mp_id:
        :param oqmd_id:
        :param icsd_ids:
        :param mp_spectra:
        :param feff_spectra:
        :param mp_bader:
        :param oqmd_bader:
        """
        self.structure = structure
        self.mp_id = mp_id
        self.oqmd_id = oqmd_id
        self.mp_spectra = [] if mp_spectra is None else mp_spectra
        self.feff_spectra = [] if feff_spectra is None else feff_spectra
        self.icsd_ids = [] if icsd_ids is None else icsd_ids
        self.mp_bader = [] if mp_bader is None else mp_bader
        self.oqmd_bader = [] if oqmd_bader is None else oqmd_bader

        self.elements = set([_trim_non_alpha(str(elt))
                             for elt in structure.species])
        self.associated_ids = {}
        self.coordination_numbers = [] if coordination_numbers is None \
            else coordination_numbers

    def has_mp_bader(self):
        return bool(len(self.mp_bader))

    def has_oqmd_bader(self):
        return bool(len(self.oqmd_bader))

    def has_bader(self):
        return self.has_mp_bader() or self.has_oqmd_bader()

    def has_mp_spectra(self):
        return bool(len(self.mp_spectra))

    def has_feff_spectra(self):
        return bool(len(self.feff_spectra))

    def has_spectra(self):
        return self.has_feff_spectra() or self.has_mp_spectra()

    def has_features(self):
        has_oxy = bool(len(self.mp_bader) + len(self.oqmd_bader))
        has_spec = bool(len(self.mp_spectra) + len(self.feff_spectra))
        return has_oxy and has_spec
