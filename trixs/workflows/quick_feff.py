# coding: utf-8
"""
Scripts to point-and-shoot create Atomate workflows for structures.

Author: Steven Torrisi
Toyota Research Institute 2019
"""


from fireworks.core.launchpad import LaunchPad
from atomate.feff.workflows import get_wf_xas
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
import json
import sys
from pymatgen.core import Structure

from os import path
from fireworks.core.rocket_launcher import launch_rocket

import argparse
import os

parser = argparse.ArgumentParser(description='Generate a FEFF workflow.')

parser.add_argument('-c', dest='file', help='Structure to load into workflow')
parser.add_argument('-a', dest='atom', help='Absorbing atom for workflow', default=-1)
parser.add_argument('-he', dest='here', help='Launch directory', default=False)
parser.add_argument('-p', dest='prod', help='Run in production mode', default=True)


def quick_feff(args):


    file = args.file
    atom = str(args.atom)
    if args.prod:
        db_file = '>>db_prod<<'
    else:
        db_file = '>>db_file<<'

    structure = Structure.from_file(file, primitive=True)

    cwd = os.getcwd()

    target = path.join(cwd,atom)


    metadata = {'file name:':file,
                'absorbing atom':atom}

    wf = get_wf_xas(structure = structure, absorbing_atom=atom,
                    db_file=db_file, feff_cmd='>>feff_cmd<<',
                    radius=10, use_primitive=True)

    lpad = LaunchPad().auto_load()
    lpad.add_wf(wf)

    os.mkdir(target)
    os.chdir(target)

    sys.exit(launch_rocket(lpad))


if __name__==('__main__'):

    # Get the arguments
    args = parser.parse_args()

    if args.atom == -1:
        raise ValueError("No atom argument specified.")

    quick_feff(args)





