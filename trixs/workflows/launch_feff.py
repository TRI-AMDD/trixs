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

from pymatgen.io.feff.sets import MPXANESSet

from sys import path
from fireworks.core.rocket_launcher import launch_rocket

import argparse
import os

parser = argparse.ArgumentParser(description='Generate a FEFF workflow.')

parser.add_argument('-c', dest='file', help='Structure to load into workflow')
parser.add_argument('-a', dest='atom', help='Absorbing atom for workflow', default=0)
parser.add_argument('-he', dest='here', help='Launch directory', default=False)
parser.add_argument('-p', dest='prod', help='Run in production mode', default=False)

parser.add_argument('-r', dest='rocket',help='Run a rocket launch right after', default=False)


stock_feff_tags = {"RECIPROCAL": ""}





def get_wf_xas_mod(structure,atom,db_file='>>db_file<<'):

    return get_wf_xas(structure = structure, absorbing_atom=atom,
                    db_file = db_file, feff_cmd='>>feff_cmd<<',
                    radius=10, use_primitive=True,

                      user_tag_settings= ({"RECIPROCAL": "",'RPATH':-1}))



def load_and_launch(args):


    file = args.file
    atom = args.atom
    rocket = args.rocket

    # TODO hack the workflows to allow this
    if args.here:
        spec = {'_launch_dir': os.getcwd()}
    else:
        spec = {}

    if args.prod:
        db_file = '>>db_prod<<'
    else:
        db_file = '>>db_file<<'

    structure = Structure.from_file(file, primitive=True)

    wf = get_wf_xas_mod(structure,atom,db_file)

    lpad = LaunchPad().auto_load()
    lpad.add_wf(wf)

    if args.rocket:
        sys.exit(launch_rocket(lpad))

    else:
        sys.exit()



def launch_from_json_compendium(args, file ='',db_file='>>db_file<<'):
    """
    When pointed to a json file where each line corresponds to a structure,
    launches each structure in sequence as an atomate workflow
    with each absorbing atom targetted.

    :param args:
    :param file:
    :param db_file:
    :return:
    """

    if args.file=='':
        target = file
    else:
        target = args.file

    if target=='':
        raise ValueError("No file given")

    wfs = []

    with open(target,'r') as f:

        cur_struc = Structure.from_file(f.readline())

        absorbing_atoms = list(set(cur_struc.species))

        for atom in absorbing_atoms:
            wfs.append(get_wf_xas_mod(cur_struc,atom,db_file))


    lpad = LaunchPad().auto_load()

    for wf in wfs:
        lpad.add_wf(wf)

    return


def get_wf_full_xas(structure, **kwargs):
    """
    Returns a workflow which computes the full spectrum
    for every atom in a unit cell.
    :param structure:
    :param kwargs:
    :return:
    """
    atoms = list(set(structure.species))

    if len(atoms)==1:
        return get_wf_xas(atoms[0],structure,**kwargs)

    wfs = []
    for atom in atoms:
        wfs.append(get_wf_xas(atom,structure,**kwargs))

    head_wf =  wfs[0]
    for wf in wfs[1:]:
        head_wf.metadata["absorbing_atom_indices"] += (wf.metadata['absorbing_atom_indices'])
        head_wf.append_wf(wf)

    split_name = head_wf.name.split(':')

    head_wf.name = ":".join([split_name[0],split_name[1],'All edges'])

    return head_wf


if __name__==('__main__'):

    lpad = LaunchPad().auto_load()

    # Get the arguments
    args = parser.parse_args()

    load_and_launch(args)




    #if args.rocket:
    #    sys.exit(lpad.)





