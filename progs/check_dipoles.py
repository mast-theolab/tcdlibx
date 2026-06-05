# Script to check the dipole moments cube wrt the fchk file

import sys
import numpy as np
import os
import argparse
import typing as tp
# from estampes.data.physics import phys_fact
from tcdlibx.calc.cube_manip import VecCubeData, VtcdData, CubeData, cube_parser
from tcdlibx.graph.helpers import EleMolecule, VibMolecule
from tcdlibx.io.estp_io import get_elemol, get_vibmol
from tcdlibx.utils.conversion_units import edip_cgs, mdip_cgs, ele_mdip_cgs, ele_edip_cgs # , MWQ2q

# FIXME: check the exceptions? leave it to the gui? 
def open_fchk(fname: str) -> tp.Union[EleMolecule, VibMolecule]:
    """Open the fchk file and return the molecule object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
    try:
        fchk = EleMolecule(get_elemol(fname))
    except IndexError:
        # try:
        fchk = VibMolecule(get_vibmol(fname))
        # Fix properly the exceptions
        # except Exception as err: print(err)

    # except Exception as err: print(err)
    return fchk


def open_cube(fname: str, legacy: bool) -> CubeData:
    """Open the cube file and return the cube object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
    elements = True
    if legacy:
        elements = False
    cubdata = cube_parser(fname, elements)
    return cubdata


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description='Check the dipole moments cube wrt the fchk file.')
    parser.add_argument('fchk', type=str, help='fchk file')
    parser.add_argument('cub', type=str, help='cube file')
    parser.add_argument('--state', type=int, default=1, help='Electronic or vibrational state of the cube file')
    parser.add_argument('--legacy', action="store_true", help='sqrt factor for old cube')
    return parser

def main():
    """Main function to check the dipole moments cube wrt the fchk file."""
    parser = build_parser()
    args = parser.parse_args()
    fchk = open_fchk(args.fchk)
    cub = open_cube(args.cub, legacy=args.legacy)
    moltype = None
    if args.state > fchk.ntrans or args.state < 1:
        print(f"State {args.state} not valid. Choose a state between 1 and {fchk.ntrans}")
    state = args.state - 1
    if isinstance(fchk, EleMolecule):
        cubdata = VecCubeData(cub)
        moltype = 'ele'
    else:
        cubdata = VtcdData(cub,
                           fchk._moldata['evec'][state],
                           fchk._moldata['freq'][state])
        moltype = 'vib'
    fchk.add_tcd(state, cubdata)

    print(f"Fchk file: {args.fchk}")
    print(f"Cube file: {args.cub}")
    print(f"State: {args.state}")
    # check only the electronic component
    fchk_dip = fchk.get_dtm(state, tps='ele', cgs=False)
    # fchk_dip_tot = fchk.get_dtm(state, tps='tot', cgs=False)
    cub_dip = fchk.get_tcd_dtm(state, cgs=False)
    if moltype == 'ele':
        fchk_dip_cgs = (ele_edip_cgs(fchk_dip[0]), ele_mdip_cgs(fchk_dip[1]))
        cub_dip_cgs = (ele_edip_cgs(cub_dip[0]), ele_mdip_cgs(cub_dip[1]))
    else:
        # convert velocity edtm to length
        cub_dip = list(cub_dip)
        # print(cub_dip[0], fchk._moldata['freq'][state])
        # cub_dip[0] = cub_dip[0] / np.sqrt(fchk._moldata['freq'][state] /phys_fact("au2cm1"))
        # cub_dip[0] = cub_dip[0] / (np.sqrt(fchk._moldata['rmas'][state])) / MWQ2q # *1822.888486209))  # convert to mass-weighted
        # cub_dip[1] = cub_dip[1] / (np.sqrt(fchk._moldata['rmas'][state])) / MWQ2q # *1822.888486209))  # convert to mass-weighted
        fchk_dip_cgs = (edip_cgs(fchk_dip[0], fchk._moldata['freq'][state]), mdip_cgs(fchk_dip[1], fchk._moldata['freq'][state]))
        # fchk_dip_tot_cgs = (edip_cgs(fchk_dip_tot[0], fchk._moldata['freq'][state]), mdip_cgs(fchk_dip_tot[1], fchk._moldata['freq'][state]))
        cub_dip_cgs = (edip_cgs(cub_dip[0], fchk._moldata['freq'][state]), mdip_cgs(cub_dip[1], fchk._moldata['freq'][state]))

    # print(f"DS: {np.dot(fchk_dip_tot_cgs[0], fchk_dip_tot_cgs[0])} RS: {np.dot(fchk_dip_tot_cgs[0], fchk_dip_tot_cgs[1])}")


    print(f"fchk EDTM: {fchk_dip[0][0]:10.5f}{fchk_dip[0][1]:10.5f}{fchk_dip[0][2]:10.5f}")
    print(f"cube EDTM: {cub_dip[0][0]:10.5f}{cub_dip[0][1]:10.5f}{cub_dip[0][2]:10.5f}")
    print(f"diff EDTM: {fchk_dip[0][0]-cub_dip[0][0]:10.5f}{fchk_dip[0][1]-cub_dip[0][1]:10.5f}{fchk_dip[0][2]-cub_dip[0][2]:10.5f}")
    print(f"fchk MDTM: {fchk_dip[1][0]:10.5f}{fchk_dip[1][1]:10.5f}{fchk_dip[1][2]:10.5f}")
    print(f"cube MDTM: {cub_dip[1][0]:10.5f}{cub_dip[1][1]:10.5f}{cub_dip[1][2]:10.5f}")
    print(f"diff MDTM: {fchk_dip[1][0]-cub_dip[1][0]:10.5f}{fchk_dip[1][1]-cub_dip[1][1]:10.5f}{fchk_dip[1][2]-cub_dip[1][2]:10.5f}")
    print("Dipole strength and rotational strength in a.u.:")
    print("{:10s}{:^15s}{:^15s}{:^15s}".format("","fchk","cube","difference"))
    print("{:10s}{:15.5f}{:15.5f}{:15.5f}".format("DS", np.dot(fchk_dip[0], fchk_dip[0]), np.dot(cub_dip[0], cub_dip[0]), np.dot(fchk_dip[0], fchk_dip[0])-np.dot(cub_dip[0], cub_dip[0])))
    print("{:10s}{:15.5f}{:15.5f}{:15.5f}".format("RS", np.dot(fchk_dip[0], fchk_dip[1]), np.dot(cub_dip[0], cub_dip[1]), np.dot(fchk_dip[0], fchk_dip[1])-np.dot(cub_dip[0], cub_dip[1])))
    print("Dipole strength and rotational strength in cgs:")
    print("{:10s}{:^15s}{:^15s}{:^15s}".format("","fchk","cube","difference"))
    print("{:10s}{:15.5E}{:15.5E}{:15.5E}".format("DS", np.dot(fchk_dip_cgs[0], fchk_dip_cgs[0]), np.dot(cub_dip_cgs[0], cub_dip_cgs[0]), np.dot(fchk_dip_cgs[0], fchk_dip_cgs[0])-np.dot(cub_dip_cgs[0], cub_dip_cgs[0])))
    print("{:10s}{:15.5E}{:15.5E}{:15.5E}".format("RS", np.dot(fchk_dip_cgs[0], fchk_dip_cgs[1]), np.dot(cub_dip_cgs[0], cub_dip_cgs[1]), np.dot(fchk_dip_cgs[0], fchk_dip_cgs[1])-np.dot(cub_dip_cgs[0], cub_dip_cgs[1])))
    
if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        sys.exit(1)
