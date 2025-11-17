# Script to check the dipole moments cube wrt the fchk file

import sys
import numpy as np
import os
import argparse
import re
from estampes.data.physics import phys_fact
from tcdlibx.calc.cube_manip import VtcdData, CubeData, cube_parser
from tcdlibx.graph.helpers import VibMolecule
from tcdlibx.io.estp_io import get_vibmol
#from tcdlibx.utils.conversion_units import edip_cgs, mdip_cgs, ele_mdip_cgs, ele_edip_cgs

# FIXME: check the exceptions? leave it to the gui? 
def open_fchk(fname: str) -> VibMolecule:
    """Open the fchk file and return the molecule object."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f'File {fname} not found.')
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
    parser.add_argument('cub', type=str, help='cube template name')
    parser.add_argument('--fout', type=str, default="vcd_test.txt", help='output filename')
    parser.add_argument('--legacy', action="store_true", help='sqrt factor for old cube')
    return parser

def main():
    """Main function to check the dipole moments cube wrt the fchk file."""
    parser = build_parser()
    args = parser.parse_args()
    fchk = open_fchk(args.fchk)
    tmplname = args.cub
    fout = args.fout
    res = re.findall('(X+)', tmplname) 
    if not res:
        print("ERROR: Missing X's in filename pattern")
        sys.exit()
    if len(res) > 1:
        print("ERROR: More than 1 sequence of X's found.")
        sys.exit()
    tmplfmt = tmplname.replace(res[0], '{{:0{:d}d}}'.format(len(res[0])))
    
    # Write header to output file before starting calculations
    with open(fout, 'w') as f:
        f.write("DTM Analysis Results\n")
        f.write("=" * 131 + "\n")
        f.write("N. {:^8s}".format("Freq."))
        f.write("{:^30s}".format("MFP (electric)"))
        f.write("{:^30s}".format("MFP (magnetic)"))
        f.write("{:^30s}".format("TCD_tot (electric)"))
        f.write("{:^30s}".format("TCD_tot (magnetic)") + "\n")
        f.write("-" * 131 + "\n")
    
    for i in range(1, fchk.ntrans + 1):
        fname = tmplfmt.format(i)
        if not os.path.exists(fname):
            print(f"ERROR: File {fname} not found.")
            sys.exit()
        tmp_cube = open_cube(fname, legacy=args.legacy)
        tcddata = VtcdData(tmp_cube,
                           fchk._moldata['evec'][i-1],
                           fchk._moldata['freq'][i-1])
        fchk.add_tcd(i-1, tcddata)
        nuc_cntr = fchk.get_dtm(i-1, tps='nuc', cgs=False)
        mfp_dtm = fchk.get_dtm(i-1, tps='tot', cgs=False)
        print(f"State {i}: Freq = {fchk._moldata['freq'][i-1]:.2f} cm^-1")
        print("MFP DTM (ele): {:10.5f} {:10.5f} {:8.4f}".format(mfp_dtm[0][0]-nuc_cntr[0][0], mfp_dtm[0][1]-nuc_cntr[0][1], mfp_dtm[0][2]-nuc_cntr[0][2]))
        print("MFP DTM (mag): {:10.5f} {:10.5f} {:8.4f}".format(mfp_dtm[1][0]-nuc_cntr[1][0], mfp_dtm[1][1]-nuc_cntr[1][1], mfp_dtm[1][2]-nuc_cntr[1][2]))
        tcd_dtm = np.array(fchk.get_tcd_dtm(i-1, cgs=False))
        tcd_dtm /= np.sqrt(fchk._moldata['rmas'][i-1])
        # freq_au = fchk._moldata['freq'][i-1] / phys_fact("au2cm1")
        # tcd_dtm[0] = tcd_dtm[0] / freq_au
        tcd_dtm[0] = tcd_dtm[0]
        print("TCD DTM (ele): {:10.5f} {:10.5f} {:8.4f}".format(tcd_dtm[0][0], tcd_dtm[0][1], tcd_dtm[0][2]))
        print("TCD DTM (mag): {:10.5f} {:10.5f} {:8.4f}".format(tcd_dtm[1][0], tcd_dtm[1][1], tcd_dtm[1][2]))
        tcd_tot = (nuc_cntr[0]+tcd_dtm[0], nuc_cntr[1]+tcd_dtm[1])
        print("MFP DS:{:10.5f}".format(np.dot(mfp_dtm[0],mfp_dtm[0])))
        print("TCD DS:{:10.5f}".format(np.dot(tcd_tot[0],tcd_tot[0])))
        print("MFP RS:{:10.5f}".format(np.dot(mfp_dtm[0],mfp_dtm[1])))
        print("TCD RS:{:10.5f}".format(np.dot(tcd_tot[0],tcd_tot[1])))
        
        # Print results for each state to output file
        with open(fout, 'a') as f:
            f.write(f"{i:3d}")
            f.write(f"{fchk._moldata['freq'][i-1]:8.2f}")
            f.write(f"{mfp_dtm[0][0]:10.5f} {mfp_dtm[0][1]:10.5f} {mfp_dtm[0][2]:8.4f}")
            f.write(f"{mfp_dtm[1][0]:10.5f} {mfp_dtm[1][1]:10.5f} {mfp_dtm[1][2]:8.4f}")
            f.write(f"{tcd_tot[0][0]:10.5f} {tcd_tot[0][1]:10.5f} {tcd_tot[0][2]:8.4f}")
            f.write(f"{tcd_tot[1][0]:10.5f} {tcd_tot[1][1]:10.5f} {tcd_tot[1][2]:10.5f}\n")
        fchk.remove_tcd(i-1)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        sys.exit(1)
