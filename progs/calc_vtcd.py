#!/bin/python3
"""
Es: python3 plot_ETCD_quiv2D.py -s '2+9' -u 300 allene_F2.anh.2nq.GVPT2.full.fchk\
        TCD:TCD/allenef2_TCD_SXXX.cube\
        NAC:new_phase/allene_F2_SXXX.anh.2nq.GVPT2.full.log\
        EXC:excstate/allene_F2_NACXXX.fchk
:author: m.fuse, j.bloino
"""
import sys
import os
import re
import argparse
from math import ceil
import numpy as np
# import custom library
from estampes.data.physics import PHYSFACT
import tcdlibx.calc.cube_manip as cb
# from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.color_out import Colors
import tcdlibx.io.fchk_io as fio
from tcdlibx.utils.var_tools import get_ivib, print_lines
# import variabiles
from tcdlibx.utils.mol_data import ELEMENTS

PROGNAME = os.path.basename(sys.argv[0])
NSTEP_BOX = 1
# SCALE_ARROW = 1.0e9
SCALE_ARROW = None
# Nessun prefattore applicato
# Vedi definizione eq6 in pccp
PRE_FACT = 1


def build_parser():
    """
    Build options parser.
    """
    par = argparse.ArgumentParser(prog=PROGNAME,
                                  formatter_class=argparse.RawTextHelpFormatter)
    # MANDATORY ARGUMENTS
    txt = "Gaussian fchk with data on state of interest"
    par.add_argument('refstate', help=txt)
    txt = """Various name templates for data of interest.
The part to replace with the electronic state index are indicated with "X"'s
Example: cube_SXX.cube
In order:
- transition current density (TCD)
- non-adiabatic couplings (NAC)
- excited-states electronic data (EXC)"""
    par.add_argument('tmplfiles', nargs='+', help=txt)
    # OPTIONAL ARGUMENTS
    par.add_argument('-p', '--print', action='store_true',
                     help='Print molecule')
    # -- ELECTRONIC STATE OPTIONS
    state = par.add_argument_group('Electronic state(s) selection options')
    state.add_argument('-l', '--lower', type=int, default=1,
                       help='Lowest electronic state to include in sum')
    state.add_argument('-u', '--upper', type=int, default=3,
                       help='Upper electronic state to include in sum')
    state.add_argument('-1', '--only', type=int,
                       help='Electronic state to include (only 1)')
    state.add_argument('-m', '--multi',
                       help='Choose multiple states (as comma-separated list)')
    state.add_argument('--noNAC', action="store_true",
                       help='Deactivates the NAC weight.')
    state.add_argument('--noweight', action="store_true",
                       help='Deactivates the weight (both NAC and transition energy).')

    # -- VIBRATIONAL STATES
    vib = par.add_argument_group('Vibrational state/mode selection options')
    txt = """ Vibrational state of interest, up to three quanta. 
    It should be expressed in the following format:'nvib * quanta',
    'nvib1 * quanta + nvib2 * quanta' or 'nvib1 + nvib2 + nvib3'
    NB: Absolute variational state not yet available
    """
    vib.add_argument('-s', '--vibstate', default='1',
                     help=txt)
    # -- DRAWING PARAMETERS
    # vis = p.add_subparsers('-2d',help='2D-projection')

    save = par.add_argument_group('Saving parameters')
    save.add_argument('--printVec', action='store_true',
                      help='print EDTM and MDTM')
    save.add_argument('--noCube', action='store_true',
                      help='Does not print any cube')
    save.add_argument('--saveSignedCube', action='store_true',
                      help='DEBUG do not use')
    save.add_argument('--printConv', action='store_true',
                      help='''Save the evolution of the ETDM and MTDM during the SOS in a file''')
    return par


def get_nac(fname, vstate=None):
    """
    Extracts NAC from FCHK using JBL parser
    """
    ext = os.path.splitext(fname)[1][1:]
    data = False
    if ext in ('fch', 'fchk'):
        if vstate is not None:
            print('ERROR: vibrational state definition not supported with fchk')
            return False
        else:
            with open(fname, 'r', encoding='utf-8') as fobj:
                line = fobj.readline()
                while not line.startswith('Nonadiabatic coupling'):
                    line = fobj.readline()
                    if not line:
                        print('ERROR: NAC not found in fchk file')
                        return False
                N = int(line.split()[-1])
                nline = ceil(N/fio.NCOLS_FCHK_R)
                data = []
                for _ in range(nline):
                    line = fobj.readline()
                    data.extend([float(item) for item in line.split()])
    elif ext in ('log', 'out'):
        if vstate is None:
            with open(fname, 'r', encoding='utf-8') as fobj:
                line = fobj.readline()
                while 'Nonadiabatic Coup.' not in line:
                    line = fobj.readline()
                    if not line:
                        print('ERROR: NAC not found in log file')
                        return False
                for _ in range(3):
                    line = fobj.readline()
                data = []
                while '-----' not in line:
                    data.extend([float(item) for item in line.split()[-3:]])
                    line = fobj.readline()
        else:
            txt = '\\s+'.join(vstate)+'\\s+'
            key = re.compile(r'^\s+'+txt.replace('(', '\\(').replace(')', '\\)')+r'-?\d+\.')
            with open(fname, 'r', encoding='utf-8') as fobj:
                line = fobj.readline()
                while 'Anharmonic Transition Moments' not in line:
                    line = fobj.readline()
                    if not line:
                        print('ERROR: NAC not found in log file (1)')
                        return False
                while 'Non-adiabatic couplings' not in line:
                    line = fobj.readline()
                    if not line:
                        print('ERROR: NAC not found in log file (2)')
                        return False
                #print(r'^\s+'+r'\s+'.join(vstate)+r'\s+-?\d+\.' )
                while not key.search(line):
                    line = fobj.readline()
                    if not line:
                        print('ERROR: NAC not found in log file (3)')
                        return False
                #print(line)
                data = float(line.split()[-1].replace('D', 'e'))
                getvar = True
                while 'AFTER VARIATIONAL CORRECTION' not in line:
                    line = fobj.readline()
                    if not line:
                        getvar = False
                        # exit
                while 'Non-adiabatic couplings' not in line:
                    line = fobj.readline()
                    if not line:
                        getvar = False
                if getvar:
                    while not key.search(line):
                        line = fobj.readline()
                        if not line:
                            print('ERROR: NAC not found in log file')
                            return False
                    #print(line)
                    data = float(line.split()[-1].replace('D', 'e'))
    else:
        print('ERROR: Unrecognized filetype for NAC.')
        return False
    return data


def get_transition_energy(fname):
    """
    reads from the fchk file the SCF Energy and the total energy
    returns the transition energy as: Etot-E0
    """
    with open(fname, 'r', encoding='utf-8') as fopen:
        line = fopen.readline()
        while not line.startswith('SCF Energy'):
            line = fopen.readline()
            if not line:
                print('ERROR: SCF not found in fchk file')
                return False
        en_zero = float(line.split()[-1])
        while not line.startswith('Total Energy'):
            line = fopen.readline()
            if not line:
                print('ERROR: Total not found in fchk file')
                return False
        en_tot = float(line.split()[-1])
        print(en_tot, en_zero)

    return en_tot-en_zero


def main():
    cp = Colors()
    # DEBUG = False
    PARSER = build_parser()
    OPTS = PARSER.parse_args()
    # Check that reference state data file exists
    if not OPTS.refstate:
        print(cp.printred('ERROR: refstate does not exist'))
        sys.exit()
    # Analys template files
    ntmpl = len(OPTS.tmplfiles)
    tmplfiles = {}
    for item in OPTS.tmplfiles:
        if item.startswith('TCD:'):
            tmplfiles['TCD'] = item[4:]
        elif item.startswith('NAC:'):
            tmplfiles['NAC'] = item[4:]
        elif item.startswith('EXC:'):
            tmplfiles['EXC'] = item[4:]
    if len(tmplfiles) == 0:
        tmplfiles['TCD'] = OPTS.tmplfiles[0]
        if ntmpl > 1:
            tmplfiles['NAC'] = OPTS.tmplfiles[1]
        else:
            tmplfiles['NAC'] = tmplfiles['TCD']
        if ntmpl > 2:
            tmplfiles['EXC'] = OPTS.tmplfiles[2]
        else:
            tmplfiles['EXC'] = tmplfiles['NAC']
    elif len(tmplfiles) != ntmpl:
        print('ERROR: Cannot mix keyword based and position templates')
        sys.exit()
    else:
        if 'NAC' not in tmplfiles or 'TCD' not in tmplfiles:
            print('Provide at least template for NAC and TCD')
        if 'EXC' not in tmplfiles:
            tmplfiles['EXC'] = tmplfiles['NAC']

    tmplexts = {}
    for key in tmplfiles:
        tmplexts[key] = os.path.splitext(tmplfiles[key])[1][1:]
    tmplftypes = {}
    tmplfmt = {}
    for key in tmplfiles:
        # Analyze pattern
        fext = tmplexts[key]
        if key == 'TCD':
            if fext in ('cub', 'cube'):
                tmplftypes[key] = 'cube'
            else:
                fmt = 'ERROR: Cube file extension expected but "{}" found'
                print(cp.printred(fmt.format(fext)))
                sys.exit()
        elif key in ('NAC', 'EXC'):
            if fext in ('fch', 'fchk'):
                tmplftypes[key] = 'fchk'
            elif fext in ('log', 'out'):
                tmplftypes[key] = 'log'
            elif fext in ('dat'):
                tmplftypes[key] = 'dat'
            else:
                fmt = 'ERROR: Formatted chk/output file extension expected but "{}" found'
                print(cp.printred(fmt.format(fext)))
                sys.exit()
        res = re.findall('(X+)', tmplfiles[key])
        if not tmplftypes[key] == 'dat':
            if not res:
                print(cp.printred("ERROR: Missing X's in filename pattern"))
                sys.exit()
            if len(res) > 1:
                print(cp.printred("ERROR: More than 1 sequence of X's found."))
                sys.exit()
            tmplfmt[key] = tmplfiles[key].replace(res[0], '{{:0{:d}d}}'.format(len(res[0])))
        else:
            tmplfmt[key] = tmplfiles[key]

    if OPTS.only:
        minels = OPTS.only
        maxels = OPTS.only
    else:
        minels = OPTS.lower
        maxels = OPTS.upper
    if minels > maxels:
        print(cp.printorange('WARNING: Electronic state bounds appear inverted. Correcting.'))
        minels, maxels = maxels, minels
    #ivib = OPTS.vibstate - 1
    # Get molecule information
    mol = fio.fchk_vib_parser(OPTS.refstate)
    evec = mol['evec']
    if OPTS.print:
        print('''
########################################
###          MOLECULAR DATA          ###
########################################

Coordinates (in Bohr)
''')
        fmt = 'Atom {:3d}  {:2s}  {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}'
        for ia in range(mol['natoms']):
            ian = mol['ian'][ia]
            xyz = mol['crd'][ia, :] / PHYSFACT.bohr2ang
            print(fmt.format(ia+1, ELEMENTS[ian], c=xyz))
    nvib = mol['natoms']*3 - 6

    lvibosc = OPTS.vibstate.split('+')
    dmodes = {}
    nquanta = 0
    for vibosc in lvibosc:
        if '*' in vibosc:
            data = vibosc.split('*')
            numq = int(data[0])
            mode = int(data[1])
        else:
            numq = 1
            mode = int(vibosc)
        nquanta += numq
        if mode > nvib:
            print('ERROR: Absolute variational state not yet available')
            sys.exit()
        if mode in dmodes:
            dmodes[mode] += numq
        else:
            dmodes[mode] = numq
    smodes = []
    lmodes = sorted(dmodes.keys(), reverse=True)
    for mode in lmodes:
        smodes.append('{}({})'.format(mode, dmodes[mode]))
    idvstate = get_ivib(nquanta, lmodes, dmodes, nvib)

    # Use 1st electronic state to build box
    title = 'CURRENT DENSITY FOR MODE {}'.format(OPTS.vibstate)
    print('''
########################################
### {:^32s} ###
########################################
'''.format(title))

    if OPTS.printConv:
        elc = []
        mag = []
        lines = []
    if tmplftypes['NAC'] == 'dat':
        NACS = np.loadtxt(tmplfmt['NAC'])

    flag = 0
    if OPTS.multi:
        lst_elst = [int(s) for s in OPTS.multi.split(',')]
    else:
        lst_elst = range(minels, maxels+1)

    for elst in lst_elst:
        fnames = {}
        for key in tmplfiles:
            fnames[key] = tmplfmt[key].format(elst)
            if not os.path.exists(fnames[key]):
                fmt = 'NOTE: Data files for the state num. {} missing. Skipping.'
                print(fmt.format(elst))
                if elst > maxels and flag:
                    print('ERROR: Not enough data file. Quitting.')
                    sys.exit()
                fnames = False
                continue
        if not fnames:
            continue


        print('Including electronic state num. {}'.format(elst))

        if OPTS.noNAC or OPTS.noweight:
            NACvib = 1.0
        else:
            if tmplftypes['NAC'] == 'fchk':
                dNAC = get_nac(fnames['NAC'])
                # all_nmodes = evec
                # mol.get_energy('hess')
                # tmp = vibana(mol.get_hess(), np.array(mol.coord), np.array(mol.atmass))
                # freq = tmp['freqs']
                # del tmp
                # PRE_FACT = 2 * np.sqrt(freq[int(idvstate-1)] /
                #                        (PC.au2cm1()*PC.finesc()) * pi)
            elif tmplftypes['NAC'] == 'log':
                dNAC = get_nac(fnames['NAC'], smodes)
                # all_nmodes = fio.get_anha_nm(OPTS.refstate)
                if not dNAC and nquanta == 1:
                    dNAC = get_nac(fnames['NAC'])
            else:
                dNAC = NACS[elst - 1]
            if dNAC is False:
                print('ERROR: Non-adiabatic couplings not found. Exiting')
                sys.exit()
            else:
                if isinstance(dNAC, list):
                    # FIXME to check
                    NACvib = np.dot(evec, np.array(dNAC))[int(idvstate-1)]
                    print('state:{} NAC:{}'.format(elst, NACvib))
                else:
                    NACvib = dNAC

        if OPTS.noweight:
            X = 1.0
        else:
            Etrans = get_transition_energy(fnames['EXC'])
            X = NACvib/Etrans
        if not flag:
            cubdat = cb.cube_parser(fnames['TCD'])
            cubdat.make_box()
            cubdat *= X
            # flag = 0

        else:
            cubdat += cb.cube_parser(fnames['TCD']) * X

        # DEBUG
        if not(OPTS.noNAC or OPTS.noweight) \
        and (OPTS.printConv or ((elst == lst_elst[-1])
                                and (OPTS.printVec or
                                     OPTS.saveSignedCube))):
            mu_state = cb.mu_integrate(cubdat)
            mg_state = cb.mag_integrate(cubdat)
            if OPTS.printVec and (elst == lst_elst[-1]):
                dcube = cb.CubeData()
                dcube.natoms = 0
                dcube.origin = np.array([0., 0., 0.])
                dcube.npts = [2, 2, 2]
                dcube.nval = 3
                dcube.step = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
                dcube.cube = np.zeros((3, 8))
                dcube.cube[:, 0] = mu_state
                cb.print_cube(dcube, 'ELC', idvstate)
                dcube.cube[:, 0] = mg_state
                cb.print_cube(dcube, 'MAG', idvstate)
            if OPTS.printConv:
                norm_mu = np.linalg.norm(mu_state)
                norm_mg = np.linalg.norm(mg_state)
                # dot_mumg = np.dot(mu_state, mg_state)
                elc.append(norm_mu)
                mag.append(norm_mg)

                cola = cp.printgreen('{:12.4E}'.format(norm_mu))
                if not flag:
                    print('Electronic EDTM: {}'.format(cola))
                    error = 0.
                else:
                    error = abs(elc[flag]-elc[flag-1])/elc[flag-1]
                    print('Electronic EDTM: {} Variation: {:12.4E}'.format(cola, error))

                lintpl = '{:4d} {a[0]:12.4E} {a[1]:12.4E} {a[2]:12.4E} {b[0]:12.4E} {b[1]:12.4E} {b[2]:12.4E}\n'
                lines.append(lintpl.format(elst, a=mu_state, b=mg_state))

                if elst == lst_elst[-1]:
                    # BUG: print in the same folder
                    print_lines(lines, elst, idvstate, '.')

        flag += 1

    if not OPTS.noCube:
        cubdat.cube *= PRE_FACT
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_U{}.cube'.format(OPTS.vibstate, lst_elst[-1])
        comment = 'VTCD Nmodes: {} Last State: {}'.format(OPTS.vibstate, lst_elst[-1])
        cb.print_cube(cubdat, fname=outfile, comment=comment)

    if OPTS.saveSignedCube:
        VEC1, VEC2 = cb.mask_cube(cubdat, mu_state)
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_{}.cube'.format(OPTS.vibstate, 'Plus')
        comment = 'VTCD Nmodes: {} Signed: {}'.format(OPTS.vibstate, 'Plus')
        cb.print_cube(cubdat, fname=outfile, comment=comment, vec_pr=VEC1)
        outfile = os.path.splitext(OPTS.refstate)[0] + '_VTCD{}_{}.cube'.format(OPTS.vibstate, 'Minus')
        comment = 'VTCD Nmodes: {} Signed: {}'.format(OPTS.vibstate, 'Minus')
        cb.print_cube(cubdat, fname=outfile, comment=comment, vec_pr=VEC2)


if __name__ == '__main__':
    main()
