#!/bin/python3
"""
LLES enhancement correction (Nafie's sum-over-states theory) of the VTCD.

Starting from the per-state ETCD cubes (ELEMENTS, velocity gauge), the
Cartesian NAC of each state (fchk) and the normal mode of the frequency
fchk, builds the per-state VTCD:

    J_i = 2 w_a zp (<i|dg/dQ_a>/sqrt(mu_a)) / w_i,fchk * TCD_i

and from it the low-lying excited-state (LLES) correction cubes of the
EDTM and of the MDTM, with the excitation energies of the enhancement
factor fitted to experiment:

    f_i = w_a^2 / (w_i,fit^2 - w_a^2)
    J_EDTM = sum_i f_i J_i
    J_MDTM = (r-O) x sum_i f_i (w_i,fchk / w_i,fit) J_i

J_i contains the velocity-gauge 1/w_i,fchk weight; the MDTM term also
replaces it with 1/w_i,fit (Sapova et al., Angew. Chem. 2026, e17979,
Eq. 16).  The moments follow the ELEMENTS conventions:
EDTM = -(1/w_a) int J and MDTM = -1/2 int (r-O) x J, O = nuclear charge
centre.  The nuclear contributions are zero: they are carried by the
zeroth-order (lr_vtcd_cube) cubes, which can be added with sum_vtcd.py.

Es: python3 calc_lles.py h2o2_td.fchk h2o2_freq.fchk \\
        h2o2_tcd_SX.cube h2o2_NACXXX.fchk energies.inp -m 4 -s 1 -e 3

energies.inp, one line per state of the -s/-e range (eV):
    state 1: 0.23
    state 2: 0.25
"""
import sys
import os
import re
import argparse
import numpy as np
# import custom library
from estampes.data.physics import PHYSFACT, phys_fact
import tcdlibx.calc.cube_manip as cb
import tcdlibx.io.estp_io as estp
from tcdlibx.utils.color_out import Colors

PROGNAME = os.path.basename(sys.argv[0])
# Relative margin above w_a below which w_fit is close to resonance
# (w_fit <= 1.1 w_a gives f >= 4.8).
NEAR_RES = 0.10


def build_parser():
    """
    Build options parser.
    """
    par = argparse.ArgumentParser(prog=PROGNAME,
                                  formatter_class=argparse.RawTextHelpFormatter)
    par.add_argument('tdfchk',
                     help='Gaussian TD fchk (excitation energies, ETran)')
    par.add_argument('freqfchk',
                     help='Gaussian fchk with the vibrational data')
    txt = """Template of the per-state ETCD cubes.
Each group of "X" is replaced by the state index (zero-padded).
Example: h2o2_tcd_SXX.cube"""
    par.add_argument('tcdtmpl', help=txt)
    txt = """Template of the per-state fchk with the NAC (same convention).
Example: h2o2_NACXXX.fchk"""
    par.add_argument('nactmpl', help=txt)
    txt = """Fitted excitation energies (eV), one line per state:
state 1: 0.23"""
    par.add_argument('energies', help=txt)
    par.add_argument('-m', '--mode', type=int, required=True,
                     help='Vibrational mode (fundamental), 1-based')
    par.add_argument('-s', '--min-state', type=int, required=True,
                     help='Lowest electronic state')
    par.add_argument('-e', '--max-state', type=int, required=True,
                     help='Highest electronic state')
    return par


def error(msg):
    print(Colors().printred('ERROR: ' + msg))
    sys.exit(1)


def warning(msg):
    print(Colors().printorange('WARNING: ' + msg))


def read_energies(fname, states):
    """Read the fitted energies (eV) from lines "state i: E"."""
    line_re = re.compile(r'^\s*state\s+(\d+)\s*:\s*(\S+)\s*$')
    energies = {}
    with open(fname, 'r', encoding='utf-8') as fopen:
        for iline, line in enumerate(fopen, start=1):
            if not line.strip():
                continue
            match = line_re.match(line)
            if match is None:
                error(f'{fname}, line {iline}: "state i: E" expected')
            state = int(match.group(1))
            if state in energies:
                error(f'{fname}: state {state} defined twice')
            try:
                value = float(match.group(2))
            except ValueError:
                error(f'{fname}, line {iline}: invalid energy')
            if value <= 0.0:
                error(f'{fname}: energy of state {state} must be positive')
            energies[state] = value
    missing = sorted(set(states) - set(energies))
    if missing:
        error(f'{fname}: missing energies for states {missing}')
    return {state: energies[state] for state in states}


def cube_name(template, state):
    """Replace each group of X with the zero-padded state index."""
    return re.sub('X+', lambda m: f'{state:0{len(m.group(0))}d}', template)


def fmt_vec(vec):
    return '[{:14.7E},{:14.7E},{:14.7E}]'.format(*vec)


def main():
    opts = build_parser().parse_args()
    if opts.min_state > opts.max_state:
        error('--min-state larger than --max-state')
    states = list(range(opts.min_state, opts.max_state + 1))

    fit_ev = read_energies(opts.energies, states)
    # TD file
    tdmol = estp.get_elemol(opts.tdfchk)
    fchk_au = tdmol['exeng']
    if opts.max_state > len(fchk_au):
        error(f'{opts.tdfchk} contains only {len(fchk_au)} states')

    # Per-state ETCD cubes
    cubes = {}
    for state in states:
        fname = cube_name(opts.tcdtmpl, state)
        if not os.path.exists(fname):
            error(f'cube file {fname} not found')
        cube = cb.cube_parser(fname)
        if state == states[0]:
            ref = cube
        elif not ref.same_system(cube) or \
                not np.allclose(ref.loc2wrd, cube.loc2wrd):
            error(f'{fname}: grid or molecule differ from the first cube')
        cubes[state] = cube

    # Normal mode and ELEMENTS prefactors 
    vbmol = estp.get_vibmol(opts.freqfchk)
    mode = opts.mode
    if not 1 <= mode <= len(vbmol['freq']):
        error(f'mode {mode} not found in {opts.freqfchk}')
    freq_cm1 = vbmol['freq'][mode - 1]
    freq_au = freq_cm1 / phys_fact('au2cm1')
    zp_ampl = 1.0 / np.sqrt(2.0 * freq_au / phys_fact('au2amu'))
    pre_fact = 2.0 * freq_au * zp_ampl
    lx_mode = vbmol['lx'][mode - 1]

    # Weight of the ETCD of each state: J_i = X_i TCD_i
    weight = {}
    for state in states:
        fname = cube_name(opts.nactmpl, state)
        if not os.path.exists(fname):
            error(f'NAC file {fname} not found')
        nac = estp.get_nac(fname)
        if nac.size != lx_mode.size:
            error(f'{fname}: NAC size differs from the normal mode size')
        weight[state] = lx_mode @ nac / fchk_au[state - 1] * pre_fact

    # Enhancement factors
    au2ev = PHYSFACT.Eh2J / PHYSFACT.e2C
    fac, ratio = {}, {}
    for state in states:
        w_fit = fit_ev[state] / au2ev
        denom = w_fit**2 - freq_au**2
        if denom == 0.0:
            error(f'state {state} is exactly resonant with the vibrational '
                  'frequency: enhancement factor undefined')
        fac[state] = freq_au**2 / denom
        ratio[state] = fchk_au[state - 1] / w_fit
        if w_fit < freq_au:
            warning(f'state {state} lies below the vibrational frequency '
                    '(w_fit < w_a): the enhancement factor is negative')
        elif w_fit <= (1.0 + NEAR_RES) * freq_au:
            warning(f'state {state} is close to resonance (w_fit within '
                    f'{100*NEAR_RES:.0f}% above w_a): f = {fac[state]:.3E}')

    # Correction cubes and per-state moments
    origin = cb.charge_centre(ref)
    edtm_cube = cb.CubeData(ref)
    edtm_cube.cube = np.zeros_like(ref.cube)
    mdtm_j = cb.CubeData(ref)
    mdtm_j.cube = np.zeros_like(ref.cube)
    print(f'\nLLES correction for mode {mode}, w_a = {freq_cm1:.2f} cm-1')
    print(f'MDTM origin at nuclear charge centre: {fmt_vec(origin)}\n')
    print(f'{"State":>5} {"w_fchk (eV)":>12} {"w_fit (eV)":>11} {"f":>12}'
          f'   {"Electronic EDTM (au)":^46}   {"Electronic MDTM (au)":^46}')
    edtm_tot = np.zeros(3)
    mdtm_tot = np.zeros(3)
    for state in states:
        tmp = cb.CubeData(ref)
        tmp.cube = fac[state] * weight[state] * cubes[state].cube
        edtm = cb.mu_integrate(tmp, freq_au, 'vtcd')
        edtm_cube.cube += tmp.cube
        tmp.cube = tmp.cube * ratio[state]
        mdtm = cb.mag_integrate(tmp, 'vtcd', origin=origin)
        mdtm_j.cube += tmp.cube
        edtm_tot += edtm
        mdtm_tot += mdtm
        print(f'{state:5d} {fchk_au[state-1]*au2ev:12.6f} '
              f'{fit_ev[state]:11.6f} {fac[state]:12.5E}   '
              f'{fmt_vec(edtm)}   {fmt_vec(mdtm)}')
    print(f'{"Total":>43}   {fmt_vec(edtm_tot)}   {fmt_vec(mdtm_tot)}')

    # The MDTM cube stores (r-O) x J, as the vtcd_cube -a cubes
    mdtm_cube = cb.CubeData(ref)
    mdtm_cube.cube = cb.calc_rot(mdtm_j, origin=origin)

    base = os.path.splitext(os.path.basename(opts.tdfchk))[0]
    vlab = str(mode)
    zero = np.zeros(3)
    for quantity, cube, elec in (('EDTM', edtm_cube, edtm_tot),
                                 ('MDTM', mdtm_cube, mdtm_tot)):
        fname = f'{base}_lles_{quantity.lower()}_v{vlab}.cube'
        comment = (f'LLES VTCD/{quantity} | States: {states[0]}-{states[-1]} '
                   f'| Mode: {mode} | Electronic {quantity}: {fmt_vec(elec)}, '
                   f'Nuclear {quantity}: {fmt_vec(zero)}')
        cb.print_cube(cube, fname=fname, comment=comment)
        print(f'File {fname} has been generated!')


if __name__ == '__main__':
    main()
