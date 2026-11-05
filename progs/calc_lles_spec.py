#!/bin/python3
"""
LLES-enhanced IR and VCD spectra (Nafie's sum-over-states theory).

For every vibrational mode a of the frequency fchk, the zeroth-order
transition moments come from the APT/AAT and the enhancement terms from
the low-lying excited states (TD fchk: length-gauge EDTM and MDTM, and the
per-state NAC), with the excitation energies fitted to experiment
(Sapova et al., Angew. Chem. 2026, e17979):

    f_e  = w_a^2 / (w_e,fit^2 - w_a^2)
    E0   = <dmu/dQ_a> / sqrt(2 w_a)          M0 = sqrt(2 w_a) <AAT_a>
    Eenh = sqrt(2/w_a)  sum_e f_e mu_e <e|dg/dQ_a>
    Menh = -sqrt(2 w_a) sum_e f_e (m_e/w_e,fit) <e|dg/dQ_a>

(atomic units, mu_e = -edil and m_e = -mdi/2 of the TD fchk, ELEMENTS
conventions as in calc_lles.py).  With E = E0 + Eenh and M = M0 + Menh:

    D = |E|^2        R = E.M

The spectra are Lorentzian-broadened, with and without the enhancement.

Es: python3 calc_lles_spec.py h2o2_td.fchk h2o2_freq.fchk \\
        h2o2_NACXXX.fchk energies.inp -s 1 -e 3

energies.inp, one line per state of the -s/-e range (eV):
    state 1: 0.23
    state 2: 0.25
"""
import os
import argparse
import numpy as np
from estampes.data.physics import phys_fact
from estampes.tools.spec import broaden
import tcdlibx.io.estp_io as estp
from tcdlibx.utils.conversion_units import (AU2EV, DQ_AMU2AU, IR2EPS, VCD2DE,
                                            ele_edip_cgs, ele_mdip_cgs)
from calc_lles import (NEAR_RES, cube_name, error, read_energies,
                       warning)

PROGNAME = os.path.basename(__file__)


def build_parser():
    """
    Build options parser.
    """
    par = argparse.ArgumentParser(prog=PROGNAME,
                                  formatter_class=argparse.RawTextHelpFormatter)
    par.add_argument('tdfchk',
                     help='Gaussian TD fchk (energies, length EDTM, MDTM)')
    par.add_argument('freqfchk',
                     help='Gaussian fchk with the vibrational data (APT, AAT)')
    txt = """Template of the per-state fchk with the NAC.
Each group of "X" is replaced by the state index (zero-padded).
Example: h2o2_NACXXX.fchk"""
    par.add_argument('nactmpl', help=txt)
    txt = """Fitted excitation energies (eV), one line per state:
state 1: 0.23"""
    par.add_argument('energies', help=txt)
    par.add_argument('-s', '--min-state', type=int, required=True,
                     help='Lowest electronic state')
    par.add_argument('-e', '--max-state', type=int, required=True,
                     help='Highest electronic state')
    par.add_argument('--hwhm', type=float, default=4.0,
                     help='Lorentzian half-width at half-maximum (cm-1)')
    par.add_argument('--fmin', type=float, default=100.0,
                     help='Lowest frequency of the spectrum (cm-1)')
    par.add_argument('--fmax', type=float, default=2000.0,
                     help='Highest frequency of the spectrum (cm-1)')
    par.add_argument('--step', type=float, default=1.0,
                     help='Frequency step of the spectrum (cm-1)')
    par.add_argument('-o', '--output',
                     help='Spectrum file (default: <tdfchk>_lles_spec.dat)')
    return par


def spectrum(grid, freq, strength, hwhm, fact):
    """Lorentzian-broadened lines: y = fact nu sum_k S_k g_k."""
    return np.array(broaden(freq, strength, grid, 'lorentzian', hwhm,
                            yfactor=fact, xfunc=lambda x: x))


def spectrum_numpy(grid, freq, strength, hwhm, fact):
    """Same as spectrum(), vectorized: ~40x faster."""
    # TODO: evaluate replacing broaden() with this for performance
    delta = grid[:, np.newaxis] - freq[np.newaxis, :]
    shape = hwhm / np.pi / (delta**2 + hwhm**2)
    return fact * grid * (shape @ strength)


def main():
    opts = build_parser().parse_args()
    if opts.min_state > opts.max_state:
        error('--min-state larger than --max-state')
    states = list(range(opts.min_state, opts.max_state + 1))
    fit_ev = read_energies(opts.energies, states)

    tdmol = estp.get_elemol(opts.tdfchk)
    if opts.max_state > len(tdmol['exeng']):
        error(f'{opts.tdfchk} contains only {len(tdmol["exeng"])} states')
    vbmol = estp.get_vibmol(opts.freqfchk)
    if vbmol['apt'] is None or vbmol['aat'] is None:
        error(f'{opts.freqfchk}: APT/AAT not found')

    freq_au = vbmol['freq'] / phys_fact('au2cm1')
    conv = DQ_AMU2AU
    sqw = np.sqrt(2.0 * freq_au)[:, np.newaxis]
    # Zeroth order
    edip0 = conv * vbmol['edi'] / sqw
    mdip0 = conv * vbmol['mdi'] * sqw

    # NAC projected on the normal modes, (mode, state)
    nac = []
    for state in states:
        fname = cube_name(opts.nactmpl, state)
        if not os.path.exists(fname):
            error(f'NAC file {fname} not found')
        tmp = estp.get_nac(fname)
        if tmp.size != vbmol['lx'].shape[1]:
            error(f'{fname}: NAC size differs from the normal mode size')
        nac.append(conv * vbmol['lx'] @ tmp)
    nac = np.array(nac).T

    # Enhancement factors
    w_fit = np.array([fit_ev[state] for state in states]) / AU2EV
    denom = w_fit[np.newaxis, :]**2 - freq_au[:, np.newaxis]**2
    if (denom == 0.0).any():
        error('a state is exactly resonant with a vibrational frequency: '
              'enhancement factor undefined')
    fac = freq_au[:, np.newaxis]**2 / denom
    for istate, state in enumerate(states):
        neg = np.flatnonzero(w_fit[istate] < freq_au) + 1
        near = np.flatnonzero((w_fit[istate] >= freq_au)
                              & (w_fit[istate] <= (1.0 + NEAR_RES) * freq_au)) + 1
        if neg.size:
            warning(f'state {state}: w_fit < w_a (negative enhancement) for '
                    f'modes {neg.tolist()}')
        if near.size:
            warning(f'state {state}: close to resonance for modes '
                    f'{near.tolist()}')

    # Length-gauge electric and magnetic transition moments (ELEMENTS signs)
    idx = [state - 1 for state in states]
    mu_e = -tdmol['edil'][idx]
    m_e = -0.5 * tdmol['mdi'][idx]
    edip_enh = np.sqrt(2.0 / freq_au)[:, np.newaxis] \
        * np.einsum('ms,ms,sx->mx', fac, nac, mu_e)
    mdip_enh = -sqw * np.einsum('ms,ms,sx->mx', fac,
                                nac / w_fit[np.newaxis, :], m_e)

    edip = edip0 + edip_enh
    # Dipole strengths in 1e-40 and rotational strengths in 1e-44 esu^2.cm^2
    edip0_cgs, edip_cgs = ele_edip_cgs(edip0), ele_edip_cgs(edip0 + edip_enh)
    mdip0_cgs, mdip_cgs = ele_mdip_cgs(mdip0), ele_mdip_cgs(mdip0 + mdip_enh)
    dip0 = np.sum(edip0_cgs**2, axis=1) * 1.0e40
    rot0 = np.sum(edip0_cgs * mdip0_cgs, axis=1) * 1.0e44
    dip = np.sum(edip_cgs**2, axis=1) * 1.0e40
    rot = np.sum(edip_cgs * mdip_cgs, axis=1) * 1.0e44

    print(f'{"Mode":>4} {"w (cm-1)":>9} {"D0":>11} {"D":>11} {"R0":>11} '
          f'{"R":>11}    (D in 1e-40 esu2cm2, R in 1e-44 esu2cm2)')
    for imode, freq in enumerate(vbmol['freq']):
        print(f'{imode+1:4d} {freq:9.2f} {dip0[imode]:11.4f} '
              f'{dip[imode]:11.4f} {rot0[imode]:11.4f} {rot[imode]:11.4f}')

    grid = np.arange(opts.fmin, opts.fmax + 0.5 * opts.step, opts.step)
    cols = [grid,
            spectrum(grid, vbmol['freq'], dip0, opts.hwhm, IR2EPS),
            spectrum(grid, vbmol['freq'], dip, opts.hwhm, IR2EPS),
            spectrum(grid, vbmol['freq'], rot0, opts.hwhm, VCD2DE),
            spectrum(grid, vbmol['freq'], rot, opts.hwhm, VCD2DE)]
    fname = opts.output or os.path.splitext(
        os.path.basename(opts.tdfchk))[0] + '_lles_spec.dat'
    np.savetxt(fname, np.array(cols).T, fmt='%10.2f' + ' %14.6E' * 4,
               header='freq(cm-1) eps_0 eps_enh deps_0 deps_enh '
                      '(L mol-1 cm-1)')
    print(f'File {fname} has been generated!')


if __name__ == '__main__':
    main()
