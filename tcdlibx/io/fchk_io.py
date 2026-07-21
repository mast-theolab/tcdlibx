# -*- coding: utf-8 -*-
"""
Legacy functions to read data from fchk files
"""
import os
import re
from math import ceil
import numpy as np
from tcdlibx.utils.custom_except import NoValidData


NCOLS_FCHK_R = 5
NCOLS_FCHK_I = 6

def fchk_vib_parser(fname):
    """Function that parse the Gaussian Formatted checkpoint file and get the
    data required by acp analysis

    Arguments:
        fname {str} -- the file name of the .fchk file
    """
    keys = {'nat': 'Number of atoms',
            'crd': 'Current cartesian coordinates',
            'ian': 'Atomic numbers',
            'atm': 'Vib-AtMass',
            'evc': 'Vib-Modes',
            'apt': 'Dipole Derivatives',
            'aat': 'AAT',
            've2': 'Vib-E2'
            }
    #name = os.path.split(fname)[-1][:-5]
    data = {}
    qtt = len(keys)
    with open(fname, 'r', encoding='utf-8') as fopen:
        deriv = 'num derivs'
        line = fopen.readline()
        while line:
            if line.startswith(keys['nat']):
                data['natoms'] = int(line.split()[-1])
                # BUG
                nmnum = data['natoms'] * 3 - 6
                qtt -= 1
            elif line.startswith(keys['crd']):
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(x) for x in line.split()])
                tmp = np.array(tmp)
                data['crd'] = tmp.reshape(-1, 3)
                qtt -= 1
            elif line.startswith(keys['ian']):
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_I)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([int(x) for x in line.split()])
                data['atnum'] = np.array(tmp)
                qtt -= 1
            elif line.startswith(keys['atm']):
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(x) for x in line.split()])
                data['atmas'] = np.array(tmp)
                qtt -= 1
            elif line.startswith(keys['apt']) and deriv not in line:
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(x) for x in line.split()])
                tmp = np.array(tmp)
                data['apt'] = tmp.reshape(-1, 3, 3)
                qtt -= 1
            elif line.startswith(keys['aat']) and deriv not in line:
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(e) for e in line.split()])
                tmp = np.array(tmp)
                data['aat'] = tmp.reshape(-1, 3, 3)
                qtt -= 1
            elif line.startswith(keys['evc']):
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(x) for x in line.split()])
                tmp = np.array(tmp)
                data['evec'] = tmp.reshape(-1, 3*data['natoms'])
                qtt -= 1
            elif line.startswith(keys['ve2']):
                nval = int(line.split()[-1])
                nline = ceil(nval/NCOLS_FCHK_R)
                tmp = []
                for _ in range(nline):
                    line = fopen.readline()
                    tmp.extend([float(e) for e in line.split()])
                data['frq'] = np.array(tmp[:nmnum])
                data['rmas'] = np.array(tmp[nmnum:2*nmnum])
                qtt -= 1
            if qtt == 0:
                break
            line = fopen.readline()
    return data

def get_anha_nm(fname):
    """
    quick function with no checks to get the
    anharm normal modes from fchk file if presente
    """
    with open(fname, 'r', encoding='utf-8') as fchk_file:
        for _ in range(3):
            line = fchk_file.readline()
            if not line:
                raise NoValidData(fname, "not a anharm calc.")
        noa = int(line.split()[4])
        ptt = 'Anharmonic Vib-Modes'
        while ptt not in line:
            line = fchk_file.readline()
            if not line:
                raise NoValidData(fname, "not a anharm calc.")
        tmp_n = int(line.split()[4])
        ltr = ceil(tmp_n / 5)
        nnm = int(tmp_n / (noa * 3))
        # nm_anhar = np.zeros((nnm, noa, 3))
        tmp_strg = ''
        for _ in range(ltr):
            line = fchk_file.readline()
            tmp_strg += line
        datan = np.array([float(e) for e in tmp_strg.split()])
        nm_anhar = np.reshape(datan, (nnm, noa * 3))
        # for i in range(nnm):
        #     for j in range(noa):
        #         for k in range(3):
        #             pos = i * (noa * 3) + j * 3 + k
        #             nm_anhar[i, j, k] = data[pos]
        return nm_anhar

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
                num = int(line.split()[-1])
                nline = ceil(num/NCOLS_FCHK_R)
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
#
