# -*- coding: utf-8 -*-
"""
Read data from fchk from estampes
"""
import numpy as np
# from math import ceil

from estampes.parser.base import DataFile
from estampes.base import QLabel
# from estampes.tools.vib import orient_modes
from estampes.tools.atom import convert_labsymb
from estampes.data.physics import PHYSFACT
# from estampes.data.property import property_units, phys_fact


def get_hnac(fname):
    """Reads NAC and transition energy from a Gaussian fchk file

    Args:
        fname (str): gaussian fchk filename
    """
    dkeys = {
             'e0': QLabel(quantity=1, refstate=0),
             'ex': QLabel(quantity=1, refstate='c'),
             'nac': QLabel(quantity=50),
             'na': QLabel('NAtoms')
    }

    dfile = DataFile(fname)
    data = dfile.get_data(**dkeys)
    natm = int(data['na'].data)
    deng = data['ex'].data[0] - data['e0'].data[0]
    nac = np.array(data['nac'].data).reshape((natm, 3))
    return (deng, nac)


def get_nac(fname):
    """Reads the Cartesian NAC (au) of the current state from a Gaussian fchk

    Args:
        fname (str): gaussian fchk filename

    Returns:
        np.ndarray: NAC as a flat array of 3*natoms elements
    """
    data = DataFile(fname).get_data(nac=QLabel(quantity=50))
    return np.array(data['nac'].data)


def get_vibmol(fname):
    """_summary_

    Args:
        fname (str): gaussian fchk filename

    Returns:
        dict: read quantities
    """

    dkeys2 = {'eng': QLabel(quantity=1),
              'atcrd': QLabel(quantity='atcrd', descriptor='last'),
              'atnum': QLabel(quantity='atnum'),
              'atmas': QLabel(quantity='atmas'),
              'hessvec': QLabel(quantity='hessvec'),
              'freq': QLabel(quantity='hessdat', descriptor='freq'),
              'rmas': QLabel(quantity='hessdat', descriptor='redmas')}
    # print(fname)
    dfile = DataFile(fname)
    res = {}
    res['fname'] = fname
    data = dfile.get_data(**dkeys2)
    atmnum = len(data['atnum'].data)
    res['atnum'] = data['atnum'].data
    # res['eng'] = data[dkeys2['eng']]['data']
    res['atlab'] = convert_labsymb(True, *data['atnum'].data)
    res['atcrd'] = np.array(data['atcrd'].data)*PHYSFACT.bohr2ang
    res['eng'] = data['eng'].data
    res['atmas'] = np.array(data['atmas'].data)
    # BUG no linear
    # nmnum = atmnum*3-6
    res['evec'] = np.reshape(np.array(data['hessvec'].data), (-1, atmnum*3))
    res['freq'] = np.array(data['freq'].data)
    res['rmas'] = np.array(data['rmas'].data)
    # res['rmas'] = tmp[nmnum:2*nmnum]
    # res['rmas'] = fchk_rmas(fname)
    # Waiting for fix in estampes

    res['lx'] = res['evec']/np.sqrt(res['rmas'])[:, np.newaxis]

    try:
        dkeys = {'apt': QLabel(quantity=101,
                               derorder=1,
                               dercoord='x')}
        data = dfile.get_data(**dkeys)
        res['apt'] = np.array(data['apt'].data).reshape(-1, 3)
        res['edi'] = np.einsum("ij,jk->ik", res['lx'], res['apt'])
    except IndexError:
        res['apt'] = None
    try:
        dkeys = {'aat': QLabel(quantity=102,
                               derorder=1,
                               dercoord='x')}
        data = dfile.get_data(**dkeys)
        res['aat'] = np.array(data['aat'].data).reshape(-1, 3)
        res['mdi'] = np.einsum("ij,jk->ik", res['lx'], res['aat'])
    except IndexError:
        res['aat'] = None

    return res


def get_mol(fname):
    """_summary_

    Args:
        fname (str): gaussian fchk filename

    Returns:
        dict: read quantities
    """

    dkeys2 = {'eng': QLabel(quantity=1),
              'atcrd': QLabel(quantity='atcrd', descriptor='last'),
              'atnum': QLabel(quantity='atnum'),
              'atmas': QLabel(quantity='atmas')}
    # print(fname)
    dfile = DataFile(fname)
    res = {}
    res['fname'] = fname
    data = dfile.get_data(**dkeys2)
    # atmnum = len(data['atnum'].data)
    res['atnum'] = data['atnum'].data
    # res['eng'] = data[dkeys2['eng']]['data']
    res['atlab'] = convert_labsymb(True, *data['atnum'].data)
    res['atcrd'] = np.array(data['atcrd'].data)*PHYSFACT.bohr2ang
    res['eng'] = data['eng'].data
    res['atmas'] = np.array(data['atmas'].data)

    return res

def get_elemol(fname):
    """_summary_

    Args:
        fname (str): gaussian fchk filename

    Returns:
        dict: read quantities
    """

    dkeys2 = {'eng': QLabel(quantity=1, refstate='0'),
              'atcrd': QLabel(quantity='atcrd', descriptor='last'),
              'atnum': QLabel(quantity='atnum'),
              'atmas': QLabel(quantity='atmas'),
              'exeng': QLabel(quantity=1, refstate='0->a'),
              'edip': QLabel(quantity=101, refstate='0->a', descriptor='vel'),
              'mdip': QLabel(quantity=102, refstate='0->a'),
              'na': QLabel(quantity='NAtoms')}
    # print(fname)
    dfile = DataFile(fname)
    res = {}
    res['fname'] = fname
    data = dfile.get_data(**dkeys2)
    # atmnum = len(data['atnum'].data)
    res['atnum'] = data['atnum'].data
    # res['eng'] = data[dkeys2['eng']]['data']
    res['atlab'] = convert_labsymb(True, *data['atnum'].data)
    res['atcrd'] = np.array(data['atcrd'].data)*PHYSFACT.bohr2ang
    res['eng'] = data['eng'].data
    res['atmas'] = np.array(data['atmas'].data)
    # BUG no linear
    # nmnum = atmnum*3-6
    # res['rmas'] = tmp[nmnum:2*nmnum]
    # res['rmas'] = fchk_rmas(fname)
    # Waiting for fix in estampes
    res['exeng'] = np.array(data['exeng'].data)
    res['edi'] = np.array(data['edip'].data)
    res['mdi'] = np.array(data['mdip'].data)

    return res
