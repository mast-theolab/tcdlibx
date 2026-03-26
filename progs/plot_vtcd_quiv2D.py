#!/bin/python3
"""
FIXME: old script to be updated
Adapted script to plot written cube in 2d
"""
import sys
import os
import argparse
# import operator
# import time
# import copy
from math import ceil, floor
import numpy as np
import matplotlib as mpl
if "DISPLAY" not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
#import custom library
from estampes.data.physics import PHYSFACT
import tcdlibx.calc.cube_manip as cb
import tcdlibx.graph.cube_graphic as cbplt
from tcdlibx.graph.helpers import molecular_voxels
from tcdlibx.utils.custom_except import NoValidData
import tcdlibx.io.fchk_io as fio
# import service file
from tcdlibx.utils.mol_data import ELEMENTS
from tcdlibx.utils.var_tools import get_ivib

#matplotlib parameters
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}',r'\usepackage{amsmath}']
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath}', r'\usepackage{lmodern}']
#mpl.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['lines.linewidth'] = 2.5


PROGNAME = os.path.basename(sys.argv[0])
NSTEP_BOX = 1
SCALE_ARROW = None


def build_parser():
    """
    Build options parser.
    """
    par = argparse.ArgumentParser(prog=PROGNAME,
                                  formatter_class=argparse.RawTextHelpFormatter)
    # MANDATORY ARGUMENTS
    txt = "Gaussian fchk with data on state of interest"
    par.add_argument('refstate', help=txt)
    txt = "Gaussian cube with data on state of interest"
    par.add_argument('cubefile', help=txt)
    # Optional
    par.add_argument('-p', '--print', action='store_true',
                     help='Print molecule')
    par.add_argument('--folder', type=str, default='Q2D_plot',
                     help='Plot destination folder, if not present will be create')
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

    draw = par.add_argument_group('Drawing parameters')
    # raw.add_argument('-v', '--view', choices=('2d', '3d'), default='3d',
    #                 help='type of graph. 2D or 3D')
    draw.add_argument('-a', '--axis', choices=('x', 'y', 'z'),
                      help='Axis for the projection')
    draw.add_argument('--vscale', type=float, default=1.,
                      help='Overall scaling factor for the arrows')
    draw.add_argument('--sfactor', type=float, default=1.,
                      help='Overall scaling factor for the arrows')
    draw.add_argument('--printNM', action='store_true',
                      help='Print normal modes')
    draw.add_argument('--scaleNM', type=float, default=1.,
                      help='Scaling factor for the Normal Modes')
    draw.add_argument('--printTitle', type=str, default=None,
                      help='Print this title')
    draw.add_argument('--setlimit',
                      help='Set the axis limits for a 2d plot as \
                      comma separated values: [xmin,xmax,ymix,ymax]')
    draw.add_argument('--symplot', action='store_true',
                      help='''Center the 2D plot at the Origin (0,0)''')
    draw.add_argument('-t', '--type', choices=('quiver', 'stream',
                                               'animated_stream',
                                               'stream2'),
                      default='quiver',
                      help='''Type of representation of vecfield:
                      quiver/streamline/animatedstreamline''')
    draw.add_argument('--figext', type=str,
                      choices=('pdf', 'eps', 'png'),
                      default='pdf',
                      help='Figure extension')
    # Quiver parameters
    quiver = par.add_argument_group('Quiver Parameters')
    quiver.add_argument('--quivbkg', action='store_true',
                      help='''Quiver background based on norm of the vector field''')
    quiver.add_argument('--filtnuclei', action='store_true',
                      help='''Filter out the current density vectors that are too close to the nuclei''')
    quiver.add_argument('--qlower', type=float, default=0.0001,
                      help='''Lower bound for the vector field values''')
    quiver.add_argument('--qupper', type=float, default=0.01,
                      help='''Upper bound for the vector field values''')
    quiver.add_argument('--subsample', type=int, default=1,
                      help='''Show only every nth vector (default: 1, no subsampling)''')
    
    molecule = par.add_argument_group('Drawing Molecule Parameters')
    molecule.add_argument('--scaleNuc', type=float, default=1.0,
                          help='''Enter the scaling factor for Nucleus circles''')
    molecule.add_argument('--scaleBond', type=float, default=1.0,
                          help='''Enter the scaling factor for Bonds''')
    # CUBE OPERATIONS
    cube = par.add_argument_group('Cube Operation Parameters')
    cube.add_argument('-g', '--grid', type=int,
                      help='Sets the step for the grid construction')
    cube.add_argument('--xmin', type=float,
                      help='Lower bound along x for the current vectors')
    cube.add_argument('--xmax', type=float,
                      help='Upper bound along x for the current vectors')
    cube.add_argument('--ymin', type=float,
                      help='Lower bound along y for the current vectors')
    cube.add_argument('--ymax', type=float,
                      help='Upper bound along y for the current vectors')
    cube.add_argument('--zmin', type=float,
                      help='Lower bound along z for the current vectors')
    cube.add_argument('--zmax', type=float,
                      help='Upper bound along z for the current vectors')
    # cube.add_argument('--saveSignedCube', action='store_true',
    #                   help='Save the last computed cube')
    cube.add_argument('--printVec', action='store_true',
                      help='print EDTM and MDTM')
    # BUG to befixed
    # cube.add_argument('--matchEDTM', action='store_true',
    #                   help='Rotate the cube data set such as EDTM lies on the x axis')
    # PARTITIONING DATA
    # 
    part = par.add_argument_group('Partitioning Cube parameters')
    part.add_argument('--partCube', type=str, default=None,
                      help='''Cube file containing the partitioning scheme,
                      Should share with the Vector field data set the
                      localTwoWorld Matrix and the number of step''')
#    part.add_argument('--groupsMol', type=str, default=None,
#                      help='''Select the groups of atoms''')

    return par


def main():
    DEBUG = True
    PARSER = build_parser()
    OPTS = PARSER.parse_args()
    # Check that reference state data file exists
    if not OPTS.refstate:
        print('ERROR: refstate does not exist')
    if not OPTS.cubefile:
        print('ERROR: cubefile does not exist')
    if OPTS.grid:
        ngrdstp = OPTS.grid
    else:
        ngrdstp = NSTEP_BOX


    SCF = [OPTS.scaleNuc, OPTS.scaleBond]

    mol = fio.fchk_vib_parser(OPTS.refstate)
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
    evec = mol['evec']

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
    title = 'CURRENT DENSITY FOR TRANSITION TO {}'.format('+'.join(smodes))
    print('''
########################################
### {:^32s} ###
########################################
'''.format(title))

    if OPTS.axis and not OPTS.type == 'animated_stream':
        resfolder = os.path.join(OPTS.folder)
        try:
            os.stat(resfolder)
        except OSError:
            os.mkdir(resfolder)
        ressubfolder = os.path.join(resfolder, "vibstate{}_{}".format(''.join(smodes), OPTS.axis))
        try:
            os.mkdir(ressubfolder)
        except OSError:
            bakname = ressubfolder+'.back'
            counter = 0
            while os.path.exists(bakname):
                counter += 1
                bakname = ressubfolder+'.back'+str(counter)
            print("WARNING: {} already exists: moved to {}".format(ressubfolder, bakname))
            os.renames(ressubfolder, bakname)
            os.mkdir(ressubfolder)

    try:
        cubdat = cb.cube_parser(OPTS.cubefile)
    except OSError:
        print('{} file does not exist. exit'.format(OPTS.cubefile))
        sys.exit()
    if OPTS.filtnuclei:
        _mvox = molecular_voxels(cubdat, minthresh=0., maxthresh=0.15)
        cubdat.cube[:, _mvox] = 0. 
    cubdat.make_box()
    AT_BONDS = cbplt.get_connect(cubdat.ian, cubdat.crd)

    # if anharm
    try:
        all_nmodes = fio.get_anha_nm(OPTS.refstate)
    except NoValidData:
        all_nmodes = evec
    cubdat *= OPTS.sfactor
    if OPTS.sfactor < 0:
        all_nmodes *= -1
    # flag = 0

    # BUG
    if OPTS.xmin or OPTS.xmax or OPTS.ymin or \
       OPTS.ymax or OPTS.zmin or OPTS.zmax:
        cubdat.cube = cbplt.set_subgrid(cubdat, OPTS.xmin,
                                        OPTS.xmax, OPTS.ymin,
                                        OPTS.ymax, OPTS.zmin,
                                        OPTS.zmax)

    # DEBUG
    # if OPTS.printVec or OPTS.saveSignedCube \
    #    or OPTS.matchEDTM:
    #     cubdat = cb.VtcdData(cubdat)
    #     mu_state = cubdat.mu_integrate()
    #     if not OPTS.matchEDTM:
    #         mg_state = cubdat.mag_integrate()
    #     if OPTS.printVec:
    #         dcube = cb.CubeData()
    #         dcube.natoms = 0
    #         dcube.origin = np.array([0., 0., 0.])
    #         dcube.npts = [2, 2, 2]
    #         dcube.nval = 3
    #         dcube.step = np.array([[1, 0, 0],
    #                                [0, 1, 0],
    #                                [0, 0, 1]])
    #         dcube.cube = np.zeros((3, 8))
    #         dcube.cube[:, 0] = mu_state
    #         cb.print_cube(dcube, 'ELC', OPTS.vibstate)
    #         dcube.cube[:, 0] = mg_state
    #         cb.print_cube(dcube, 'MAG', OPTS.vibstate)

    if OPTS.axis:
        # if OPTS.matchEDTM:
        #     # BUG
        #     # the cube and grind has been rotated
        #     RMAT = cubdat.rotate(mu_state)
        #     for ith in range(len(mol['crd'])):
        #         mol['crd'][ith] = list(np.dot(RMAT, mol['crd'][ith]))
        #         all_nmodes[int(idvstate-1),
        #                    ith*3:ith*3+3] = np.dot(RMAT, all_nmodes[int(idvstate-1),
        #                                                             ith*3:ith*3+3])

        fig0, ax0 = plt.subplots()
        _, x_ax, y_ax = cbplt.check_ax(OPTS.axis)
        if OPTS.printTitle:
            ax0.set_title('{}'.format(OPTS.printTitle))
        if OPTS.setlimit:
            tmp_b = [float(s) for s in OPTS.setlimit[1:-1].split(',')]
        elif OPTS.symplot:
            val1 = abs(max(ceil(10 * cubdat.box[x_ax, 0]),
                           floor(10 * cubdat.box[x_ax, -1])))/10
            val2 = abs(max(ceil(10 * cubdat.box[y_ax, 0]),
                           floor(10 * cubdat.box[y_ax, -1])))/10
            tmp_b = [-val1, val1, -val2, val2]
        else:
            tmp_b = [cubdat.box[x_ax, 0], cubdat.box[x_ax, -1],
                     cubdat.box[y_ax, 0], cubdat.box[y_ax, -1]]
        label = ['x', 'y', 'z']
        ax0.set_xlim(tmp_b[0:2])
        ax0.set_ylim(tmp_b[2:4])
        #ax0.text(-3,4.5,'x5', fontsize=20)
        ax0.set_xlabel(r'$\mathit{{{}}}$ axis / Bohr'.format(label[x_ax]))
        ax0.set_ylabel(r'$\mathit{{{}}}$ axis / Bohr'.format(label[y_ax]))
        cbplt.draw_mol2d(ax0, cubdat.crd, cubdat.ian, OPTS.axis, SCF,
                         conmat=AT_BONDS, to_bohr=True, vollimit=[OPTS.xmin, OPTS.xmax, OPTS.ymin, OPTS.ymax, OPTS.zmin, OPTS.zmax])
        if OPTS.printNM:
            cbplt.draw_nm2dcw(ax0, cubdat.crd, all_nmodes[int(idvstate-1)],
                            # OPTS.axis, OPTS.scaleNM, to_bohr=True)
                              cubdat.ian, OPTS.axis, OPTS.scaleNM, to_bohr=True)
        if OPTS.type == 'quiver':
            clim = [OPTS.qlower, OPTS.qupper]
            bkg = False
            if OPTS.quivbkg:
                bkg = True
            QUIV = cbplt.quiver_plt(ax0, cubdat, OPTS.axis, OPTS.vscale, background=bkg,
                                clim=clim, subsample=OPTS.subsample)
            fig0.savefig(os.path.join(ressubfolder, 'quiv2D.{}'.format(OPTS.figext)))
        elif OPTS.type == 'stream':
            STRM = cbplt.stream_plt(ax0, cubdat, OPTS.axis)
            fig0.savefig(os.path.join(ressubfolder, 'stream2D.{}'.format(OPTS.figext)))
        else:
            STRM = cbplt.stream2_plt(ax0, cubdat, OPTS.axis)
            if OPTS.type == 'animated_stream':
                ANIM = cbplt.animated_stream(fig0, STRM)
                plt.show()
            else:
                fig0.savefig(os.path.join(ressubfolder,
                                          'stream2D.{}'.format(OPTS.figext)))
        plt.close()

    #if OPTS.saveSignedCube:
    #    VEC1, VEC2 = cb.mask_cube(cubdat, mu_state)
    #    cb.print_cube(cubdat, 'Plus', OPTS.vibstate, vec_pr=VEC1)
    #    cb.print_cube(cubdat, 'Minus', OPTS.vibstate, vec_pr=VEC2)


if __name__ == '__main__':
    main()
