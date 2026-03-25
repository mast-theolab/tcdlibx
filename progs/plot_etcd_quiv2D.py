#!/bin/python3
"""
Plot Electronic Transition Current Density (ETCD) in 2D with quiver plots.

Example: python3 plot_ETCD_quiv2D.py -a y --vscale=100 \
        TCD_S001.cube
"""
import sys
import os
# import re
import argparse
from math import ceil, floor
# import numpy as np
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import custom library

from estampes.data.physics import PHYSFACT

import tcdlibx.calc.cube_manip as cb
import tcdlibx.graph.cube_graphic as cbplt
from tcdlibx.utils.color_out import Colors
# import variabiles
from tcdlibx.utils.mol_data import ELEMENTS # , AT_COL2, AT_RAD

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
    txt = "Transition current density cube file to visualize"
    par.add_argument('cubefile', help=txt)
    # OPTIONAL ARGUMENTS
    par.add_argument('-p', '--print', action='store_true',
                     help='Print molecule')
    draw = par.add_argument_group('Drawing parameters')
    draw.add_argument('-a', '--axis', choices=('x', 'y', 'z'), default='z',
                      help='Axis for the projection')
    draw.add_argument('--vscale', type=float, default=5.,
                      help='Overall scaling factor for the arrows')
    draw.add_argument('-g', '--grid', type=int,
                      help='Sets the step for the grid construction')
    draw.add_argument('--xmin', type=float,
                      help='Lower bound along x for the current vectors')
    draw.add_argument('--xmax', type=float,
                      help='Upper bound along x for the current vectors')
    draw.add_argument('--ymin', type=float,
                      help='Lower bound along y for the current vectors')
    draw.add_argument('--ymax', type=float,
                      help='Upper bound along y for the current vectors')
    draw.add_argument('--zmin', type=float,
                      help='Lower bound along z for the current vectors')
    draw.add_argument('--zmax', type=float,
                      help='Upper bound along z for the current vectors')
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
    draw.add_argument('--quivbkg', action='store_true',
                      help='''Quiver background based on norm of the vector field''') 
    draw.add_argument('--figext', type=str,
                      choices=('pdf', 'eps', 'png'),
                      default='pdf',
                      help='Figure extension')
    # Molecule Drawing Parameters
    molecule = par.add_argument_group('Drawing Molecule Parameters')
    molecule.add_argument('--scaleNuc', type=float, default=1.0,
                          help='''Enter the scaling factor for Nucleus circles''')
    molecule.add_argument('--scaleBond', type=float, default=1.0,
                          help='''Enter the scaling factor for Bonds''')

    return par


if __name__ == '__main__':
    cp = Colors()
    PARSER = build_parser()
    OPTS = PARSER.parse_args()
    # Scale to use print_mol2d
    SCF = [OPTS.scaleNuc, OPTS.scaleBond]
    
    # Check that data files exist
    if not os.path.exists(OPTS.cubefile):
        print(cp.printred('ERROR: cube file {} does not exist'.format(OPTS.cubefile)))
        sys.exit()
    # Set grid step
    if OPTS.grid:
        ngrdstp = OPTS.grid
    else:
        ngrdstp = NSTEP_BOX
    # Get molecule information
    # Parse and load cube data
    print('Loading molecular and transition current density from cube file...')
    cubdat = cb.cube_parser(OPTS.cubefile)
    cubdat.make_box()


    if OPTS.print:
        print('''
########################################
###          MOLECULAR DATA          ###
########################################

Coordinates (in Bohr)
''')
        fmt = 'Atom {:3d}  {:2s}  {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}'
        for ia in range(cubdat.natoms):
            ian = cubdat.ian[ia]
            xyz = cubdat.crd[ia, :] / PHYSFACT.bohr2ang
            print(fmt.format(ia+1, ELEMENTS[ian], c=xyz))

    # Apply subgrid if specified
    if OPTS.xmin or OPTS.xmax or OPTS.ymin or \
       OPTS.ymax or OPTS.zmin or OPTS.zmax:
        cubdat.cube = cbplt.set_subgrid(cubdat,
                                OPTS.xmin, OPTS.xmax, OPTS.ymin,
                                OPTS.ymax, OPTS.zmin, OPTS.zmax)

    # 2D projection plotting
    resfolder = os.path.join("Q2D_plot")
    if not os.path.exists(resfolder):
        os.mkdir(resfolder)
    
    vec2, box2 = cbplt.simp_proj(cubdat, OPTS.axis)
    fig0, ax0 = plt.subplots()
    if OPTS.printTitle:
        ax0.set_title('{}'.format(OPTS.printTitle))
    if OPTS.setlimit:
        tmp = [float(s) for s in OPTS.setlimit[1:-1].split(',')]
    elif OPTS.symplot:
        val1 = abs(max(ceil(10 * box2[0, 0]),
                       floor(10 * box2[0, -1])))/10
        val2 = abs(max(ceil(10 * box2[1, 0]),
                       floor(10 * box2[1, -1])))/10
        tmp = [-val1, val1, -val2, val2]
    else:
        tmp = [box2[0, 0], box2[0, -1], box2[1, 0], box2[1, -1]]
    
    _, x_ax, y_ax = cbplt.check_ax(OPTS.axis)
    ax0.set_xlim(tmp[0:2])
    ax0.set_ylim(tmp[2:4])
    label = ['x', 'y', 'z']
    ax0.set_xlabel(r'$\mathit{{{}}}$ axis / Bohr'.format(label[x_ax]))
    ax0.set_ylabel(r'$\mathit{{{}}}$ axis / Bohr'.format(label[y_ax]))
    
    AT_BONDS = cbplt.get_connect(cubdat.ian, cubdat.crd)
    # Draw molecule
    cbplt.draw_mol2d(ax0, cubdat.crd, cubdat.ian, OPTS.axis, SCF,
                     conmat=AT_BONDS, to_bohr=True, vollimit=[OPTS.xmin, OPTS.xmax, OPTS.ymin, OPTS.ymax, OPTS.zmin, OPTS.zmax])
    
    # Draw current density vectors
    if OPTS.type == 'quiver':
        bkg = False
        if OPTS.quivbkg:
            bkg = True
        QUIV = cbplt.quiver_plt(ax0, cubdat, OPTS.axis, OPTS.vscale, background=bkg)
        fig0.savefig(os.path.join(resfolder, 'quiv2D_axis{}.{}'.format(OPTS.axis, OPTS.figext)))
    elif OPTS.type == 'stream':
        STRM = cbplt.stream_plt(ax0, cubdat, OPTS.axis)
        fig0.savefig(os.path.join(resfolder, 'stream2D_axis{}.{}'.format(OPTS.axis, OPTS.figext)))
    else:
        STRM = cbplt.stream2_plt(ax0, cubdat, OPTS.axis)
        if OPTS.type == 'animated_stream':
            ANIM = cbplt.animated_stream(fig0, STRM)
            plt.show()
        else:
            fig0.savefig(os.path.join(resfolder,
                                          'stream2D_axis{}.{}'.format(OPTS.axis, OPTS.figext)))
        plt.close()


    # colma = plt.get_cmap("seismic")
    # quiv = ax0.quiver(box2[0, :], box2[1, :], vec2[0, :],
    #                   vec2[1, :], units='width',
    #                   scale=OPTS.vscale, cmap=colma,
    #                   zorder=3)
    
    fig0.savefig(os.path.join(resfolder, 'etcd_quiv2D.pdf'))
    plt.close()
    print('2D plot saved as {}'.format(os.path.join(resfolder, 'etcd_quiv2D.pdf')))

