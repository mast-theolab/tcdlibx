#!/bin/python3
"""
Animated GIF of a vector cube (e.g. a VTCD or one of its Helmholtz components)
with moving arrows, while the nuclei vibrate along a normal mode.
See tcdlibx.graph.vfield_anim for the definition of the motion.
"""
import os
import json
import argparse

import numpy as np

import tcdlibx.calc.cube_manip as cb
import tcdlibx.io.fchk_io as fio
from tcdlibx.graph.vfield_anim import (ANIM_DEFAULTS, HARTREE2CM, VFieldAnimator,
                                       mode_displacements, grab_frame, save_gif)
from tcdlibx.utils.vtk_utils import vtk

PROGNAME = os.path.basename(__file__)


def build_parser():
    dft = ANIM_DEFAULTS
    par = argparse.ArgumentParser(prog=PROGNAME, description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    par.add_argument('fchk', help='Gaussian fchk with the normal modes')
    par.add_argument('cubefile', help='vector cube (nval=3) on the geometry of the fchk')
    par.add_argument('mode', type=int, help='normal mode (1-based)')
    par.add_argument('-o', '--output', required=True, help='output GIF')
    par.add_argument('--nucfactor', type=float, default=dft['nucfactor'],
                     help='factor on the physical amplitude of the nuclei (default %(default)s)')
    par.add_argument('--npoints', type=int, default=dft['npoints'],
                     help='number of arrows (default %(default)s)')
    par.add_argument('--scalevdw', type=float, default=dft['scalevdw'],
                     help='seeds in the shell between 0.1*s and s vdW radii (default %(default)s)')
    par.add_argument('--seed', type=int, default=dft['seed'], help='random seed (default %(default)s)')
    par.add_argument('--qupper-pct', type=float, default=dft['qupper_pct'],
                     help='percentile of |F| used as cap of the arrow length (default %(default)s)')
    par.add_argument('--qlower', type=float, default=dft['qlower'],
                     help='arrows weaker than qlower * reference are not drawn (default %(default)s)')
    par.add_argument('--qlower-ref', choices=('max', 'cap'), default=dft['qlower_ref'],
                     help='reference of --qlower: max|F| or the cap (default %(default)s)')
    par.add_argument('--arrowlen', type=float, default=dft['arrowlen'],
                     help='length (Angstrom) of an arrow at the cap (default %(default)s)')
    par.add_argument('--travel', type=float, default=dft['travel'],
                     help='travel of an arrow at the cap in units of --arrowlen (default %(default)s)')
    par.add_argument('--nframes', type=int, default=dft['nframes'],
                     help='frames per period (default %(default)s)')
    par.add_argument('--period', type=float, default=dft['period'],
                     help='period of the GIF in seconds (default %(default)s)')
    par.add_argument('--size', type=int, nargs=2, default=(900, 675))
    par.add_argument('--scene', help='tcdvis scene JSON (camera)')
    par.add_argument('--azimuth', type=float, default=0.)
    par.add_argument('--elevation', type=float, default=0.)
    par.add_argument('--zoom', type=float, default=1.)
    return par


def main():
    opts = build_parser().parse_args()
    vib = fio.fchk_vib_parser(opts.fchk)
    cubdat = cb.cube_parser(opts.cubefile)
    cubdat.make_box()
    if not (np.array_equal(np.asarray(vib['atnum']).astype(int), np.asarray(cubdat.ian).astype(int))
            and np.allclose(vib['crd'], cubdat.crd, atol=1e-4)):
        raise SystemExit('atoms or geometry differ between fchk and cube')
    drdq, omega = mode_displacements(np.asarray(vib['evec'])[opts.mode - 1], vib['atmas'],
                                     np.asarray(vib['frq'])[opts.mode - 1])
    anim_opts = {key: getattr(opts, key) for key in ANIM_DEFAULTS if hasattr(opts, key)}
    anim = VFieldAnimator(cubdat, cubdat.ian, drdq, omega, anim_opts)
    print(f'mode {opts.mode}: {omega * HARTREE2CM:.2f} cm-1, largest nuclear displacement '
          f'{anim.maxdisp:.3f} A, cap {anim.cap:.4e}, qlower {anim.low:.4e}, {anim.nseeds} arrows')

    ren = vtk.vtkRenderer()
    ren.SetBackground(1., 1., 1.)
    rwin = vtk.vtkRenderWindow()
    rwin.SetOffScreenRendering(1)
    rwin.AddRenderer(ren)
    rwin.SetSize(*opts.size)
    anim.add_to(ren)
    cam = ren.GetActiveCamera()
    if opts.scene:
        with open(opts.scene, encoding='utf-8') as fobj:
            scene = json.load(fobj)
        cam.SetPosition(scene['Camera:Position'])
        cam.SetFocalPoint(scene['Camera:FocalPoint'])
        cam.SetViewUp(scene['Camera:ViewUp'])
        cam.SetViewAngle(scene['Camera:ViewAngle'])
    else:
        ren.ResetCamera()
    cam.Azimuth(opts.azimuth)
    cam.Elevation(opts.elevation)
    cam.OrthogonalizeViewUp()
    cam.Zoom(opts.zoom)
    ren.ResetCameraClippingRange()

    frames = []
    for iframe in range(opts.nframes):
        anim.set_phase(ren, 2 * np.pi * iframe / opts.nframes)
        rwin.Render()
        frames.append(grab_frame(rwin))
    save_gif(frames, opts.output, opts.period)
    print(f'written {opts.output} ({len(frames)} frames)')


if __name__ == '__main__':
    main()
