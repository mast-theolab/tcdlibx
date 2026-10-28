"""
Animation of a vector field (e.g. a VTCD or one of its Helmholtz components)
with moving arrows, while the nuclei vibrate along a normal mode.

Time dependence (classical vibration of the mode):
  * nuclei: R(t) = R0 + (dR/dQ) Q0 f sin(wt), with Q0 = sqrt(3/w) the classical
    turning point of v = 1 (atomic units, mass-weighted normal coordinate) and
    f = nucfactor;
  * field: F(r, t) = F(r) cos(wt), the cube field being the amplitude when the
    nuclei cross the equilibrium geometry moving along +dR/dQ (no sign change).

Arrows start from points of the molecular volume (sample_molecular_volume) and
move along the field lines, x(t) = Phi(sin(wt)), where Phi solves
dPhi/dtau = k Fc(Phi): the velocity is k Fc cos(wt) and the motion is exactly
periodic.  Fc is the field with its length capped at qupper (direction kept).
The arrow length is arrowlen * |Fc|/qupper * |cos(wt)|, and k is chosen so that
an arrow at the cap travels `travel` times its own length between t = 0 and the
turning point.  Arrows weaker than qlower times the reference (max|F| or the
cap) are not drawn.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from estampes.data.physics import PHYSFACT
import tcdlibx.graph.cube_graphvtk as cubetk
from tcdlibx.graph.helpers import sample_molecular_volume, DEFAULT_PARAMETERS
from tcdlibx.utils.vtk_utils import vtk

AMU = 1822.888486
HARTREE2CM = 219474.6313705

# default animation parameters
ANIM_DEFAULTS = {'nucfactor': 3.0,
                 'npoints': 1500,
                 'scalevdw': 1.0,
                 'seed': 0,
                 'qupper_pct': 99.,
                 'qlower': 0.01,
                 'qlower_ref': 'cap',
                 'arrowlen': 0.4,
                 'travel': 1.0,
                 'nframes': 40,
                 'period': 2.4}


def mode_displacements(evec: np.ndarray, masses: np.ndarray,
                       freq: float) -> tuple[np.ndarray, float]:
    """dR/dQ (Bohr per au of Q) and w (Hartree) of a normal mode.

    Args:
        evec: Cartesian displacements of the mode (any normalisation), 3N or (N, 3)
        masses: atomic masses (amu)
        freq: harmonic wavenumber (cm-1)
    """
    masses = np.asarray(masses, dtype=float)
    lmw = np.sqrt(masses)[:, None] * np.asarray(evec, dtype=float).reshape(-1, 3)
    lmw /= np.linalg.norm(lmw)
    return lmw / np.sqrt(masses * AMU)[:, None], freq / HARTREE2CM


def capped(field: np.ndarray, cap: float) -> np.ndarray:
    """Field with the length capped at cap, keeping the direction."""
    norm = np.linalg.norm(field, axis=-1, keepdims=True)
    return np.where(norm > cap, field * cap / np.where(norm > 0, norm, 1.), field)


def trajectories(seeds: np.ndarray, interp, cap: float, kfac: float,
                 ntau: int = 40) -> np.ndarray:
    """Phi(tau) for tau in [-1, 1] (RK4), shape (2*ntau+1, nseeds, 3)."""
    def vel(x):
        return kfac * capped(interp(x), cap)
    dtau = 1. / ntau
    path = np.zeros((2 * ntau + 1,) + seeds.shape)
    path[ntau] = seeds
    for direction, rng in ((1, range(ntau, 2 * ntau)), (-1, range(ntau, 0, -1))):
        for i in rng:
            x = path[i]
            h = direction * dtau
            k1 = vel(x)
            k2 = vel(x + 0.5 * h * k1)
            k3 = vel(x + 0.5 * h * k2)
            k4 = vel(x + h * k3)
            path[i + direction] = x + h / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
    return path


class VFieldAnimator:
    """Moving arrows of a vector cube and vibrating nuclei in a VTK renderer.

    Args:
        cubdat: vector cube data (CubeData-like: npts, cube (3, N), crd and ian in Bohr)
        atnum: atomic numbers
        drdq: dR/dQ (Bohr per au of Q), shape (N, 3)
        omega: harmonic frequency (Hartree)
        opts: animation parameters (keys of ANIM_DEFAULTS, missing ones take the default)
        molconf: molecule style (keys of DEFAULT_PARAMETERS['molconfig'])
    """

    def __init__(self, cubdat, atnum, drdq: np.ndarray, omega: float,
                 opts: dict | None = None, molconf: dict | None = None):
        self.opts = dict(ANIM_DEFAULTS)
        self.opts.update(opts or {})
        self.molconf = dict(DEFAULT_PARAMETERS['molconfig'])
        self.molconf.update(molconf or {})
        if cubdat.nval != 3:
            raise ValueError('the cube must be a vector field (nval=3)')
        if not cubdat.box.size:
            cubdat.make_box()
        nx, ny, nz = cubdat.npts
        box = cubdat.box.reshape(3, nx, ny, nz)
        axes = [box[0, :, 0, 0], box[1, 0, :, 0], box[2, 0, 0, :]]
        if not all(np.all(np.diff(ax) > 0) for ax in axes):
            raise ValueError('cube grid is not orthogonal or not stored x-slowest')
        field = np.asarray(cubdat.cube).T.reshape(nx, ny, nz, 3)
        self.atnum = [int(z) for z in atnum]
        self.crd0 = np.asarray(cubdat.crd, dtype=float)
        self.disp = np.asarray(drdq) * np.sqrt(3. / omega) * self.opts['nucfactor']
        self.interp = RegularGridInterpolator(axes, field, bounds_error=False, fill_value=0.)
        norm = np.linalg.norm(field.reshape(-1, 3), axis=1)
        self.cap = float(np.percentile(norm, self.opts['qupper_pct']))
        ref = self.cap if self.opts['qlower_ref'] == 'cap' else norm.max()
        self.low = self.opts['qlower'] * ref
        np.random.seed(self.opts['seed'])
        seeds = sample_molecular_volume(cubdat, self.opts['npoints'],
                                        scale=self.opts['scalevdw']) / PHYSFACT.bohr2ang
        kfac = self.opts['travel'] * (self.opts['arrowlen'] / PHYSFACT.bohr2ang) / self.cap
        self.path = trajectories(seeds, self.interp, self.cap, kfac)
        self.nseeds = len(seeds)
        self._build_arrows()
        self.mol = None

    @property
    def maxdisp(self) -> float:
        """Largest nuclear displacement (Angstrom)."""
        return float(np.linalg.norm(self.disp, axis=1).max() * PHYSFACT.bohr2ang)

    def _build_arrows(self):
        self.pts = vtk.vtkPoints()
        self.pts.SetNumberOfPoints(self.nseeds)
        self.vecs = vtk.vtkDoubleArray()
        self.vecs.SetNumberOfComponents(3)
        self.vecs.SetNumberOfTuples(self.nseeds)
        self.vecs.SetName('vector')
        self.mags = vtk.vtkDoubleArray()
        self.mags.SetNumberOfTuples(self.nseeds)
        self.mags.SetName('scalar')
        poly = vtk.vtkPolyData()
        poly.SetPoints(self.pts)
        poly.GetPointData().SetVectors(self.vecs)
        poly.GetPointData().SetScalars(self.mags)
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(12)
        arrow.SetShaftResolution(12)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetColorModeToColorByScalar()
        lut = vtk.vtkLookupTable()  # colours of the tcdlibx streamlines
        lut.SetHueRange(0.6, 0.0)
        lut.Build()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0., 1.)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)

    def _at_tau(self, tau: float) -> np.ndarray:
        ntau = (self.path.shape[0] - 1) // 2
        s = (tau + 1.) * ntau
        i = min(int(np.floor(s)), self.path.shape[0] - 2)
        w = s - i
        return (1 - w) * self.path[i] + w * self.path[i + 1]

    def set_phase(self, ren, phase: float):
        """Update arrows and molecule in the renderer for the phase wt (radians)."""
        if self.mol is not None:
            ren.RemoveActor(self.mol.actor)
        self.mol = cubetk.fillmolecule_custom(
            self.atnum, (self.crd0 + np.sin(phase) * self.disp) * PHYSFACT.bohr2ang,
            opacity=self.molconf['opacity'], bond_radius=self.molconf['bond_radius'],
            atom_radius_scale=self.molconf['atom_radius_scale'],
            tubes_mode=self.molconf['tubes_mode'],
            bond_tollerance=self.molconf['bond_tollerance'])
        ren.AddActor(self.mol.actor)
        pos = self._at_tau(np.sin(phase))
        fval = capped(self.interp(pos), self.cap)
        fnorm = np.linalg.norm(fval, axis=1)
        scale = self.opts['arrowlen'] * np.cos(phase) / self.cap
        for k in range(self.nseeds):
            self.pts.SetPoint(k, *(pos[k] * PHYSFACT.bohr2ang))
            vec = fval[k] * scale if fnorm[k] >= self.low else (0., 0., 0.)
            self.vecs.SetTuple3(k, *vec)
            self.mags.SetValue(k, fnorm[k] / self.cap)
        self.pts.Modified()
        self.vecs.Modified()
        self.mags.Modified()

    def add_to(self, ren):
        ren.AddActor(self.actor)
        self.set_phase(ren, 0.)

    def remove_from(self, ren):
        ren.RemoveActor(self.actor)
        if self.mol is not None:
            ren.RemoveActor(self.mol.actor)
            self.mol = None


def grab_frame(rwin):
    """Current content of a VTK render window as an RGB numpy array."""
    from vtk.util import numpy_support as ns
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rwin)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = w2i.GetOutput()
    nx, ny, _ = img.GetDimensions()
    arr = ns.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(ny, nx, -1)
    return np.ascontiguousarray(arr[::-1, :, :3])


def save_gif(frames: list, fname: str, period: float):
    """Write RGB frames (numpy arrays) to a looping GIF of the given period (s)."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    palette = imgs[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    gif[0].save(fname, save_all=True, append_images=gif[1:], loop=0,
                duration=int(1000 * period / len(frames)), optimize=False)
