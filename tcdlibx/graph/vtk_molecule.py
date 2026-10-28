"""
Molecule-related VTK actors: structure, normal modes and dipole-moment vectors.
"""
from collections.abc import Sequence

import numpy as np

from tcdlibx.utils.vtk_utils import vtk
from tcdlibx.graph.helpers import MyvtkActor
from tcdlibx.graph.vtk_geometry import create_vector_field_polydata


def fillmolecule(atm: np.ndarray, crd: np.ndarray,
                 opacity: float = 1,
                 bond_radius: float | None = None,
                 atom_radius_scale: float | None = None,
                 tubes_mode: bool = False,
                 bond_tollerance: float = 0.23) -> MyvtkActor:
    """
    Create a VTK molecule visualization (single-cylinder bonds).
    See fillmolecule_custom for the arguments.
    """
    return fillmolecule_custom(atm, crd, opacity=opacity, bond_radius=bond_radius,
                               atom_radius_scale=atom_radius_scale,
                               tubes_mode=tubes_mode,
                               bond_tollerance=bond_tollerance,
                               multi_cylinder_bonds=False)


def fillmolecule_custom(atm: np.ndarray, crd: np.ndarray,
                        opacity: float = 1,
                        bond_radius: float | None = None,
                        atom_radius_scale: float | None = None,
                        tubes_mode: bool = False,
                        bond_tollerance: float = 0.23,
                        excluded_atoms: list[int] | None = None,
                        multi_cylinder_bonds: bool = True) -> MyvtkActor:
    """
    Create a VTK molecule visualization.

    Args:
        atm: List of atomic numbers
        crd: Coordinates array
        opacity (float): Molecule opacity (0.0-1.0)
        bond_radius (float): Custom bond radius
        atom_radius_scale (float): Custom atomic radius scale factor
        tubes_mode (bool): If True, atoms use same radius as bonds
        bond_tollerance (float): Tolerance of the VTK bond perceiver
        excluded_atoms: bonds with both ends in this list are hidden
        multi_cylinder_bonds (bool): draw multiple bonds as several cylinders
    """
    # TODO implement a true bond perceiver
    mol = vtk.vtkMolecule()
    for i in range(len(atm)):
        mol.AppendAtom(atm[i], *crd[i])

    # Use VTK bond perceiver
    bond_perceiver = vtk.vtkSimpleBondPerceiver()
    bond_perceiver.SetInputData(mol)
    bond_perceiver.SetTolerance(bond_tollerance)
    bond_perceiver.Update()
    molout = bond_perceiver.GetOutput()

    mapper = vtk.vtkMoleculeMapper()
    mapper.UseLiquoriceStickSettings()
    if multi_cylinder_bonds:
        mapper.UseMultiCylindersForBondsOn()

    # Apply bond radius if provided
    if bond_radius is not None:
        mapper.SetBondRadius(bond_radius)

    # Apply atom radius scale
    if tubes_mode and bond_radius is not None:
        # In tubes mode, atoms use the same size as bonds
        mapper.SetAtomicRadiusScaleFactor(bond_radius)
    elif atom_radius_scale is not None:
        # Normal mode - use the specified atom radius scale
        mapper.SetAtomicRadiusScaleFactor(atom_radius_scale)

    if excluded_atoms is not None:
        for i in range(molout.GetNumberOfBonds()):
            bond = molout.GetBond(i)
            atom1_id = bond.GetBeginAtomId()
            atom2_id = bond.GetEndAtomId()
            if atom1_id in excluded_atoms and atom2_id in excluded_atoms:
                molout.SetBondOrder(i, 0)  # Exclude this bond

    mapper.SetInputData(molout)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    return MyvtkActor(actor, mol)


def draw_nm3d(crd: np.ndarray, evec: np.ndarray, ian: np.ndarray,
              cngsign: bool = True,
              chgwght: bool = True,
              scale: float = 1,
              color: str | Sequence[float] = 'blue') -> MyvtkActor:
    """ Draw a 3D quiver plot of the normal modes.

    Args:
        crd (np.ndarray): atomic coordinates
        evec (np.ndarray): normal mode eigenvectors
        ian (np.ndarray): atomic numbers
        cngsign (bool, optional): Invert the phase of the eigenvectors. Defaults to True.
        chgwght (bool, optional): Weight the eigenvectors by the atomic numbers. Defaults to True.
        scale (float, optional): Scale factor for the arrows. Defaults to 1.
        color (str | tuple, optional): Color name (str) or RGB tuple (0.0-1.0). Defaults to 'blue'.

    Returns:
        MyvtkActor: actor with the arrows, uniformly colored
    """

    # change the sign to be opposed to the TCD (must be done before weighting)
    if cngsign:
        evec = evec * -1

    # weighted by the charge
    if chgwght:
        levec = evec * np.array(ian)[:, np.newaxis]
    else:
        levec = evec

    norms = np.sqrt(np.einsum('ij,ij->i', levec, levec)).max()
    # normalized
    norm_evec = levec / norms
    natm = int(crd.shape[0])
    polydata = create_vector_field_polydata(crd, norm_evec, np.ones(natm), vector_name='vector', scalar_name='ones')

    arrow = vtk.vtkArrowSource()
    # increase the resolution of the arrow for better visualization
    arrow.SetTipResolution(12)
    arrow.SetShaftResolution(12)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    # Turn off scalar visibility on mapper to use uniform color
    glyph_mapper.ScalarVisibilityOff()
    clrs = vtk.vtkNamedColors()

    # Handle color parameter - accept both string names and RGB tuples
    if isinstance(color, (tuple, list)):
        # RGB tuple/list (values 0.0-1.0)
        glyph_actor.GetProperty().SetColor(color[0], color[1], color[2])
    else:
        # String color name
        glyph_actor.GetProperty().SetColor(clrs.GetColor3d(color))

    return MyvtkActor(glyph_actor, glyphs)


def draw_vectors(crd: np.ndarray, vecs: np.ndarray, tps: np.ndarray, scale: float = 1) -> MyvtkActor:
    """Draw one arrow per vector (e.g. total/partial dipole moments) at the given positions.

    Args:
        crd (np.ndarray): positions of the arrows, shape (N, 3)
        vecs (np.ndarray): vectors, shape (N, 3)
        tps (np.ndarray): type of each vector, used for the color:
                      [1,2,3,4] 4 total electric
                      [-1,-2,-3,-4] magnetic
        scale (float, optional): scale factor for the arrows. Defaults to 1.

    Returns:
        MyvtkActor: actor with the arrows
    """

    # weighted by the charge
    # TODO: unsure whether to normalize by the first vector (previous behavior, [0]) or
    # by the largest one; max is used for now, to be checked with tests
    norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)).max()
    # normalized
    norm_vecs = vecs / norms

    # multiply magnetic 1e3
    # norm_vecs[tps < 0,:]
    polydata = create_vector_field_polydata(crd, norm_vecs, np.asarray(tps, dtype=float),
                                            vector_name='vector', scalar_name='ones')

    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(12)
    arrow.SetShaftResolution(12)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    # color map
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(-4, 102/255, 0, 102/255)
    lut.AddRGBPoint(-3, 153/255, 0, 0)
    lut.AddRGBPoint(-2, 1, 0, 0)
    lut.AddRGBPoint(-1, 1, 128/255, 0)
    # lut.AddRGBPoint(0, 0,1,0)
    lut.AddRGBPoint(1, 125/255, 1, 0)
    lut.AddRGBPoint(2, 0, 1, 0)
    lut.AddRGBPoint(3, 0, 102/255, 0)
    lut.AddRGBPoint(4, 0, 102/255, 51/255)
    glyph_mapper.SetLookupTable(lut)

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    glyphs.SetColorModeToColorByScalar()

    return MyvtkActor(glyph_actor, glyphs)
