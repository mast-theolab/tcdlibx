# -*- coding: utf-8 -*-
"""
VTK objects
"""
# import sys
# import os
import typing as tp
import numpy as np

# Import VTK with error handling
from tcdlibx.utils.vtk_utils import vtk

from tcdlibx.calc.cube_manip import CubeData, VecCubeData
from tcdlibx.graph.helpers import MyvtkActor
# from math import ceil

def dots2vtkarray(dots: np.ndarray) -> vtk.vtkPolyData:
    points1 = vtk.vtkPoints()
    for coords in dots:
        points1.InsertNextPoint(coords)

    pointPolyData1 = vtk.vtkPolyData()
    pointPolyData1.SetPoints(points1)
    return pointPolyData1

def create_vector_field_polydata(positions: np.ndarray, 
                                vectors: np.ndarray,
                                scalars: tp.Optional[np.ndarray] = None,
                                vector_name: str = 'vectors',
                                scalar_name: str = 'scalars') -> vtk.vtkPolyData:
    """
    Create a vtkPolyData with vectors at specified positions in space.
    
    Args:
        positions: numpy array of shape (N, 3) with 3D positions
        vectors: numpy array of shape (N, 3) with 3D vectors at each position
        scalars: optional numpy array of shape (N,) with scalar values
        vector_name: name for the vector array
        scalar_name: name for the scalar array
        
    Returns:
        vtkPolyData with points, vectors, and optional scalars
        
    """
    n_points = positions.shape[0]
    assert vectors.shape == (n_points, 3), "Vectors must have shape (N, 3)"
    
    # Create VTK points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(n_points)
    
    # Create vector array
    vector_array = vtk.vtkDoubleArray()
    vector_array.SetNumberOfComponents(3)
    vector_array.SetNumberOfTuples(n_points)
    vector_array.SetName(vector_name)
    # If scalars are not provided filled with norm
    scalar_array = vtk.vtkDoubleArray()
    scalar_array.SetNumberOfComponents(1)
    scalar_array.SetNumberOfTuples(n_points)
    scalar_array.SetName(scalar_name)
    
    # Check if scalars are provided
    # written two times the loops to avoid the if inside the loop
    if scalars is not None:
        assert scalars.shape == (n_points,), "Scalars must have shape (N,)"

        # Fill points and vectors and scalars
        for i in range(n_points):
            points.SetPoint(i, positions[i])
            vector_array.SetTuple3(i, *vectors[i])
            scalar_array.SetValue(i, scalars[i])
    else:
        # Fill points and vectors only
        for i in range(n_points):
            points.SetPoint(i, positions[i])
            vector_array.SetTuple3(i, *vectors[i])
            scalar_array.SetValue(i, np.linalg.norm(vectors[i]))
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(vector_array)
    polydata.GetPointData().SetActiveVectors(vector_name)
    polydata.GetPointData().AddArray(scalar_array)
    polydata.GetPointData().SetActiveScalars(scalar_name)
    
    return polydata


def apply_spatial_clipping(grid_data: vtk.vtkImageData, 
                          clip_bounds: tp.Dict[str, tp.Optional[float]]) -> vtk.vtkImageData:
    """
    Apply spatial clipping to VTK ImageData using coordinate bounds.
    
    Args:
        grid_data: Input VTK ImageData to clip
        clip_bounds: Dictionary with coordinate bounds, e.g.:
                    {'xmin': -5.0, 'xmax': 5.0, 'ymin': None, 'ymax': 3.0, ...}
                    None values indicate no limit in that direction
    
    Returns:
        Clipped VTK ImageData
    """
    if not clip_bounds:
        return grid_data
    
    # Get grid dimensions and bounds
    dimensions = grid_data.GetDimensions()
    bounds = grid_data.GetBounds()  # [xmin, xmax, ymin, ymax, zmin, zmax]
    
    # Calculate grid spacing
    dx = (bounds[1] - bounds[0]) / max(1, dimensions[0] - 1)
    dy = (bounds[3] - bounds[2]) / max(1, dimensions[1] - 1) 
    dz = (bounds[5] - bounds[4]) / max(1, dimensions[2] - 1)
    
    # Convert coordinate bounds to grid indices
    def coord_to_index(coord, bounds_min, spacing, max_index):
        """Convert coordinate to grid index, clamped to valid range"""
        if coord is None:
            return None
        index = int((coord - bounds_min) / spacing)
        return max(0, min(index, max_index))
    
    # Calculate index bounds for each axis
    imin = coord_to_index(clip_bounds.get('xmin'), bounds[0], dx, dimensions[0] - 1)
    imax = coord_to_index(clip_bounds.get('xmax'), bounds[0], dx, dimensions[0] - 1)
    jmin = coord_to_index(clip_bounds.get('ymin'), bounds[2], dy, dimensions[1] - 1)
    jmax = coord_to_index(clip_bounds.get('ymax'), bounds[2], dy, dimensions[1] - 1)
    kmin = coord_to_index(clip_bounds.get('zmin'), bounds[4], dz, dimensions[2] - 1)
    kmax = coord_to_index(clip_bounds.get('zmax'), bounds[4], dz, dimensions[2] - 1)
    
    # Use default bounds if not specified
    if imin is None:
        imin = 0
    if imax is None:
        imax = dimensions[0] - 1
    if jmin is None:
        jmin = 0
    if jmax is None:
        jmax = dimensions[1] - 1
    if kmin is None:
        kmin = 0
    if kmax is None:
        kmax = dimensions[2] - 1
    
    # Ensure max >= min for each axis
    if imax < imin:
        imax = imin
    if jmax < jmin:
        jmax = jmin
    if kmax < kmin:
        kmax = kmin
    
    # Use vtkExtractVOI for vtkImageData
    extract_filter = vtk.vtkExtractVOI()
    extract_filter.SetInputData(grid_data)
    extract_filter.SetVOI(imin, imax, jmin, jmax, kmin, kmax)
    extract_filter.Update()
    return extract_filter.GetOutput()


def fillcubeimage(data, vec=True, logscale=False, aslist=False):
    """
    Fills a vtkImageData object

    Arguments:
        data {dict} -- dictionary with a cube dataset
        logscale {bool} -- if True store the logaritm 

    Returns:
        vtkImageData
    """
    if vec and data.nval != 3:
        vec = False
    def setupcube(npts, loc2wrd, narray):
        vtkimage = vtk.vtkImageData()
        vtkimage.SetDimensions(*npts)
        vtkimage.SetOrigin(*loc2wrd[:3, 3])
        vtkimage.SetSpacing(*np.diag(loc2wrd[:3,:3]))
        vtkimage.AllocateScalars(vtk.VTK_DOUBLE, narray)
        return vtkimage

    if vec:
        cubeimage = setupcube(data.npts, data.loc2wrd, narray=4)
        vect = vtk.vtkDoubleArray()
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfTuples(cubeimage.GetNumberOfPoints())
        vect.SetName('vector')
        norm = vtk.vtkDoubleArray()
        norm.SetNumberOfComponents(1)
        norm.SetNumberOfTuples(cubeimage.GetNumberOfPoints())
        norm.SetName('scalar')
    elif aslist:
        narray = data.nval
        scname = 'scalar'
        cubeimage = []
        for i in range(narray):
            cubeimage.append(setupcube(data.npts, data.loc2wrd, narray=1))
        arnpts = cubeimage[-1].GetNumberOfPoints()
    else:
        narray = data.nval
        scname = 'scalar{:02d}'
        cubeimage = setupcube(data.npts, data.loc2wrd, narray=narray)
        arnpts = cubeimage.GetNumberOfPoints()

    # NYI
    # cubeimage.AllocateScalars(vtk.VTK_DOUBLE, narray)
    # BUG only orthogonal grids
    # cubeimage.SetSpacing(*np.diag(data.loc2wrd[:3,:3]))
    if vec:
        scalar = np.sqrt(np.einsum("ij,ij->j", data.cube, data.cube))
        if logscale:
            scalar = np.log10(scalar+1e-13)
            scalar += 13
            scalar /= 1200
            # scalar[scalar < 3.1] = 0
        for i in range(data.npts[0]):
            for j in range(data.npts[1]):
                for k in range(data.npts[2]):
                    indices = (k * data.npts[1] + j) * data.npts[0] + i
                    indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                    # norval = np.sqrt(np.dot(data.cube[:, indicesf], data.cube[:, indicesf]))
                    norval = scalar[indicesf]
                    norm.SetValue(indices, norval)
                    vect.SetTuple3(indices, *data.cube[:, indicesf])
        cubeimage.GetPointData().AddArray(vect)
        cubeimage.GetPointData().AddArray(norm)
    elif narray == 1:
        norm = vtk.vtkDoubleArray()
        norm.SetNumberOfComponents(1)
        norm.SetNumberOfTuples(arnpts)
        norm.SetName("scalar")
        for i in range(data.npts[0]):
            for j in range(data.npts[1]):
                for k in range(data.npts[2]):
                    indices = (k * data.npts[1] + j) * data.npts[0] + i
                    indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                    norval = data.cube[indicesf]
                    norm.SetValue(indices, norval)
        cubeimage.GetPointData().AddArray(norm)
    else:
        for m in range(narray):
            norm = vtk.vtkDoubleArray()
            norm.SetNumberOfComponents(1)
            norm.SetNumberOfTuples(arnpts)
            norm.SetName(scname.format(m))
            for i in range(data.npts[0]):
                for j in range(data.npts[1]):
                    for k in range(data.npts[2]):
                        indices = (k * data.npts[1] + j) * data.npts[0] + i
                        indicesf = (i * data.npts[1] + j) * data.npts[2] + k
                        norval = data.cube[m, indicesf]
                        norm.SetValue(indices, norval)
            if aslist:
                cubeimage[m].GetPointData().AddArray(norm)
            else:
                cubeimage.GetPointData().AddArray(norm)
    # print(norm.GetRange())
    return cubeimage

def fillmolecule(atm, crd, 
                 opacity=1, bond_radius=None, atom_radius_scale=None,
                 tubes_mode=False,
                 bond_tollerance=0.23) -> MyvtkActor:
    """
    Create a VTK molecule visualization.

    Args:
        atm: List of atomic numbers
        crd: Coordinates array
        opacity (float): Molecule opacity (0.0-1.0)
        bond_radius (float): Custom bond radius
        atom_radius_scale (float): Custom atomic radius scale factor
        tubes_mode (bool): If True, atoms use same radius as bonds
    """
    # TODO implement a true bond perceiver
    mol = vtk.vtkMolecule()
    atoms = []
    for i in range(len(atm)):
        atoms.append(mol.AppendAtom(atm[i], *crd[i]))
    
    # Use VTK bond perceiver
    bond_perceiver = vtk.vtkSimpleBondPerceiver()
    bond_perceiver.SetInputData(mol)
    bond_perceiver.SetTolerance(bond_tollerance)
    bond_perceiver.Update()
    molout = bond_perceiver.GetOutput()
    
    mapper = vtk.vtkMoleculeMapper()
    mapper.UseLiquoriceStickSettings()
    # mapper.UseMultiCylindersForBondsOn()
    
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
    
    mapper.SetInputData(molout)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    return MyvtkActor(actor, mol)


def fillmolecule_custom(atm, crd, 
                 opacity=1, bond_radius=None, atom_radius_scale=None,
                 tubes_mode=False,
                 bond_tollerance=0.23,
                 excluded_atoms=None) -> MyvtkActor:
    """
    Create a VTK molecule visualization.

    Args:
        atm: List of atomic numbers
        crd: Coordinates array
        representation (legacy parameter)
        opacity (float): Molecule opacity (0.0-1.0)
        bond_radius (float): Custom bond radius
        atom_radius_scale (float): Custom atomic radius scale factor
        tubes_mode (bool): If True, atoms use same radius as bonds
    """
    # TODO implement a true bond perceiver
    mol = vtk.vtkMolecule()
    atoms = []
    for i in range(len(atm)):
        atoms.append(mol.AppendAtom(atm[i], *crd[i]))
    
    # Use VTK bond perceiver
    bond_perceiver = vtk.vtkSimpleBondPerceiver()
    bond_perceiver.SetInputData(mol)
    bond_perceiver.SetTolerance(bond_tollerance)
    bond_perceiver.Update()
    molout = bond_perceiver.GetOutput()
    
    mapper = vtk.vtkMoleculeMapper()
    mapper.UseLiquoriceStickSettings()
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

def _quivfromcube(cubdata: VecCubeData,
logscale=False) -> vtk.vtkImageData:
    """
    Return a vtkImageData object with the vector field
    from a VecCubeData object.
    """
    grid = fillcubeimage(cubdata, logscale=logscale)
    grid.GetPointData().SetActiveVectors('vector')
    grid.GetPointData().SetActiveScalars('scalar')
    return grid

def quiv3d(
    vecdata: tp.Union[VecCubeData, vtk.vtkPolyData],
    lower: float = 0.0001,
    upper: float = 0.01,
    scale: float = 1,
    logscale: bool = False,
    subsample_factor: tp.Optional[int] = None,
    glyphmode: str = 'arrow',
    clip_bounds: tp.Optional[tp.Dict[str, tp.Optional[float]]] = None
) -> MyvtkActor:
    """
    Return a vtk actor with 3D quiver plot with optional spatial clipping

    Args:
        vecdata: VecCubeData object or vtkPolyData with vector field
        lower/upper: lower and upper bounds to filter the vector field
        scale: scale factor for arrows
        logscale: if True, use logarithmic scaling
        subsample_factor: if provided, show only every nth vector for better performance
        glyphmode: 'arrow' or 'cone' for glyph type
        clip_bounds: dict with xmin, xmax, ymin, ymax, zmin, zmax for spatial clipping

    Returns:
        MyvtkActor with quiver plot
    """
    if isinstance(vecdata, VecCubeData):
        _grid = fillcubeimage(vecdata, logscale=logscale)
    elif isinstance(vecdata, vtk.vtkPolyData):
        _grid = vecdata
    else:
        raise TypeError("vecdata must be a VecCubeData or vtkPolyData object")

    _grid.GetPointData().SetActiveVectors('vector')
    _grid.GetPointData().SetActiveScalars('scalar')
    
    # Apply spatial clipping if requested
    if clip_bounds is not None and isinstance(vecdata, VecCubeData):
        _grid = apply_spatial_clipping(_grid, clip_bounds)

    # Apply subsampling if requested
    # Only apply subsampling for VecCubeData
    # For vtkPolyData, we assume it is from a clustering
    if subsample_factor is not None and subsample_factor > 1 and isinstance(vecdata, VecCubeData):
        # For image data, use masking
        geom_filter = vtk.vtkImageDataGeometryFilter()
        geom_filter.SetInputData(_grid)
        geom_filter.Update()

        mask = vtk.vtkMaskPoints()
        mask.SetInputConnection(geom_filter.GetOutputPort())
        mask.SetOnRatio(subsample_factor)
        mask.Update()
        _grid = mask.GetOutput()
    if glyphmode == 'arrow':
        vtkglyph = vtk.vtkArrowSource()
    elif glyphmode == 'cone':
        vtkglyph = vtk.vtkConeSource()
        vtkglyph.SetResolution(12)
        # vtkglyph.SetHeight(0.8)
        # vtkglyph.SetRadius(0.3)
    else:
        raise ValueError("glyphmode must be 'arrow' or 'cone'")
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(_grid)
    glyphs.SetSourceConnection(vtkglyph.GetOutputPort())
    # the mapper
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByScalar()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    glyphs.SetColorModeToColorByScalar()
    # color map
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(lower, 1, 0, 0)
    lut.AddRGBPoint(upper, 0, 1, 0)
    glyph_mapper.SetLookupTable(lut)
    # filtering
    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputData(_grid)
    threshold.SetLowerThreshold(lower)
    threshold.SetUpperThreshold(upper)
    glyphs.SetInputConnection(threshold.GetOutputPort())

    return MyvtkActor(glyph_actor, glyphs)


def draw_nm3d(crd, evec, ian,
              cngsign=True,
              chgwght=True, scale=1,
              color="blue"):
    """_summary_

    Args:
        crd (_type_): _description_
        evec (_type_): _description_
        ian (_type_): _description_
        cngsign (bool, optional): _description_. Defaults to True.
        chgwght (bool, optional): _description_. Defaults to True.
        scale (int, optional): _description_. Defaults to 1.
        color (str or tuple, optional): Color name (str) or RGB tuple (0.0-1.0). Defaults to "blue".

    Returns:
        _type_: _description_
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
    # PolyData
    # points = vtk.vtkPoints()
    # points.SetNumberOfPoints(natm)
    # vect = vtk.vtkDoubleArray()
    # vect.SetNumberOfComponents(3)
    # vect.SetNumberOfTuples(natm)
    # ones = vtk.vtkDoubleArray()
    # ones.SetNumberOfComponents(1)
    # ones.SetNumberOfTuples(natm)
    # for i in range(natm):
    #     points.SetPoint(i, crd[i, :])
    #     ones.SetValue(i, 1.)
    #     vect.SetTuple3(i, *norm_evec[i, :])
    # vect.SetName('vector')
    # ones.SetName('ones')
    # polydata = vtk.vtkPolyData()
    # polydata.SetPoints(points)
    # polydata.GetPointData().AddArray(vect)
    # polydata.GetPointData().AddArray(ones)
    # polydata.GetPointData().SetActiveVectors('vector')
    # polydata.GetPointData().SetActiveScalars('ones')
    polydata = create_vector_field_polydata(crd, norm_evec, np.ones(natm), vector_name='vector', scalar_name='ones')

    arrow = vtk.vtkArrowSource()
    # increase the resolution of the arrow for better visualization
    arrow.SetTipResolution(12)
    arrow.SetShaftResolution(12)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper =  vtk.vtkPolyDataMapper()
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

def draw_vectors(crd, vecs, tps, scale=1):
    """_summary_

    Args:
        crd (_type_): _description_
        vecs (_type_): _description_
        tps (_type_): [1,2,3,4] 4 total electric
                      [-1,-2,-3,-4] magnetic
        scale (int, optional): _description_. Defaults to 1.
        color (str, optional): _description_. Defaults to "blue".

    Returns:
        _type_: _description_
    """

    # weighted by the charge
    # norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)).max()
    norms = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))[0]
    # normalized
    norm_vecs = vecs / norms

    # multiply magnetic 1e3
    # norm_vecs[tps < 0,:]
    natm = int(crd.shape[0])
    # PolyData
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(natm)
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(natm)
    ones = vtk.vtkDoubleArray()
    ones.SetNumberOfComponents(1)
    ones.SetNumberOfTuples(natm)
    for i in range(natm):
        points.SetPoint(i, crd[i, :])
        ones.SetValue(i, tps[i])
        vect.SetTuple3(i, *norm_vecs[i, :])
    vect.SetName('vector')
    ones.SetName('ones')
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(vect)
    polydata.GetPointData().AddArray(ones)
    polydata.GetPointData().SetActiveVectors('vector')
    polydata.GetPointData().SetActiveScalars('ones')

    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(12)
    arrow.SetShaftResolution(12)
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    # the mapper
    glyph_mapper =  vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.VisibilityOn()

    # color map
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(-4, 102/255,0,102/255)
    lut.AddRGBPoint(-3, 153/255,0,0)
    lut.AddRGBPoint(-2, 1,0,0)
    lut.AddRGBPoint(-1, 1,128/255,0)
    # lut.AddRGBPoint(0, 0,1,0)
    lut.AddRGBPoint(1, 125/255,1,0)
    lut.AddRGBPoint(2, 0,1,0)
    lut.AddRGBPoint(3, 0,102/255,0)
    lut.AddRGBPoint(4, 0,102/255,51/255)
    glyph_mapper.SetLookupTable(lut)

    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    # Scale factor
    glyphs.SetScaleFactor(scale)
    glyphs.SetColorModeToColorByScalar()

    return MyvtkActor(glyph_actor, glyphs)


def fillstreamline(cubdata: CubeData,
                   nseeds: tp.Optional[int] = 150,
                   center: tp.Optional[list] = [0., 0., 0.],
                   opacity: tp.Optional[float] = 0.3,
                   clipping: tp.Optional[tuple] = (1e2, 1e5),
                   minspeed: tp.Optional[tp.Union[float, None]] = None,
                   seeds: tp.Optional[tp.Union[np.ndarray, None]] = None,
                   scale_rad=1,
                   clip_bounds: tp.Optional[tp.Dict[str, tp.Optional[float]]] = None) -> MyvtkActor:
    """Generate streamlines from vector field with optional spatial clipping.

    Args:
        cubdata (CubeData): Vector field cube data
        nseeds (tp.Optional[int], optional): Number of seed points. Defaults to 150.
        center (tp.Optional[list], optional): Center for seed distribution. Defaults to [0., 0., 0.].
        opacity (tp.Optional[float], optional): Streamline opacity. Defaults to 0.3.
        clipping (tp.Optional[tuple], optional): Magnitude clipping bounds. Defaults to (1e2, 1e5).
        minspeed (tp.Optional[tp.Union[float, None]], optional): Minimum speed for termination. Defaults to None.
        seeds (tp.Optional[tp.Union[np.ndarray, None]], optional): Custom seed points. Defaults to None.
        scale_rad (int, optional): Radius scaling factor. Defaults to 1.
        clip_bounds (tp.Optional[tp.Dict[str, tp.Optional[float]]], optional): 
            Spatial clipping bounds with keys: xmin, xmax, ymin, ymax, zmin, zmax. Defaults to None.

    Returns:
        MyvtkActor: VTK actor containing streamlines and optional clipped data reference
    """

    # Vectors stuff
    # https://stackoverflow.com/questions/57309203/plotting-vector-fields-efficiently-using-vtk-avoiding-excessive-looping
    _grid = fillcubeimage(cubdata)
    _grid.GetPointData().SetActiveVectors('vector')
    _grid.GetPointData().SetActiveScalars('scalar')
    
    # Apply spatial clipping if requested
    clipped_grid = _grid
    if clip_bounds is not None:
        clipped_grid = apply_spatial_clipping(_grid, clip_bounds)
    
    _bounds = clipped_grid.GetScalarRange()
    # visualizing only a portion between the two bounds
    _bounds2 = (_bounds[1]/clipping[1],
                _bounds[1]/clipping[0])
    # defining the seed points
    if seeds is None:
        _seeds = vtk.vtkPointSource()
        _seeds.SetCenter(*center)
        _seeds.SetNumberOfPoints(nseeds)
        _seeds.SetRadius(5.0)
        # possible different distributions, see at
        # https://vtk.org/doc/nightly/html/classvtkPointSource.html#a2029a3636eef7a32db31a10c9a904f9c
        # random distribution, seek how to get and save point positions
        _seeds.Update()
        _flag = True
    else:
        _seeds = dots2vtkarray(seeds)
        _flag = False

    # Streamlines stuff
    integrator=vtk.vtkRungeKutta45()
    streamline = vtk.vtkStreamTracer()
    streamline.SetInputData(clipped_grid)  # Use clipped grid
    if _flag:
        streamline.SetSourceConnection(_seeds.GetOutputPort())
    else:
        streamline.SetSourceData(_seeds)
    streamline.SetMaximumPropagation(50)
    streamline.SetIntegrator(integrator)
    streamline.SetInitialIntegrationStep(.1)
    streamline.SetIntegrationDirectionToBoth()
    streamline.SetComputeVorticity(True)
    if minspeed is None:
        minspeed = _bounds2[1] / 1e4
    streamline.SetTerminalSpeed(minspeed)
    # Building the tubes upone the streamlines
    streamtube = vtk.vtkTubeFilter()
    streamtube.SetInputConnection(streamline.GetOutputPort())
    # streamtube.SetInputArrayToProcess(3, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "norm")
    # streamtube.SetInputArrayToProcess(grid.GetPointData().GetNormals())
    streamtube.SetRadius(0.01)
    streamtube.SetNumberOfSides(12)
    # changes the radius according to the field magnitude
    streamtube.SetVaryRadiusToVaryRadiusByScalar()
    streamtube.CappingOn()
    streamtube.Update()
    # To change the colors
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6, 0.0)
    lut.Build()

    streamline_mapper = vtk.vtkPolyDataMapper()
    streamline_mapper.SetInputConnection(streamtube.GetOutputPort())
    # streamline_mapper.SetColorModeToMapScalars()
    # streamline_mapper.SetColorModeToMapScalars()
    streamline_mapper.ScalarVisibilityOn()
    # streamline_mapper.SetScalarModeToUsePointFieldData()
    streamline_mapper.SetScalarRange(*_bounds2)
    # streamline_mapper.ColorByArrayComponent("norm", 0)
    # streamline_mapper.SetLookupTable(ctf)
    streamline_mapper.SetLookupTable(lut)
    # streamline_mapper.SelectColorArray('norm')
    streamline_actor = vtk.vtkActor()
    streamline_actor.SetMapper(streamline_mapper)
    streamline_actor.GetProperty().SetOpacity(opacity)
    streamline_actor.VisibilityOn()
    
    # Create MyvtkActor with both streamtube and raw streamline data
    actor_wrapper = MyvtkActor(streamline_actor, streamtube)
    # Store the raw streamline data for particle animation
    actor_wrapper.streamline_data = streamline.GetOutput()
    
    return actor_wrapper

def countur(cubedata, isoval, active='scalar', colors=None, opacity=0.3):
    """
    Return vtk actor with the isosurface plotted
    param: xax, yax, zax 3d grid of dimension NxNxN
    val: values of the grid dots
    isoval: list of the wanted isovalue
    """
    assert len(isoval) < 3
    grid = fillcubeimage(cubedata)
    # grid.GetPointData().SetActiveVectors("vector")
    # grid.GetPointData().SetActiveScalars(active)
    return _countur(grid, isoval, active, colors, opacity)

def _countur(grid, isoval, active='scalar', colors=None, opacity=0.3):
    # grid.GetPointData().SetActiveVectors("vector")
    grid.GetPointData().SetActiveScalars(active)
    # bounds = grid.GetScalarRange()

    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputData(grid)
    contourFilter.SetArrayComponent(0)
    # set isoval and define lut
    if colors is None or len(colors) < len(isoval):
        colors = [[1,0,0], # red
                 [0,0,1]] # blue
    elif isinstance(colors[0], str):
        clrs = vtk.vtkNamedColors()
        colors = [clrs.GetColor3d(clr) for clr in colors]
    # Add checks
    lut = vtk.vtkColorTransferFunction()
    for i in range(len(isoval)):
        contourFilter.SetValue(i, isoval[i])
        lut.AddRGBPoint(isoval[i], *colors[i])
    contourFilter.Update()
    # mapper
    isosurf_mapper = vtk.vtkPolyDataMapper()
    isosurf_mapper.SetInputConnection(contourFilter.GetOutputPort())
    isosurf_mapper.ScalarVisibilityOn()
    # colors
    isosurf_mapper.SetLookupTable(lut)
    # actor
    isosurf_actor = vtk.vtkActor()
    isosurf_actor.SetMapper(isosurf_mapper)
    isosurf_actor.GetProperty().SetOpacity(opacity)
    isosurf_actor.VisibilityOn()

    return MyvtkActor(isosurf_actor, contourFilter)

def volumerendering(cubedata, active='scalar',
                    lower=0.0, upper=1.0,
                    opacity=1.0):
    grid = fillcubeimage(cubedata)
    return _volumerendering(grid, active, lower, upper, opacity)

def _volumerendering(grid: vtk.vtkImageData,
                     active: str = 'scalar',
                     lower: float = 0.0,
                     upper: float = 1.0,
                     opacity: float = 1.0) -> MyvtkActor:
    """
    Return a vtk actor with volume rendering
    from a vtkImageData object.
    Args:
        grid (vtk.vtkImageData): Input image data for volume rendering.
        lower (float): Lower bound for scalar range.
        upper (float): Upper bound for scalar range.
        opacity (float): Opacity for the volume rendering.  
    Returns:
        MyvtkActor: Actor containing the volume rendering.
    """
    grid.GetPointData().SetActiveScalars(active)
    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(lower, 0.0)
    opacityTransferFunction.AddPoint(upper, opacity)
    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(lower, 1.0, 0.0, 0.0)  # Red for low values
    colorTransferFunction.AddRGBPoint(upper, 0.0, 0.0, 1.0)  # Blue for high values
    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    # The mapper / ray cast function know how to render the data
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(grid)
    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    return MyvtkActor(volume, volumeMapper)


def draw_colorbar(targetactor: vtk.vtkActor, title: str,
                  nlabs: int = 5) -> MyvtkActor:
    """
    

    Args:
        targetactor (vtk.vtkActor): _description_
        title (str): _description_
        opacity (int, optional): _description_. Defaults to 1.
    """
    # Create a scalar bar
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(targetactor.GetMapper().GetLookupTable())
    scalarBar.SetTitle(title)
    scalarBar.SetNumberOfLabels(nlabs)
    # check these values
    scalarBar.SetMaximumWidthInPixels(100)
    scalarBar.SetMaximumHeightInPixels(400)
    scalarBar.SetPosition(0.1, 0.1)
    scalarBar.SetOrientationToVertical()
    scalarBar.GetTitleTextProperty().SetColor(0,0,0)
    scalarBar.GetTitleTextProperty().SetBold(True)

    return MyvtkActor(scalarBar, None)

def draw_cones_nogrid(cubdata: VecCubeData, point: np.ndarray, scale: float=0.1) -> MyvtkActor:
    npoints = point.shape[0]
    pcons = vtk.vtkPoints()
    pcons.SetNumberOfPoints(npoints)
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(npoints)
    ones = vtk.vtkDoubleArray()
    ones.SetNumberOfComponents(1)
    ones.SetNumberOfTuples(npoints)
    for i, val in enumerate(point):
        pcons.SetPoint(i, val)
        ones.SetValue(i, scale)
        vect.SetTuple3(i, *cubdata.get_value(val))
    vect.SetName('vector')
    ones.SetName('ones')
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pcons)
    polydata.GetPointData().AddArray(vect)
    polydata.GetPointData().AddArray(ones)
    polydata.GetPointData().SetActiveVectors('vector')
    polydata.GetPointData().SetActiveScalars('ones')
    cone = vtk.vtkConeSource()
    coneglyphs = vtk.vtkGlyph3D()
    coneglyphs.SetInputData(polydata)
    coneglyphs.SetSourceConnection(cone.GetOutputPort())
    coneglyph_mapper =  vtk.vtkPolyDataMapper()
    coneglyph_mapper.SetInputConnection(coneglyphs.GetOutputPort())
    coneglyph_actor = vtk.vtkActor()
    coneglyph_actor.SetMapper(coneglyph_mapper)
    coneglyph_actor.VisibilityOn()

    coneglyphs.SetVectorModeToUseVector()
    coneglyphs.SetScaleModeToScaleByScalar()
    return MyvtkActor(coneglyph_actor, coneglyphs)

def draw_ellipsoid(points: np.ndarray) -> MyvtkActor:
    """
    Draw an ellipsoid using the provided points.

    Args:
        points (np.ndarray): Array of 3D points defining the ellipsoid.

    Returns:
        MyvtkActor: Actor containing the ellipsoid representation.
    """
    # Create a point cloud
    pointPolyData = dots2vtkarray(points)

    # Create spheres at each point
    sphereFilter = vtk.vtkSphereSource()
    sphereFilter.SetRadius(0.05)
    sphereFilter.SetPhiResolution(8)
    sphereFilter.SetThetaResolution(8)

    # Use vtkGlyph3D to place spheres at each point
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetInputData(pointPolyData)
    glyph3D.SetSourceConnection(sphereFilter.GetOutputPort())
    glyph3D.Update()

    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    actor.GetProperty().SetOpacity(0.5)

    return MyvtkActor(actor, glyph3D)


class StreamlineParticleAnimator:
    """
    Handles animated particles along streamlines in VTK.
    """
    
    def __init__(self, renderer: vtk.vtkRenderer, particle_type: str = "sphere"):
        """
        Initialize the particle animator.
        
        Args:
            renderer: VTK renderer to add particles to
            particle_type: Type of particle to use ("sphere" or "cone")
        """
        self.renderer = renderer
        self.particle_actors = []
        self.is_animating = False
        self.particle_type = particle_type
        
    def create_particles(self, streamline_polydata: vtk.vtkPolyData, 
                        num_particles: int = 5) -> list[dict]:
        """
        Create animated particles along streamlines.
        
        Args:
            streamline_polydata: VTK polydata containing streamlines
            num_particles: Maximum number of particles to create
            
        Returns:
            List of particle data dictionaries
        """
        particles = []
        
        # Extract streamlines from polydata
        streamlines = self._extract_streamlines(streamline_polydata)
        
        # Create particles for each streamline
        for i, line_points in enumerate(streamlines):
            if i >= num_particles:  # Limit number of particles
                break
                
            if len(line_points) < 2:
                continue
            
            # Create particle actor based on type
            particle_actor = self._create_particle_actor()
            
            # Store particle data
            particle_data = {
                'actor': particle_actor,
                'points': line_points,
                'position': 0.0,
                'speed': 0.003 + i * 0.001,  # Slower speeds for better visibility
                'active': True,
                'transform': vtk.vtkTransform() if self.particle_type == "cone" else None
            }
            
            # Set initial position and orientation
            if len(line_points) > 0:
                # For cones, set orientation first, then position
                if self.particle_type == "cone" and len(line_points) > 1:
                    direction = self._calculate_direction(line_points[0], line_points[1])
                    self._orient_cone(particle_actor, direction, particle_data['transform'])
                
                # Set position after orientation for cones
                particle_actor.SetPosition(line_points[0])
            
            particles.append(particle_data)
            self.renderer.AddActor(particle_actor)
            
        self.particle_actors = particles
        return particles
    
    def _extract_streamlines(self, streamline_polydata: vtk.vtkPolyData) -> list[list]:
        """
        Extract individual streamlines from VTK polydata.
        
        Args:
            streamline_polydata: VTK polydata containing streamlines
            
        Returns:
            List of streamlines, each as a list of 3D points
        """
        streamlines = []
        streamline_polydata.GetLines().InitTraversal()
        
        id_list = vtk.vtkIdList()
        while streamline_polydata.GetLines().GetNextCell(id_list):
            line_points = []
            for i in range(id_list.GetNumberOfIds()):
                point_id = id_list.GetId(i)
                point = streamline_polydata.GetPoint(point_id)
                line_points.append(point)
            
            if len(line_points) > 1:
                streamlines.append(line_points)
                
        return streamlines
    
    def _create_particle_actor(self) -> vtk.vtkActor:
        """
        Create a particle actor based on the particle type.
        
        Returns:
            VTK actor for the particle
        """
        if self.particle_type == "cone":
            return self._create_cone_particle()
        else:
            return self._create_sphere_particle()
    
    def _create_sphere_particle(self) -> vtk.vtkActor:
        """
        Create a glowing sphere actor for a particle.
        
        Returns:
            VTK actor for the particle
        """
        # Create sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.03)  # Radius appropriate for atomic units scale
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        # Create actor with glowing appearance
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0)  # Bright orange
        actor.GetProperty().SetSpecular(0.9)
        actor.GetProperty().SetSpecularPower(100)
        actor.GetProperty().SetAmbient(0.7)  # Higher ambient light for glow effect
        actor.GetProperty().SetDiffuse(0.9)  # Higher diffuse for brightness
        
        return actor
    
    def _create_cone_particle(self) -> vtk.vtkActor:
        """
        Create a cone actor for a particle.
        
        Returns:
            VTK actor for the particle
        """
        # Create cone
        cone = vtk.vtkConeSource()
        cone.SetRadius(0.06)  # Increased radius for better visibility
        cone.SetHeight(0.12)  # Increased height for better visibility
        cone.SetResolution(12)
        cone.SetDirection(1, 0, 0)  # Default direction along X-axis
        cone.SetCenter(0, 0, 0)  # Center the cone at origin
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cone.GetOutputPort())
        
        # Create actor with glowing appearance
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0.8, 1)  # Bright cyan for cones
        actor.GetProperty().SetSpecular(0.9)
        actor.GetProperty().SetSpecularPower(100)
        actor.GetProperty().SetAmbient(0.7)  # Higher ambient light for glow effect
        actor.GetProperty().SetDiffuse(0.9)  # Higher diffuse for brightness
        
        # Store reference to cone source for later direction updates
        actor.cone_source = cone
        
        return actor
    
    def _calculate_direction(self, p1: tuple, p2: tuple) -> tuple:
        """
        Calculate the direction vector from p1 to p2.
        
        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)
            
        Returns:
            Normalized direction vector
        """
        import math
        
        # Calculate direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            return (dx/length, dy/length, dz/length)
        else:
            return (1, 0, 0)  # Default direction
    
    def _orient_cone(self, actor: vtk.vtkActor, direction: tuple, transform: vtk.vtkTransform):
        """
        Orient the cone to point in the given direction.
        
        Args:
            actor: VTK actor to orient
            direction: Direction vector (x, y, z)
            transform: VTK transform object to store the transformation
        """
        import math
        
        # Normalize direction vector
        dx, dy, dz = direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        else:
            dx, dy, dz = 1, 0, 0  # Default direction
        
        # Get the cone source from the stored reference
        if hasattr(actor, 'cone_source'):
            cone_source = actor.cone_source
            # Set the cone direction directly in the source
            cone_source.SetDirection(dx, dy, dz)
            cone_source.Update()
        
        # Don't apply any transform to the actor - let it use world coordinates directly
        actor.SetUserTransform(None)
    
    def update_particles(self) -> None:
        """
        Update particle positions along their streamlines.
        """
        if not self.particle_actors:
            return
            
        for particle in self.particle_actors:
            if not particle['active']:
                continue
                
            points = particle['points']
            if len(points) < 2:
                continue
                
            # Update position
            particle['position'] += particle['speed']
            if particle['position'] >= 1.0:
                particle['position'] = 0.0
            
            # Interpolate position along streamline
            interpolated_pos = self._interpolate_position(points, particle['position'])
            
            # For cones, update orientation first, then position
            if self.particle_type == "cone" and particle['transform'] is not None:
                direction = self._get_direction_at_position(points, particle['position'])
                self._orient_cone(particle['actor'], direction, particle['transform'])
            
            # Set position after orientation for cones
            particle['actor'].SetPosition(interpolated_pos)
    
    def _get_direction_at_position(self, points: list, position: float) -> tuple:
        """
        Get the direction vector at a specific position along the streamline.
        
        Args:
            points: List of 3D points defining the streamline
            position: Normalized position along streamline (0-1)
            
        Returns:
            Direction vector at the given position
        """
        total_segments = len(points) - 1
        segment_pos = position * total_segments
        segment_idx = int(segment_pos)
        
        if segment_idx >= total_segments:
            segment_idx = total_segments - 1
        
        # Get direction from current segment
        if segment_idx + 1 < len(points):
            p1 = points[segment_idx]
            p2 = points[segment_idx + 1]
            return self._calculate_direction(p1, p2)
        else:
            # Use previous segment direction
            if segment_idx > 0:
                p1 = points[segment_idx - 1]
                p2 = points[segment_idx]
                return self._calculate_direction(p1, p2)
            else:
                return (1, 0, 0)  # Default direction
    
    def _interpolate_position(self, points: list, position: float) -> list:
        """
        Interpolate position along a streamline.
        
        Args:
            points: List of 3D points defining the streamline
            position: Normalized position along streamline (0-1)
            
        Returns:
            Interpolated 3D position
        """
        total_segments = len(points) - 1
        segment_pos = position * total_segments
        segment_idx = int(segment_pos)
        local_pos = segment_pos - segment_idx
        
        if segment_idx >= total_segments:
            segment_idx = total_segments - 1
            local_pos = 1.0
        
        # Linear interpolation between points
        p1 = points[segment_idx]
        p2 = points[segment_idx + 1] if segment_idx + 1 < len(points) else points[segment_idx]
        
        return [
            p1[0] + local_pos * (p2[0] - p1[0]),
            p1[1] + local_pos * (p2[1] - p1[1]),
            p1[2] + local_pos * (p2[2] - p1[2])
        ]
    
    def start_animation(self) -> None:
        """
        Start the particle animation.
        """
        self.is_animating = True
        for particle in self.particle_actors:
            particle['active'] = True
    
    def stop_animation(self) -> None:
        """
        Stop the particle animation and remove all particles.
        """
        self.is_animating = False
        
        # Remove all particle actors from renderer
        for particle in self.particle_actors:
            self.renderer.RemoveActor(particle['actor'])
            particle['active'] = False
        
        self.particle_actors = []
    
    def set_particle_speed(self, speed_multiplier: float) -> None:
        """
        Adjust the speed of all particles.
        
        Args:
            speed_multiplier: Multiplier for particle speeds
        """
        for i, particle in enumerate(self.particle_actors):
            base_speed = 0.003 + i * 0.001
            particle['speed'] = base_speed * speed_multiplier
    
    def set_particle_visibility(self, visible: bool) -> None:
        """
        Show or hide all particles.
        
        Args:
            visible: Whether particles should be visible
        """
        for particle in self.particle_actors:
            particle['actor'].SetVisibility(visible)
    
    def update_particle_type(self, new_particle_type: str, streamline_polydata: vtk.vtkPolyData, num_particles: int = None) -> None:
        """
        Update the particle type for existing animation.
        
        Args:
            new_particle_type: New particle type ("sphere" or "cone")
            streamline_polydata: VTK polydata containing streamlines
            num_particles: New number of particles (optional, keeps current if None)
        """
        if new_particle_type == self.particle_type and (num_particles is None or num_particles == len(self.particle_actors)):
            return  # No change needed
            
        # Store animation state
        was_animating = self.is_animating
        
        # Stop current animation and remove particles
        self.stop_animation()
        
        # Update particle type
        self.particle_type = new_particle_type
        
        # Create new particles with new type
        if num_particles is None:
            num_particles = len(self.particle_actors) if self.particle_actors else 5
        
        self.create_particles(streamline_polydata, num_particles)
        
        # Restart animation if it was running
        if was_animating:
            self.start_animation()
    
    def get_particle_count(self) -> int:
        """
        Get the current number of particles.
        
        Returns:
            Number of active particles
        """
        return len(self.particle_actors)
    
    def force_render_update(self) -> None:
        """
        Force the renderer to update the display.
        """
        if hasattr(self, 'ren') and self.ren is not None:
            self.ren.GetRenderWindow().Render()


def create_streamline_particles(renderer: vtk.vtkRenderer, 
                               polydata: vtk.vtkPolyData,
                               num_particles: int = 10,
                               particle_type: str = "sphere") -> StreamlineParticleAnimator:
    """
    Factory function to create a StreamlineParticleAnimator instance.
    
    Args:
        renderer: VTK renderer to add particles to
        polydata: VTK polydata containing streamline data
        num_particles: Number of particles to create
        particle_type: Type of particles ("sphere" or "cone")
        
    Returns:
        StreamlineParticleAnimator instance with particles created
    """
    animator = StreamlineParticleAnimator(renderer, particle_type)
    
    # Only create particles if polydata has lines/points
    if polydata.GetNumberOfPoints() > 0 and polydata.GetNumberOfCells() > 0:
        animator.create_particles(polydata, num_particles)
    
    return animator



