"""
Visualization of fields from cube data: quiver plots, streamlines, isosurfaces,
volume rendering, plus small scene decorations (color bar, seed points).
"""
import numpy as np
from collections.abc import Sequence

from tcdlibx.utils.vtk_utils import vtk
from tcdlibx.calc.cube_manip import CubeData, VecCubeData
from tcdlibx.graph.helpers import MyvtkActor
from tcdlibx.graph.vtk_clipping import apply_spatial_clipping
from tcdlibx.graph.vtk_cube import fillcubeimage
from tcdlibx.graph.vtk_geometry import create_vector_field_polydata, dots2vtkarray


def quiv3d(
    vecdata: VecCubeData | vtk.vtkPolyData,
    lower: float = 0.0001,
    upper: float = 0.01,
    scale: float = 1,
    logscale: bool = False,
    subsample_factor: int | None = None,
    glyphmode: str = 'arrow',
    clip_bounds: dict[str, float | None] | None = None
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


def fillstreamline(cubdata: CubeData,
                   nseeds: int = 150,
                   center: Sequence[float] = (0., 0., 0.),
                   opacity: float = 0.3,
                   clipping: tuple[float, float] = (1e2, 1e5),
                   minspeed: float | None = None,
                   seeds: np.ndarray | None = None,
                   scale_rad: float = 1,
                   clip_bounds: dict[str, float | None] | None = None) -> MyvtkActor:
    """Generate streamlines from vector field with optional spatial clipping.

    Args:
        cubdata (CubeData): Vector field cube data
        nseeds (int, optional): Number of seed points. Defaults to 150.
        center (Sequence[float], optional): Center for seed distribution. Defaults to (0., 0., 0.).
        opacity (float, optional): Streamline opacity. Defaults to 0.3.
        clipping (tuple, optional): Magnitude clipping bounds. Defaults to (1e2, 1e5).
        minspeed (float | None, optional): Minimum speed for termination. Defaults to None.
        seeds (np.ndarray | None, optional): Custom seed points. Defaults to None.
        scale_rad (int, optional): Radius scaling factor. Defaults to 1.
        clip_bounds (dict[str, float | None] | None, optional):
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
    integrator = vtk.vtkRungeKutta45()
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
    actor_wrapper.streamline_data = streamline.GetOutput()  # pyright: ignore[reportAttributeAccessIssue]

    return actor_wrapper


def countur(cubedata: CubeData, isoval: list[float], active: str = 'scalar',
            colors: list | None = None, opacity: float = 0.3) -> MyvtkActor:
    """
    Return vtk actor with the isosurface plotted
    cubedata: scalar cube dataset
    isoval: list of the wanted isovalues
    active: name of the array to contour
    colors: one color per isovalue (names or RGB)
    """
    assert len(isoval) < 3
    grid = fillcubeimage(cubedata)
    # grid.GetPointData().SetActiveVectors("vector")
    # grid.GetPointData().SetActiveScalars(active)
    return _countur(grid, isoval, active, colors, opacity)


def _countur(grid: vtk.vtkImageData, isoval: list[float], active: str = 'scalar',
             colors: list | None = None, opacity: float = 0.3) -> MyvtkActor:
    # grid.GetPointData().SetActiveVectors("vector")
    grid.GetPointData().SetActiveScalars(active)
    # bounds = grid.GetScalarRange()

    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputData(grid)
    contourFilter.SetArrayComponent(0)
    # set isoval and define lut
    if colors is None or len(colors) < len(isoval):
        colors = [[1, 0, 0],  # red
                  [0, 0, 1]]  # blue
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


def volumerendering(cubedata: CubeData, active: str = 'scalar',
                    lower: float = 0.0, upper: float = 1.0,
                    opacity: float = 1.0) -> MyvtkActor:
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
    Create a scalar bar using the lookup table of an actor.

    Args:
        targetactor (vtk.vtkActor): actor whose mapper lookup table is shown
        title (str): title of the bar
        nlabs (int, optional): number of labels. Defaults to 5.
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
    scalarBar.GetTitleTextProperty().SetColor(0, 0, 0)
    scalarBar.GetTitleTextProperty().SetBold(True)

    return MyvtkActor(scalarBar, None)


def draw_cones_nogrid(cubdata: VecCubeData, point: np.ndarray, scale: float = 0.1) -> MyvtkActor:
    vectors = np.array([cubdata.get_value(val) for val in point])
    polydata = create_vector_field_polydata(point, vectors, np.full(point.shape[0], scale),
                                            vector_name='vector', scalar_name='ones')
    cone = vtk.vtkConeSource()
    coneglyphs = vtk.vtkGlyph3D()
    coneglyphs.SetInputData(polydata)
    coneglyphs.SetSourceConnection(cone.GetOutputPort())
    coneglyph_mapper = vtk.vtkPolyDataMapper()
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
