"""
Spatial clipping of cube data and the preview planes shown in the scene.
"""
from tcdlibx.utils.vtk_utils import vtk


def apply_spatial_clipping(grid_data: vtk.vtkImageData,
                           clip_bounds: dict[str, float | None]) -> vtk.vtkImageData:
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
    def coord_to_index(coord: float | None, bounds_min: float,
                       spacing: float, max_index: int) -> int | None:
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


def create_clip_plane_actors(
    scene_bounds: tuple[float, float, float, float, float, float],
    initial_values: dict[str, float],
) -> tuple[dict[str, "vtk.vtkActor"], dict[str, "vtk.vtkPlaneSource"]]:
    """Create six semi-transparent plane actors for clip-bound preview.

    One actor per bound key (``'xmin'``, ``'xmax'``, ``'ymin'``, ``'ymax'``,
    ``'zmin'``, ``'zmax'``).  Yellow = min bound, Blue = max bound.
    All actors are created with ``VisibilityOff()``.

    Args:
        scene_bounds: ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the VTK scene
                      in Bohr, used as the extent of each plane.
        initial_values: mapping from bound key to initial coordinate value;
                        should cover all six keys.

    Returns:
        ``(actors, sources)`` — two dicts keyed by bound string, holding the
        ``vtkActor`` and ``vtkPlaneSource`` respectively.
    """
    bnd = scene_bounds
    colors = {
        'xmin': (1.0, 1.0, 0.0), 'xmax': (0.0, 0.0, 1.0),
        'ymin': (1.0, 1.0, 0.0), 'ymax': (0.0, 0.0, 1.0),
        'zmin': (1.0, 1.0, 0.0), 'zmax': (0.0, 0.0, 1.0),
    }
    actors: dict[str, vtk.vtkActor] = {}
    sources: dict[str, vtk.vtkPlaneSource] = {}

    for key in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'):
        source = vtk.vtkPlaneSource()
        source.SetResolution(10, 10)
        move_clip_plane(source, key, initial_values[key], bnd)
        source.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        r, g, b = colors[key]
        actor.GetProperty().SetColor(r, g, b)
        actor.GetProperty().SetOpacity(0.35)
        actor.GetProperty().LightingOff()
        actor.VisibilityOff()

        sources[key] = source
        actors[key] = actor

    return actors, sources


def move_clip_plane(
    source: "vtk.vtkPlaneSource",
    key: str,
    val: float,
    scene_bounds: tuple[float, float, float, float, float, float],
) -> None:
    """Update the geometry of a single clip-plane source.

    Args:
        source: the ``vtkPlaneSource`` to update.
        key: bound key (``'xmin'``, ``'xmax'``, ``'ymin'``, etc.).
        val: new coordinate value in Bohr.
        scene_bounds: ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the scene.
    """
    bnd = scene_bounds
    axis = key[0]
    if axis == 'x':
        source.SetOrigin(val, bnd[2], bnd[4])
        source.SetPoint1(val, bnd[3], bnd[4])
        source.SetPoint2(val, bnd[2], bnd[5])
    elif axis == 'y':
        source.SetOrigin(bnd[0], val, bnd[4])
        source.SetPoint1(bnd[1], val, bnd[4])
        source.SetPoint2(bnd[0], val, bnd[5])
    else:  # 'z'
        source.SetOrigin(bnd[0], bnd[2], val)
        source.SetPoint1(bnd[1], bnd[2], val)
        source.SetPoint2(bnd[0], bnd[3], val)
    source.Modified()
