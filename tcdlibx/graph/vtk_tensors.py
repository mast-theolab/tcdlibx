"""
Tensor visualization: vector glyphs and norm-based surfaces for 3x3 tensors (APT, AAT).
"""
import numpy as np

from tcdlibx.utils.vtk_utils import vtk
from tcdlibx.graph.helpers import (MyvtkActor, VibMolecule,
                                   fibonacci_spiral_samples_on_unit_sphere,
                                   compute_atom_mask)
from tcdlibx.graph.vtk_geometry import create_vector_field_polydata, dots2vtkarray


def draw_tensor_spheres(positions: np.ndarray,
                        tensors: np.ndarray,
                        sphere_radius: float = 0.5,
                        nb_sphere_samples: int = 50,
                        vector_scale: float = 0.3,
                        opacity: float = 0.8,
                        show_spheres: bool = True,
                        weights: np.ndarray | None = None,
                        color_scheme: str = 'magnitude') -> MyvtkActor:
    """
    Visualize 3x3 tensors at given positions as vectors on unit spheres.

    Args:
        positions (np.ndarray): Array of shape (N, 3) with tensor positions
        tensors (np.ndarray): Array of shape (N, 3, 3) with 3x3 tensors
        sphere_radius (float): Radius of the visualization spheres
        nb_sphere_samples (int): Number of fibonacci samples on each sphere
        vector_scale (float): Scale factor for the vectors
        opacity (float): Opacity of the spheres
        show_spheres (bool): Whether to show the underlying spheres
        weights (np.ndarray): Masses, required for the 'weight_mag' color scheme
        color_scheme (str): Color scheme for vectors ('magnitude', 'weight_mag', 'uniform')

    Returns:
        MyvtkActor: VTK actor containing the tensor visualization
    """

    if positions.shape[0] != tensors.shape[0]:
        raise ValueError("Number of positions must match number of tensors")

    if tensors.shape[1:] != (3, 3):
        raise ValueError("Tensors must be 3x3 matrices")

    # Generate fibonacci samples on unit sphere
    unit_sphere_points = fibonacci_spiral_samples_on_unit_sphere(nb_sphere_samples)

    # Prepare arrays for all vectors across all atoms
    all_points = []
    all_vectors = []
    all_scalars = []

    for atom_idx, (pos, tensor) in enumerate(zip(positions, tensors)):
        # Scale sphere points by radius and translate to atom position
        sphere_points = unit_sphere_points * sphere_radius + pos

        # Apply tensor to each unit vector to get the vector field
        tensor_vectors = np.zeros_like(sphere_points)
        for i, unit_vec in enumerate(unit_sphere_points):
            tensor_vectors[i] = np.dot(tensor, unit_vec)

        # Calculate scalars for coloring
        if color_scheme == 'magnitude':
            scalars = np.linalg.norm(tensor_vectors, axis=1)
        elif color_scheme == 'weight_mag':
            if weights is None:
                raise ValueError("Weights must be provided for 'weight_mag' color scheme")
            scalars = np.linalg.norm(tensor_vectors, axis=1)/np.sqrt(weights[atom_idx])
        else:  # uniform
            scalars = np.ones(nb_sphere_samples)

        all_points.extend(sphere_points)
        all_vectors.extend(tensor_vectors)
        all_scalars.extend(scalars)

    # Convert to numpy arrays
    all_points = np.array(all_points)
    all_vectors = np.array(all_vectors)
    all_scalars = np.array(all_scalars)

    # Create polydata with all vectors
    polydata = create_vector_field_polydata(
        all_points,
        all_vectors,
        all_scalars,
        vector_name='tensor_vectors',
        scalar_name=color_scheme
    )

    # Create arrows for vector visualization
    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(12)
    arrow.SetShaftResolution(12)

    # Create glyphs
    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByScalar()
    glyphs.SetScaleFactor(vector_scale)
    glyphs.SetColorModeToColorByScalar()

    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyphs.GetOutputPort())

    # Set up color mapping
    if color_scheme == 'magnitude' or color_scheme == 'weight_mag':
        # Create a color lookup table
        lut = vtk.vtkColorTransferFunction()

        # Get the scalar range for proper color mapping
        scalar_min = np.min(all_scalars)
        scalar_max = np.max(all_scalars)
        scalar_range = scalar_max - scalar_min

        if scalar_range > 0:
            # Map colors based on actual scalar range
            lut.AddRGBPoint(scalar_min, 0, 0, 1)                    # Blue for minimum
            lut.AddRGBPoint(scalar_min + 0.5 * scalar_range, 0, 1, 0)  # Green for medium
            lut.AddRGBPoint(scalar_max, 1, 0, 0)                    # Red for maximum
        else:
            # Fallback if all scalars are the same
            lut.AddRGBPoint(scalar_min, 0, 1, 0)  # Green for uniform values

        mapper.SetLookupTable(lut)
        # Set the scalar range for the mapper
        mapper.SetScalarRange(scalar_min, scalar_max)
    else:  # uniform
        mapper.ScalarVisibilityOff()

    # Create actor for vectors
    vector_actor = vtk.vtkActor()
    vector_actor.SetMapper(mapper)
    vector_actor.GetProperty().SetOpacity(opacity)

    # Create sphere actors if requested
    if show_spheres:
        # Create spheres at tensor positions
        sphere_points = dots2vtkarray(positions)

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(sphere_radius * 1)  # Change 1 to scale the spheres
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)

        sphere_glyph = vtk.vtkGlyph3D()
        sphere_glyph.SetInputData(sphere_points)
        sphere_glyph.SetSourceConnection(sphere_source.GetOutputPort())
        sphere_glyph.Update()

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_glyph.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 1.0, 0.6)  # Pale yellow
        sphere_actor.GetProperty().SetOpacity(0.3)

        # Create an assembly to combine both actors
        assembly = vtk.vtkAssembly()
        assembly.AddPart(vector_actor)
        assembly.AddPart(sphere_actor)

        return MyvtkActor(assembly, glyphs)
    else:
        return MyvtkActor(vector_actor, glyphs)


def _get_filtered_tensors(mol_data: VibMolecule, tensor_type: str, tensor_level: str,
                          atom_filter: list[str] | None,
                          frag_filter: list[int] | None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the atoms and the (natoms, 3, 3) tensors to be drawn.

    Returns:
        (atom_mask, positions, tensors) restricted to the selected atoms
    """
    if not hasattr(mol_data, 'crd') or not hasattr(mol_data, '_moldata'):
        raise ValueError("mol_data must have 'crd' and '_moldata' attributes")
    positions = mol_data.crd
    natoms = positions.shape[0]
    # combined atomic symbol + fragment index filter
    atom_mask = compute_atom_mask(mol_data, atom_filter=atom_filter, frag_filter=frag_filter)

    kind = tensor_type.lower()
    if kind not in ('apt', 'aat'):
        raise ValueError("tensor_type must be 'apt' or 'aat'")
    total = getattr(mol_data, kind)
    if total is None:
        raise ValueError(f"{kind.upper()} tensor data not available in molecule")
    if tensor_level == 'tot':
        data = total
    elif tensor_level in ('ele', 'nuc'):
        data = mol_data._moldata[kind + tensor_level]
    else:
        raise ValueError("tensor_level must be 'tot', 'ele', or 'nuc'")
    # stored as (natoms*3, 3) or already (natoms, 3, 3)
    full_tensors = data.reshape(natoms, 3, 3) if data.ndim == 2 else data
    return atom_mask, positions[atom_mask], full_tensors[atom_mask]


def draw_molecular_tensors(mol_data: VibMolecule,
                           tensor_type: str = 'apt',
                           tensor_level: str = 'tot',
                           sphere_radius: float = 0.5,
                           nb_sphere_samples: int = 50,
                           vector_scale: float = 2,
                           opacity: float = 0.8,
                           color_scheme: str = 'magnitude',
                           atom_filter: list[str] | None = None,
                           frag_filter: list[int] | None = None,
                           show_spheres: bool = False) -> MyvtkActor:
    """
    Convenience function to visualize molecular tensors from VibMolecule data.

    Args:
        mol_data: VibMolecule (or object with compatible interface)
        tensor_type (str): Type of tensor to visualize ('apt' or 'aat')
        tensor_level (str): Contribution to visualize ('tot', 'ele', or 'nuc')
        sphere_radius (float): Radius of the visualization spheres
        nb_sphere_samples (int): Number of fibonacci samples on each sphere
        vector_scale (float): Scale factor for the vectors
        opacity (float): Opacity of the visualization
        color_scheme (str): Color scheme for vectors ('magnitude', 'weight_mag', 'uniform')
        atom_filter (list): List of atomic symbols to filter (e.g., ['C', 'N', 'O'])
        frag_filter (list): List of fragment indices (0-based) to restrict to. Combined
            with atom_filter via AND (an atom must satisfy both filters when both are set)
        show_spheres (bool): Whether to show pale yellow spheres at tensor positions

    Returns:
        MyvtkActor: VTK actor containing the tensor visualization
    """
    atom_mask, filtered_positions, filtered_tensors = _get_filtered_tensors(
        mol_data, tensor_type, tensor_level, atom_filter, frag_filter)

    # Get masses for weight_mag color scheme
    if color_scheme == 'weight_mag':
        full_masses = mol_data.atmas
        filtered_masses = full_masses[atom_mask]
    else:
        filtered_masses = None

    return draw_tensor_spheres(
        positions=filtered_positions,
        tensors=filtered_tensors,
        sphere_radius=sphere_radius,
        nb_sphere_samples=nb_sphere_samples,
        vector_scale=vector_scale,
        opacity=opacity,
        show_spheres=show_spheres,
        weights=filtered_masses,
        color_scheme=color_scheme
    )


def draw_tensor_surface_from_norms(positions: np.ndarray,
                                   tensors: np.ndarray,
                                   sphere_radius: float = 0.5,
                                   nb_sphere_samples: int = 500,
                                   surface_scale: float = 1.0,
                                   opacity: float = 0.7,
                                   weights: np.ndarray | None = None,
                                   color_scheme: str = 'magnitude') -> MyvtkActor:
    """
    Generate independent surfaces on each atom using the norm of vectors computed with tensor operations.
    Each atom gets its own separate surface mesh.
    The process follows these steps:
    1. Compute vectors from tensor x position on sphere points (T·r)
    2. Calculate the magnitude/norm of these vectors
    3. Create new vectors using displacement directions with magnitudes as the norm
    4. Convert to point cloud and create surface using vtkDelaunay3D and vtkDataSetSurfaceFilter
    5. Create separate surface for each atom and combine them using vtkAppendPolyData

    Args:
        positions (np.ndarray): Array of shape (N, 3) with tensor positions
        tensors (np.ndarray): Array of shape (N, 3, 3) with 3x3 tensors
        sphere_radius (float): Base radius for the sampling spheres
        nb_sphere_samples (int): Number of fibonacci samples on each sphere
        surface_scale (float): Scale factor for the surface based on magnitudes
        opacity (float): Opacity of the surface
        weights (np.ndarray): Masses, required (and used to scale the magnitudes) for 'weight_mag'
        color_scheme (str): Color scheme for the surface ('magnitude', 'weight_mag', 'uniform', 'signed')

    Returns:
        MyvtkActor: VTK actor containing the independent tensor-based surfaces
    """
    if positions.shape[0] != tensors.shape[0]:
        raise ValueError("Number of positions must match number of tensors")

    if tensors.shape[1:] != (3, 3):
        raise ValueError("Tensors must be 3x3 matrices")

    # Generate fibonacci samples on unit sphere
    unit_sphere_points = fibonacci_spiral_samples_on_unit_sphere(nb_sphere_samples)

    # Create independent surfaces for each atom
    individual_surfaces = []
    all_scalars_global = []

    for atom_idx, (pos, tensor) in enumerate(zip(positions, tensors)):
        # Step 1: Compute vectors from tensor × position on sphere points
        tensor_vectors = np.zeros_like(unit_sphere_points)
        magnitudes = np.zeros(nb_sphere_samples)
        cosine_values = np.zeros(nb_sphere_samples)  # For signed color scheme

        for i, unit_vec in enumerate(unit_sphere_points):
            # Apply tensor to unit vector: T·r
            tensor_result = np.dot(tensor, unit_vec)
            tensor_vectors[i] = tensor_result

            # Step 2: Calculate magnitude/norm of the vectors
            magnitude = np.linalg.norm(tensor_result)
            magnitudes[i] = magnitude

            # Calculate cosine of angle between unit_vec and tensor_result for signed color scheme
            if magnitude > 1e-12:  # Avoid division by zero
                # Normalize tensor_result to get unit vector
                tensor_result_normalized = tensor_result / magnitude
                # Calculate cosine of angle (dot product of unit vectors)
                cosine_values[i] = np.dot(unit_vec, tensor_result_normalized)
            else:
                cosine_values[i] = 0.0

            # Apply weights only for the 'weight_mag' scheme
            if color_scheme == 'weight_mag':
                if weights is None:
                    raise ValueError("Weights must be provided for 'weight_mag' color scheme")
                magnitudes[i] /= np.sqrt(weights[atom_idx])

        # Step 3: Create new vectors using displacement directions with computed magnitudes
        # Direction = unit vector on sphere, Magnitude = norm from tensor operation
        surface_points = np.zeros_like(unit_sphere_points)

        for i, unit_vec in enumerate(unit_sphere_points):
            # New vector = direction * (base_radius + scale * magnitude)
            radius_at_point = sphere_radius + surface_scale * magnitudes[i]
            surface_points[i] = pos + unit_vec * radius_at_point

        # Step 4: Create individual surface for this atom
        # Convert to point cloud
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(surface_points))

        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfTuples(len(surface_points))
        scalars.SetName('tensor_magnitude')

        # Choose scalar values based on color scheme
        if color_scheme == 'signed':
            scalar_values = cosine_values
        else:
            scalar_values = magnitudes

        for i, (point, scalar) in enumerate(zip(surface_points, scalar_values)):
            points.SetPoint(i, point)
            scalars.SetValue(i, scalar)

        # Create polydata for this atom's point cloud
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(scalars)

        # Create Delaunay 3D triangulation for this atom only
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        # Extract the surface from this atom's 3D mesh
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputConnection(delaunay.GetOutputPort())
        surface_filter.Update()

        # Store this atom's surface
        individual_surfaces.append(surface_filter.GetOutput())

        # Add appropriate scalars to global collection
        if color_scheme == 'signed':
            all_scalars_global.extend(cosine_values)
        else:
            all_scalars_global.extend(magnitudes)

    # Step 5: Combine all individual surfaces using vtkAppendPolyData
    append_filter = vtk.vtkAppendPolyData()
    for surface in individual_surfaces:
        append_filter.AddInputData(surface)
    append_filter.Update()

    # Create mapper for the combined surfaces
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append_filter.GetOutputPort())

    # Set up color mapping using global scalar range
    if color_scheme in ('magnitude', 'weight_mag'):
        # Create a color lookup table
        lut = vtk.vtkColorTransferFunction()

        # Get the global scalar range for proper color mapping across all atoms
        scalar_min = np.min(all_scalars_global)
        scalar_max = np.max(all_scalars_global)
        scalar_range = scalar_max - scalar_min

        if scalar_range > 0:
            # Map colors based on actual scalar range
            lut.AddRGBPoint(scalar_min, 0, 0, 1)                      # Blue for minimum
            lut.AddRGBPoint(scalar_min + 0.5 * scalar_range, 0, 1, 0)  # Green for medium
            lut.AddRGBPoint(scalar_max, 1, 0, 0)                      # Red for maximum
        else:
            # Fallback if all scalars are the same
            lut.AddRGBPoint(scalar_min, 0, 1, 0)  # Green for uniform values

        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(scalar_min, scalar_max)
    elif color_scheme == 'signed':
        # Create a color lookup table for signed values (-1 to 1)
        lut = vtk.vtkColorTransferFunction()

        # Red for -1 (cosine = -1, opposite direction)
        lut.AddRGBPoint(-1.0, 1, 0, 0)  # Red
        # White for 0 (cosine = 0, perpendicular)
        lut.AddRGBPoint(0.0, 1, 1, 1)   # White
        # Blue for 1 (cosine = 1, same direction)
        lut.AddRGBPoint(1.0, 0, 0, 1)   # Blue

        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(-1.0, 1.0)
    else:  # uniform
        mapper.ScalarVisibilityOff()

    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)

    # Set uniform color if not using magnitude/signed coloring
    if color_scheme == 'uniform':
        actor.GetProperty().SetColor(0.8, 0.6, 0.2)  # Golden color
    else:
        # For magnitude/signed coloring, ensure property doesn't interfere with scalar coloring
        # Set InterpolateScalarsBeforeMapping to ensure smooth color transitions
        mapper.SetInterpolateScalarsBeforeMapping(True)

    return MyvtkActor(actor, append_filter)


def draw_molecular_tensor_surface(mol_data: VibMolecule,
                                  tensor_type: str = 'apt',
                                  tensor_level: str = 'tot',
                                  sphere_radius: float = 0.5,
                                  nb_sphere_samples: int = 50,
                                  surface_scale: float = 1.0,
                                  opacity: float = 0.7,
                                  color_scheme: str = 'magnitude',
                                  atom_filter: list[str] | None = None,
                                  frag_filter: list[int] | None = None) -> MyvtkActor:
    """
    Create tensor-based surfaces from molecular data using norm-based surface generation.

    Args:
        mol_data: VibMolecule (or object with compatible interface)
        tensor_type (str): Type of tensor to visualize ('apt' or 'aat')
        tensor_level (str): Contribution to visualize ('tot', 'ele', or 'nuc')
        sphere_radius (float): Base radius for the sampling spheres
        nb_sphere_samples (int): Number of fibonacci samples on each sphere
        surface_scale (float): Scale factor for surface modulation based on magnitudes
        opacity (float): Opacity of the surface
        color_scheme (str): Color scheme for the surface ('magnitude', 'weight_mag', 'uniform', 'signed')
        atom_filter (list): List of atomic symbols to filter (e.g., ['C', 'N', 'O'])
        frag_filter (list): List of fragment indices (0-based) to restrict to. Combined
            with atom_filter via AND (an atom must satisfy both filters when both are set)

    Returns:
        MyvtkActor: VTK actor containing the tensor-based surface
    """
    atom_mask, filtered_positions, filtered_tensors = _get_filtered_tensors(
        mol_data, tensor_type, tensor_level, atom_filter, frag_filter)

    # Get masses for the weight_mag color scheme
    if color_scheme == 'weight_mag':
        filtered_masses = mol_data.atmas[atom_mask]
    else:
        filtered_masses = None

    return draw_tensor_surface_from_norms(
        positions=filtered_positions,
        tensors=filtered_tensors,
        sphere_radius=sphere_radius,
        nb_sphere_samples=nb_sphere_samples,
        surface_scale=surface_scale,
        opacity=opacity,
        weights=filtered_masses,
        color_scheme=color_scheme
    )
