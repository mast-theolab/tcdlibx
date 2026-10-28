"""
Low-level helpers to build VTK point sets and vector-field polydata.
"""
import numpy as np

from tcdlibx.utils.vtk_utils import vtk


def dots2vtkarray(dots: np.ndarray) -> vtk.vtkPolyData:
    """Return a vtkPolyData holding the (N, 3) points (no cells)."""
    points1 = vtk.vtkPoints()
    for coords in dots:
        points1.InsertNextPoint(coords)

    pointPolyData1 = vtk.vtkPolyData()
    pointPolyData1.SetPoints(points1)
    return pointPolyData1


def create_vector_field_polydata(positions: np.ndarray,
                                 vectors: np.ndarray,
                                 scalars: np.ndarray | None = None,
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
