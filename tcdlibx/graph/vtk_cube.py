"""
Conversion of cube datasets into VTK images.
"""
import numpy as np
from vtkmodules.util import numpy_support

from tcdlibx.utils.vtk_utils import vtk
from tcdlibx.calc.cube_manip import CubeData, VecCubeData


def fillcubeimage(data: CubeData, vec: bool = True, logscale: bool = False,
                  aslist: bool = False) -> vtk.vtkImageData | list[vtk.vtkImageData]:
    """
    Convert a CubeData into vtkImageData.

    Args:
        data: cube dataset
        vec: if True and nval == 3, store the field as 'vector' and its norm as 'scalar'
        logscale: with vec, store the rescaled log10 of the norm
        aslist: for scalar fields, return one image per field instead of one image
            with several arrays

    Returns:
        vtkImageData, or a list of vtkImageData when aslist is used on a scalar field
    """
    if vec and data.nval != 3:
        vec = False
    nx, ny, nz = data.npts
    # cube files are z-fastest, VTK is x-fastest: vtk_order[vtk_index] = cube_index
    vtk_order = np.arange(nx * ny * nz).reshape(nx, ny, nz).transpose(2, 1, 0).ravel()

    def setupcube(narray: int) -> vtk.vtkImageData:
        vtkimage = vtk.vtkImageData()
        vtkimage.SetDimensions(*data.npts)
        vtkimage.SetOrigin(*data.loc2wrd[:3, 3])
        # BUG only orthogonal grids
        vtkimage.SetSpacing(*np.diag(data.loc2wrd[:3, :3]))
        vtkimage.AllocateScalars(vtk.VTK_DOUBLE, narray)
        return vtkimage

    def vtkarray(values: np.ndarray, name: str) -> vtk.vtkDoubleArray:
        array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(values, dtype=np.float64), deep=True)
        array.SetName(name)
        return array

    if vec:
        cubeimage = setupcube(4)
        scalar = np.sqrt(np.einsum("ij,ij->j", data.cube, data.cube))
        if logscale:
            scalar = np.log10(scalar+1e-13)
            scalar += 13
            scalar /= 1200
        cubeimage.GetPointData().AddArray(vtkarray(data.cube[:, vtk_order].T, 'vector'))
        cubeimage.GetPointData().AddArray(vtkarray(scalar[vtk_order], 'scalar'))
        return cubeimage

    # nval == 1 cubes are stored as 1D arrays
    fields = data.cube.reshape(data.nval, -1)
    if aslist:
        images = []
        for m in range(data.nval):
            image = setupcube(1)
            image.GetPointData().AddArray(vtkarray(fields[m][vtk_order], 'scalar'))
            images.append(image)
        return images
    cubeimage = setupcube(data.nval)
    for m in range(data.nval):
        name = 'scalar' if data.nval == 1 else 'scalar{:02d}'.format(m)
        cubeimage.GetPointData().AddArray(vtkarray(fields[m][vtk_order], name))
    return cubeimage


def _quivfromcube(cubdata: VecCubeData, logscale: bool = False) -> vtk.vtkImageData:
    """
    Return a vtkImageData object with the vector field
    from a VecCubeData object.
    """
    grid = fillcubeimage(cubdata, logscale=logscale)
    grid.GetPointData().SetActiveVectors('vector')
    grid.GetPointData().SetActiveScalars('scalar')
    return grid
