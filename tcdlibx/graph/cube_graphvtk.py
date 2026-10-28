"""
Facade of the VTK drawing helpers.

The implementation lives in the vtk_* modules of this package; everything is
re-exported here so that ``tcdlibx.graph.cube_graphvtk`` keeps working.

- vtk_geometry:  point sets and vector-field polydata
- vtk_clipping:  spatial clipping of cube images and preview planes
- vtk_cube:      CubeData -> vtkImageData
- vtk_molecule:  molecule structure, normal modes, dipole vectors
- vtk_fields:    quiver, streamlines, isosurfaces, volume rendering, color bar
- vtk_tensors:   APT/AAT tensor glyphs and surfaces
- vtk_particles: animated particles along streamlines
"""
from tcdlibx.graph.helpers import fibonacci_spiral_samples_on_unit_sphere
from tcdlibx.graph.vtk_clipping import (apply_spatial_clipping,
                                        create_clip_plane_actors,
                                        move_clip_plane)
from tcdlibx.graph.vtk_cube import _quivfromcube, fillcubeimage
from tcdlibx.graph.vtk_fields import (_countur, _volumerendering, countur,
                                      draw_colorbar, draw_cones_nogrid,
                                      draw_ellipsoid, fillstreamline, quiv3d,
                                      volumerendering)
from tcdlibx.graph.vtk_geometry import (create_vector_field_polydata,
                                        dots2vtkarray)
from tcdlibx.graph.vtk_molecule import (draw_nm3d, draw_vectors, fillmolecule,
                                        fillmolecule_custom)
from tcdlibx.graph.vtk_particles import (ParticleData,
                                         StreamlineParticleAnimator,
                                         create_streamline_particles)
from tcdlibx.graph.vtk_tensors import (_get_filtered_tensors,
                                       draw_molecular_tensor_surface,
                                       draw_molecular_tensors,
                                       draw_tensor_spheres,
                                       draw_tensor_surface_from_norms)

__all__ = [
    'ParticleData', 'StreamlineParticleAnimator', 'apply_spatial_clipping',
    'countur', 'create_clip_plane_actors', 'create_streamline_particles',
    'create_vector_field_polydata', 'dots2vtkarray', 'draw_colorbar',
    'draw_cones_nogrid', 'draw_ellipsoid', 'draw_molecular_tensor_surface',
    'draw_molecular_tensors', 'draw_nm3d', 'draw_tensor_spheres',
    'draw_tensor_surface_from_norms', 'draw_vectors',
    'fibonacci_spiral_samples_on_unit_sphere', 'fillcubeimage',
    'fillmolecule', 'fillmolecule_custom', 'fillstreamline', 'move_clip_plane',
    'quiv3d', 'volumerendering',
    # private helpers, kept importable (progs/tcdvis.py uses _countur)
    '_countur', '_get_filtered_tensors', '_quivfromcube', '_volumerendering',
]
