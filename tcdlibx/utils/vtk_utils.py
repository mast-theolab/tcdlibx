"""
VTK import utilities with error handling for missing optional dependencies
"""
import sys
import warnings

def import_vtk():
    """
    Import VTK with error handling for missing optional modules.
    
    Returns:
        vtk module or a proxy with core functionality
    """
    try:
        # Try normal VTK import
        import vtk
        return vtk
    except ImportError as e:
        error_msg = str(e)
        
        # Handle specific known issues
        if any(x in error_msg for x in ["vtkIOFides", "libviskores", "libanari"]):
            warnings.warn(f"VTK import issue: {error_msg}. Using fallback import.", ImportWarning)
            
            # Try importing without the problematic module
            import importlib.util
            
            # Create dummy modules for problematic imports
            for problematic_module in ["vtkmodules.vtkIOFides", "vtkmodules.vtkRenderingAnari"]:
                if problematic_module not in sys.modules:
                    spec = importlib.util.spec_from_loader(problematic_module, loader=None)
                    dummy_module = importlib.util.module_from_spec(spec)
                    sys.modules[problematic_module] = dummy_module
            
            # Try importing VTK again
            try:
                import vtk
                return vtk
            except ImportError:
                # Last resort: import core modules individually
                try:
                    import vtkmodules.vtkCommonCore as core
                    import vtkmodules.vtkCommonDataModel as data
                    import vtkmodules.vtkFiltersCore as filters
                    import vtkmodules.vtkFiltersSources as sources
                    import vtkmodules.vtkRenderingCore as rendering
                    
                    # Create a minimal VTK proxy
                    class VTKMinimal:
                        def __init__(self):
                            # Import the most commonly used VTK classes
                            self.vtkPoints = core.vtkPoints
                            self.vtkPolyData = data.vtkPolyData
                            self.vtkImageData = data.vtkImageData
                            self.vtkSphereSource = sources.vtkSphereSource
                            self.vtkPolyDataMapper = rendering.vtkPolyDataMapper
                            self.vtkActor = rendering.vtkActor
                            self.vtkRenderer = rendering.vtkRenderer
                            self.vtkRenderWindow = rendering.vtkRenderWindow
                            self.vtkRenderWindowInteractor = rendering.vtkRenderWindowInteractor
                            
                            # Add more as needed
                            self._modules = [core, data, filters, sources, rendering]
                        
                        def __getattr__(self, name):
                            # Try to find the attribute in imported modules
                            for module in self._modules:
                                if hasattr(module, name):
                                    return getattr(module, name)
                            
                            # Try additional VTK modules for common classes
                            if name == 'vtkTransform':
                                try:
                                    import vtkmodules.vtkCommonTransforms as transforms
                                    if hasattr(transforms, name):
                                        return getattr(transforms, name)
                                except ImportError:
                                    pass
                            
                            raise AttributeError(f"VTK has no attribute '{name}'")
                    
                    return VTKMinimal()
                
                except ImportError as final_error:
                    raise ImportError(f"Could not import VTK or fallback modules: {final_error}") from e
        else:
            # Re-raise if it's not a known issue
            raise

# Global VTK instance
vtk = import_vtk()

# Import Qt VTK widget
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError as e:
    # Try alternative import path
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError as e2:
        warnings.warn(f"Could not import QVTKRenderWindowInteractor: {e}, {e2}")
        
        # Create a mock class for environments where Qt VTK is not available
        class MockQVTKRenderWindowInteractor:
            def __init__(self, *args, **kwargs):
                raise ImportError("QVTKRenderWindowInteractor is not available. VTK Qt integration may not be installed.")
        
        QVTKRenderWindowInteractor = MockQVTKRenderWindowInteractor
