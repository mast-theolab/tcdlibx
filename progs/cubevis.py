"""Script to visualize vector fields from cube files, specifically designed for Transition Current Density (TCD) data.

    author: m. fuse
    """
# Standard library imports
import copy
import os
import sys
import typing as tp

# Third-party imports
import numpy as np

# PySide6/Qt imports
from PySide6.QtCore import QLocale, QRect, QTimer
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QIntValidator
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QStyle, QVBoxLayout
)

# Local imports - tcdlibx package
from tcdlibx.calc.cube_manip import VecCubeData, ScalarsCube, cube_parser
from tcdlibx.graph.helpers import VolumeMolecule, filtervecatom, sample_molecular_volume, DEFAULT_PARAMETERS, write_molecule_pov, vdw_boolean_mask, fill_molecule_from_cubedata
import tcdlibx.graph.cube_graphvtk as cubetk
from tcdlibx.gui.dialogs import (
    SavePngDialog, SavePngSeriesDialog, StreamLineSetupDialog, CubeDialog, QuiverSetupDialog, SaveSceneDialog, ExportPOVDialog, NMConfigDialog, MoleculeConfigDialog
)
from tcdlibx.io.estp_io import PHYSFACT
from tcdlibx.io.jsonio import read_json, write_json
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.var_tools import range_parse
from tcdlibx.utils.vtk_utils import QVTKRenderWindowInteractor, vtk

# Module configuration
DEBUG = True

class TCDvis(QMainWindow):
    """Cube file Visualization Application.
    
    A PySide6/VTK-based application for visualizing Transition Current Density (TCD) and other cube files.
    
    This class provides a complete GUI interface for:
    - Loading and visualizing molecular structure data from cube files.
    - Displaying electronic and vibrational TCD data as streamlines, quiver plots, etc.
    - Animated particle visualization along streamlines
    - Interactive isosurface and vector field visualization
    - Export capabilities for images and data
    
    Args:
        moldata: Optional molecular data (EleMolecule or VibMolecule instance).
                If provided, the application will initialize with this data loaded.
    """
    
    def __init__(self, moldata: VolumeMolecule | None = None):
        """Initialize the TCD visualization application.
        
        Args:
            moldata: Optional molecular data to load on startup.
        """
        super().__init__()

        self._lastfname = 'tcdfigure.png'
        # self._cube = None
        self._fchk = moldata # to avoid refactoring, we keep the name _fchk for the molecule data
        if self._fchk is not None:
            self._ndata = self._fchk.nvol
        else:
            self._ndata = 0
        self._activest = 1
        self._actors = {}
        self._menus = {}
        self._has_auto_fragment = False  # Track if automatic fragment for missing atoms was created
        
        # Animation variables for streamline particles
        self._animation_timer = None
        self._particle_animator = None
        self._animation_step = 0
        self._animation_speed = 0.02
        self._show_particles = False
        self._seeds = None
        
        self._default = copy.deepcopy(DEFAULT_PARAMETERS) 

        self.initUI()

    def initUI(self):
        self.setWindowTitle("TCD Visualization")
        self.setGeometry(100, 100, 800, 600)

        self.createMenus()
        self.vl = QVBoxLayout()
        self.createBar()
        self.createVTKrender()

    def createMenus(self):
        # Main menu
        self._menus['main'] = []
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu('Menu')
        ## Open cube file
        openButton = QAction(QIcon(''), 'Open', self)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open a cube file')
        openButton.triggered.connect(self.open_cube)
        fileMenu.addAction(openButton)
        ## save png
        saveButton = QAction(QIcon(''), 'SavePNG', self)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save a png')
        saveButton.triggered.connect(self.save_png)
        fileMenu.addAction(saveButton)
        ## Save png to make gif
        gifButton = QAction(QIcon(''), 'SaveGifPNGs', self)
        gifButton.setStatusTip('Save pngs to make a gif')
        gifButton.triggered.connect(self.save_png_rotation)
        fileMenu.addAction(gifButton)
        # Open a Scene
        osceneButton = QAction(QIcon(''), 'OpenScene', self)
        #openButton.setShortcut('Ctrl+O')
        osceneButton.setStatusTip('Open a VTKScene from a JSON file')
        osceneButton.triggered.connect(self.open_scene)
        fileMenu.addAction(osceneButton)
        # Save a Scene
        ssceneButton = QAction(QIcon(''), 'SaveScene', self)
        # saveButton.setShortcut('Ctrl+S')
        ssceneButton.setStatusTip('Save a VKScene in a JSON file')
        ssceneButton.triggered.connect(self.save_scene)
        fileMenu.addAction(ssceneButton)
        # Export scene as POV-Ray file
        exportPOVButton = QAction(QIcon(''), 'ExportScenePOV', self)
        exportPOVButton.setStatusTip('Export the current scene to a POV-Ray file')
        exportPOVButton.triggered.connect(self.export_scene_pov)
        fileMenu.addAction(exportPOVButton)
        # For Now disable export POV-Ray until we have a better implementation
        exportPOVButton.setEnabled(False)

        ## Exit
        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self._cjclose)
        fileMenu.addAction(exitButton)
        # Mol menu
        self._menus['mol'] = {}
        fileMol = mainMenu.addMenu('Mol.')
        grpsButton = QAction(QIcon(''), 'Open Groups', self)
        grpsButton.setStatusTip('Open a Json file with groups')
        grpsButton.triggered.connect(self.opengroups)
        grpsButton.setEnabled(False)
        self._menus['mol']['grps'] = grpsButton
        fileMol.addAction(grpsButton)
        
        # Molecule transparency toggle
        transparentButton = QAction(QIcon(''), 'Transparent Molecule', self)
        transparentButton.setStatusTip('Make molecule transparent (alpha 0.3)')
        transparentButton.setCheckable(True)
        transparentButton.triggered.connect(self.toggle_mol_transparency)
        transparentButton.setEnabled(False)
        self._menus['mol']['transparent'] = transparentButton
        fileMol.addAction(transparentButton)
        
        # Molecule settings
        molSettingsButton = QAction(QIcon(''), 'Settings', self)
        molSettingsButton.setStatusTip('Configure molecule display settings (wireframe, opacity, radii)')
        molSettingsButton.triggered.connect(self.configure_molecule)
        molSettingsButton.setEnabled(False)
        self._menus['mol']['settings'] = molSettingsButton
        fileMol.addAction(molSettingsButton)
        
        # Cube menus
        self._menus['cubes'] = {}
        fileetcd = mainMenu.addMenu('Cubes')
        aimButton = QAction(QIcon(''), 'Add AIM', self)
        aimButton.setStatusTip('Loads a cube file with AIM partition of the cube')
        aimButton.triggered.connect(self.open_aim)
        aimButton.setEnabled(False)
        self._menus['cubes']['aim'] = aimButton
        fileetcd.addAction(aimButton)
        saimButton = QAction(QIcon(''), 'Show AIM Frag Space', self)
        saimButton.setStatusTip('Shows the volume of each fragments in AIM partition scheme')
        saimButton.triggered.connect(self.show_aim_space)
        saimButton.setEnabled(False)
        saimButton.setCheckable(True)
        self._menus['cubes']['saim'] = saimButton
        fileetcd.addAction(saimButton)
        # ETCD dtm
        fileTCDmt = fileetcd.addMenu('Cube DTM')
        fileTCDmt.setEnabled(False)
        self._menus['cubes']['dmt'] = fileTCDmt
        self._menus['etcddtm'] = {}
        tcdmtButton = QAction(QIcon(''), 'Total', self)
        tcdmtButton.setStatusTip('Display the integrated field and its rotor')
        tcdmtButton.setCheckable(True)
        tcdmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(tcdmtButton)
        self._menus['etcddtm']['tot'] = tcdmtButton
        frtcmtButton = QAction(QIcon(''), 'Fragments Contr.', self)
        frtcmtButton.setStatusTip('Contributions of each fragment the field')
        frtcmtButton.setCheckable(True)
        frtcmtButton.setEnabled(False)
        frtcmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(frtcmtButton)
        self._menus['etcddtm']['frags'] = frtcmtButton
        nomtButton = QAction(QIcon(''), 'Clear', self)
        nomtButton.setStatusTip('remove all vectors')
        nomtButton.triggered.connect(self.cleartcdmt)
        fileTCDmt.addAction(nomtButton)

    def createBar(self):                
        # buttons
        grid = QGridLayout()
        self.vl.addLayout(grid)
        hlay = QHBoxLayout()
        grid.addLayout(hlay, 0, 0)

        # groups
        hlay.addItem(QSpacerItem(100, 10, QSizePolicy.Expanding))
        # TCD types
        self.prop = QComboBox()
        self.prop.setGeometry(QRect(40, 40, 491, 31))
        self.prop.addItem('                   ')
        self.prop.currentIndexChanged.connect(self._enablefieldsetup)
        grid.addWidget(self.prop, 0, 1)
        self.etcdch = QPushButton('Show', self)
        self.etcdch.clicked.connect(self.showtcd)
        self.etcdch.setEnabled(False)
        message = QLabel("IsoVal:")
        self.isoline = QLineEdit()
        isoval = QDoubleValidator()
        isoval.setRange(0.0000001, 2.)  # TODO: Make range configurable
        isoval.setLocale(QLocale('English'))
        self.isoline.setValidator(isoval)
        self.isoline.setText(f"{self._default['isoval']['iso']:.6f}")
        self.isoline.editingFinished.connect(self._setisoval)
        self.isoline.setEnabled(False)
        hlay3 = QHBoxLayout()
        hlay3.addWidget(self.etcdch)
        hlay3.addItem(QSpacerItem(100, 10, QSizePolicy.Expanding))
        hlay.addWidget(message)
        hlay.addWidget(self.isoline)
        pixmapi = getattr(QStyle, "SP_MessageBoxInformation")
        ticon = self.style().standardIcon(pixmapi)
        self._fieldsetup = QPushButton(icon=ticon,
                                       text="Setup", parent=self)
        self._fieldsetup.clicked.connect(self._fieldstp)
        self._fieldsetup.setEnabled(False)
        hlay.addWidget(self._fieldsetup)
        grid.addLayout(hlay3, 0, 2)

    # VTK functions
    def createVTKrender(self):
        self.frame = QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1.0, 1.0, 1.0)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        if self._fchk is not None:
            # Determine excluded atoms if hide auto group is enabled
            excluded_atoms = None
            if (self._default['molconfig']['hide_auto_group'] and 
                self._has_auto_fragment and 
                self._fchk._frags is not None and 
                len(self._fchk._frags) > 0):
                excluded_atoms = self._fchk._frags[-1]  # Last fragment is auto-generated
                
            self._actors['mol'] =  cubetk.fillmolecule_custom(
                self._fchk.atnum,
                self._fchk.crd,
                opacity=self._default['molconfig']['opacity'],
                bond_radius=self._default['molconfig']['bond_radius'],
                atom_radius_scale=self._default['molconfig']['atom_radius_scale'],
                tubes_mode=self._default['molconfig']['tubes_mode'],
                bond_tollerance=self._default['molconfig']['bond_tollerance'],
                excluded_atoms=excluded_atoms
            )
            self.ren.AddActor(self._actors['mol'].actor)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()
        self.iren.Start()

    def __del__(self):
        """Clean up animation timer when object is destroyed"""
        if hasattr(self, '_animation_timer') and self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer.deleteLater()
        if hasattr(self, '_particle_animator') and self._particle_animator is not None:
            self._particle_animator.stop_animation()

    def _cleanactors(self):
        """Removes all actors from the renderer"""
        # Stop any running animation first
        if self._show_particles:
            self.stop_animation()
        # Reset animation flag
        self._default["vfield"]["animate_particles"] = False
        
        self.ren.RemoveAllViewProps()

    def _updatereder(self):
        self.vtkWidget.GetRenderWindow().Render()

    def animate_particles(self):
        """Update particle positions along streamlines using VTK animator"""
        if self._particle_animator:
            self._particle_animator.update_particles()
            self._updatereder()
    
    def start_animation(self):
        """Start the particle animation"""
        if self._animation_timer is None:
            self._animation_timer = QTimer()
            self._animation_timer.timeout.connect(self.animate_particles)
        
        self._animation_timer.start(100)  # 100ms = 10 FPS for smoother animation
        self._show_particles = True
        
        if self._particle_animator:
            self._particle_animator.start_animation()
    
    def stop_animation(self):
        """Stop the particle animation"""
        if self._animation_timer:
            self._animation_timer.stop()
        self._show_particles = False
        
        if self._particle_animator:
            self._particle_animator.stop_animation()
            self._particle_animator = None
        
        self._updatereder()

    def save_png(self):
        # Get current render window size for aspect ratio calculation
        render_window = self.vtkWidget.GetRenderWindow()
        current_size = render_window.GetSize()
        figure_size = (current_size[0], current_size[1])
        
        pngdialog = SavePngDialog(fname=self._lastfname, figure_size=figure_size)
        pngdialog.exec()
        self._lastfname = pngdialog._fname
        if pngdialog._okexit:
            # Get the desired output size and DPI from dialog
            output_width, output_height = pngdialog.get_size()
            target_dpi = pngdialog._dpi
            
            # Get current render window size
            render_window = self.vtkWidget.GetRenderWindow()
            current_size = render_window.GetSize()
            
            # Calculate magnification needed to achieve target DPI
            # VTK's default is 72 DPI, so magnification = target_dpi / 72
            magnification = target_dpi / 72.0
            
            # Use vtkRenderLargeImage to properly handle DPI
            w2if = vtk.vtkRenderLargeImage()
            w2if.SetInput(self.ren)
            w2if.SetMagnification(int(magnification))
            

            # base_width = int(output_width * magnification)
            base_width = int(output_width)
            # base_height = int(output_height * magnification)
            base_height = int(output_height)
            
            # Temporarily set render window to base size
            original_size = render_window.GetSize()
            render_window.SetSize(base_width, base_height)
            render_window.Render()
            
            # Now render with magnification
            w2if.Update()

            # Create PNG writer and set DPI information
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(self._lastfname)
            writer.SetInputConnection(w2if.GetOutputPort())
            
            # Write the image
            writer.Write()
                        
            # Restore original render window size
            render_window.SetSize(original_size[0], original_size[1])
            render_window.Render()
            # BUG metadata of figure are wrong

    def save_png_rotation(self):
        #
        pngdialog = SavePngSeriesDialog(fname="test-XXX.png")
        pngdialog.exec()
        fpath = pngdialog._fname
        if pngdialog._okexit:
            # prova
            camera = self.ren.GetActiveCamera()
            focal_point = camera.GetFocalPoint()
            view_up = camera.GetViewUp()
            position = camera.GetPosition()
            center = self._actors['mol'].GetCenter()

            axis = [0,0,0]
            axis[0] = -1*camera.GetViewTransformMatrix().GetElement(0,0)
            axis[1] = -1*camera.GetViewTransformMatrix().GetElement(0,1)
            axis[2] = -1*camera.GetViewTransformMatrix().GetElement(0,2)

            print(position,focal_point,view_up,)

            print(camera.GetViewTransformMatrix())
            print(camera.GetViewTransformMatrix().GetElement(0,0))
            print(camera.GetViewTransformMatrix().GetElement(0,1))
            print(camera.GetViewTransformMatrix().GetElement(0,2))

            for n,q in enumerate([10]*35):

                transform = vtk.vtkTransform()
                transform.Identity()

                transform.Translate(*center)
                transform.RotateWXYZ(q,view_up)
                transform.RotateWXYZ(0,axis)
                transform.Translate(*[-1*x for x in center])

                new_position = [0,0,0]
                new_focal_point = [0,0,0]
                transform.TransformPoint(position,new_position)
                transform.TransformPoint(focal_point,new_focal_point)

                camera.SetPosition(new_position)
                camera.SetFocalPoint(new_focal_point)

                focal_point = camera.GetFocalPoint()
                view_up = camera.GetViewUp()
                position = camera.GetPosition()

                camera.OrthogonalizeViewUp()
                self.ren.ResetCameraClippingRange()

                renderWindow = self.vtkWidget.GetRenderWindow()
                renderWindow.Render()
                windowToImageFilter = vtk.vtkWindowToImageFilter()
                windowToImageFilter.SetInput(renderWindow)
                windowToImageFilter.Update()

                fpath = fpath.replace("XXX", "{:03d}")
                ifpath = fpath.format(n)
                w2if = vtk.vtkWindowToImageFilter()
                # w2if = vtk.vtkRenderLargeImage()
                # w2if.SetInput(self.ren)
                # w2if.SetMagnification(6)
                w2if.SetInput(renderWindow)
                w2if.Update()

                writer = vtk.vtkPNGWriter()
                writer.SetFileName(ifpath)
                writer.SetInputConnection(w2if.GetOutputPort())
                writer.Write()

    # menu functions
    def _cjclose(self):
        self.close()

    def _reset_interface_state(self):
        """Reset interface state when switching between molecule types"""
        # Reset normal mode checkboxes (only relevant for vibrational molecules)
        if 'mol' in self._menus and 'nm' in self._menus['mol']:
            if 'disp' in self._menus['mol']['nm']:
                self._menus['mol']['nm']['disp'].setChecked(False)
        
        # Reset normal mode configuration to defaults
        self._default['nmconfig'] = {
            'invert_phase': False,
            'scale_factor': 1.0,
            'color': (0.0, 0.0, 1.0)
        }
        
        # Reset transparency checkbox
        if 'mol' in self._menus and 'transparent' in self._menus['mol']:
            self._menus['mol']['transparent'].setChecked(False)
        
        # Reset TCD-related interface elements
        self._reset_tcd_interface()
        
        # Reset any other interface elements that might cause conflicts
        # between vibrational and electronic molecule types

    def _reset_tcd_interface(self):
        """Reset TCD-related interface elements to their default state"""
        # Reset TCD checkbox/control
        # self.etcdch.setEnabled(False)
        
        # Reset isoline control
        self.isoline.setText(f"{self._default['isoval']['iso']:.6f}")
        self.isoline.setEnabled(False)
        
        # Reset property combo box to first item
        if hasattr(self, 'prop'):
            self.prop.clear()
            self.prop.addItem('')
            self.prop.setCurrentIndex(0)
        
        # Disable TCD-related menu items
        if hasattr(self, '_menus'):
            if 'etcd' in self._menus and 'dmt' in self._menus['etcd']:
                self._menus['etcd']['dmt'].setEnabled(False)
            if 'vtcd' in self._menus and 'dmt' in self._menus['vtcd']:
                self._menus['vtcd']['dmt'].setEnabled(False)
            if 'etcddtm' in self._menus and 'frags' in self._menus['etcddtm']:
                self._menus['etcddtm']['frags'].setEnabled(False)
            if 'vtcddtm' in self._menus and 'frags' in self._menus['vtcddtm']:
                self._menus['vtcddtm']['frags'].setEnabled(False)
        
        # Reset DTM-related elements
        self._reset_dtm_interface()

    def _reset_dtm_interface(self):
        """Reset DTM-related interface elements to their default state"""
        # Clear any existing DTM visualizations
        if hasattr(self, '_actors') and 'mfpdtm' in self._actors:
            self.ren.RemoveActor(self._actors['mfpdtm'].actor)
            self._actors.pop('mfpdtm')
        
        # Disable and uncheck DTM menu items
        if hasattr(self, '_menus'):
            # Reset electronic DTM menus
            if 'mol' in self._menus and 'edmt' in self._menus['mol']:
                self._menus['mol']['edmt'].setEnabled(False)
            if 'eledmt' in self._menus:
                for key in self._menus['eledmt']:
                    if key != 'menu' and hasattr(self._menus['eledmt'][key], 'setChecked'):
                        self._menus['eledmt'][key].setChecked(False)
            
            # Reset vibrational DTM menus  
            if 'mol' in self._menus and 'vibdmt' in self._menus['mol']:
                self._menus['mol']['vibdmt'].setEnabled(False)
            if 'vibdmt' in self._menus:
                for key in self._menus['vibdmt']:
                    if key != 'menu' and hasattr(self._menus['vibdmt'][key], 'setChecked'):
                        self._menus['vibdmt'][key].setChecked(False)



    def _fieldstp(self):
        current_prop = self.prop.currentText().lower()

        # Compute field bounding box for clip-plane preview in setup dialogs
        try:
            _tcd_bnd = self._fchk.get_vol("1")
            _orig = _tcd_bnd.loc2wrd[:3, 3]
            _spac = np.diag(_tcd_bnd.loc2wrd[:3, :3])
            _ends = _orig + (np.array(_tcd_bnd.npts) - 1) * _spac
            _vtk_bounds = (
                float(min(_orig[0], _ends[0])), float(max(_orig[0], _ends[0])),
                float(min(_orig[1], _ends[1])), float(max(_orig[1], _ends[1])),
                float(min(_orig[2], _ends[2])), float(max(_orig[2], _ends[2])),
            )
        except Exception:
            _vtk_bounds = None
        
        if current_prop == "streamlines":
            # Handle streamlines setup dialog
            if self._default["vfield"]["mspeed"] is None:
                self._default["vfield"]["mspeed"] = self._fchk.get_vol("1")._maxnorm/1e4
            fieldprm = StreamLineSetupDialog(vfmax=self._default["vfield"]["vfmax"],
                                             vfmin=self._default["vfield"]["vfmin"],
                                             mspeed=self._default["vfield"]["mspeed"],
                                             maxval=self._fchk.get_vol("1")._maxnorm,
                                             nseeds=self._default["vfield"]["npoints"],
                                             scale=self._default["vfield"]["scalellipse"],
                                             direction=self._default["vfield"]["showdir"],
                                             showellipse=self._default["vfield"]["showseeds"],
                                             showbar=self._default["vfield"]["showbar"],
                                             animate_particles=self._default["vfield"]["animate_particles"],
                                             num_particles=self._default["vfield"]["num_particles"],
                                             particle_type=self._default["vfield"]["particle_type"],
                                             sampling_method=self._default["vfield"]["sampling_method"],
                                             scalevdw=self._default["vfield"]["scalevdw"],
                                             enable_clipping=self._default["vfield"]["enable_clipping"],
                                             clip_bounds=self._default["vfield"]["clip_bounds"],
                                             vtk_renderer=self.ren,
                                             vtk_render_window=self.vtkWidget.GetRenderWindow(),
                                             vtk_scene_bounds=_vtk_bounds)
            fieldprm.exec()
            # Update the default values
            self._default["vfield"]["vfmax"] = fieldprm._vfmax
            self._default["vfield"]["vfmin"] = fieldprm._vfmin
            self._default["vfield"]["mspeed"] = fieldprm._mspeed
            
            if fieldprm._recalseeds:
                # Remove old seeds visualization if it exists
                if 'seeds' in self._actors:
                    self.ren.RemoveActor(self._actors['seeds'].actor)
                    del self._actors['seeds']
                    
                if fieldprm._sampling_method == "ellipsoid":
                    self._seeds = self._fchk.sample_ellipse_space(npts=fieldprm._nseeds, scale=fieldprm._scale)
                else:  # molecular volume
                    current_tcd = self._fchk.get_vol("1")
                    self._seeds = sample_molecular_volume(current_tcd, fieldprm._nseeds, scale=fieldprm._scalevdw)
                    
                # If show seeds is enabled, create new seeds visualization
                if fieldprm._showellipse:
                    self._actors['seeds'] = cubetk.draw_ellipsoid(self._seeds)
                    self.ren.AddActor(self._actors['seeds'].actor)
            self._default["vfield"]["scalellipse"] = fieldprm._scale
            self._default["vfield"]["scalevdw"] = fieldprm._scalevdw
            self._default["vfield"]["sampling_method"] = fieldprm._sampling_method
            self._default["vfield"]["npoints"] = fieldprm._nseeds
            self._default["vfield"]["showdir"] = fieldprm._direction
            self._default["vfield"]["showseeds"] = fieldprm._showellipse
            self._default["vfield"]["showbar"] = fieldprm._showbar
            self._default["vfield"]["animate_particles"] = fieldprm._animate_particles
            self._default["vfield"]["num_particles"] = fieldprm._num_particles
            self._default["vfield"]["particle_type"] = fieldprm._particle_type
            self._default["vfield"]["enable_clipping"] = fieldprm._enable_clipping
            self._default["vfield"]["clip_bounds"] = fieldprm._clip_bounds
            
            # Handle streamline-specific actors
            if 'tcd' in self._actors:
                if fieldprm._redrawstream:
                    self.showtcd()
                else:
                    # Handle animation changes
                    if self._default["vfield"]["animate_particles"] and not self._show_particles:
                        # Start animation
                        if hasattr(self._actors['tcd'], 'streamline_data'):
                            streamline_polydata = self._actors['tcd'].streamline_data
                        else:
                            streamline_polydata = self._actors['tcd'].actor.GetMapper().GetInput()
                        
                        self._particle_animator = cubetk.create_streamline_particles(
                            self.ren, streamline_polydata, 
                            num_particles=self._default["vfield"]["num_particles"],
                            particle_type=self._default["vfield"]["particle_type"])
                        self.start_animation()
                    elif not self._default["vfield"]["animate_particles"] and self._show_particles:
                        # Stop animation
                        self.stop_animation()
                    elif self._default["vfield"]["animate_particles"] and self._show_particles:
                        # Animation is already running, check if particle type or count changed
                        if (hasattr(self, '_particle_animator') and 
                            (self._particle_animator.particle_type != self._default["vfield"]["particle_type"] or
                             self._particle_animator.get_particle_count() != self._default["vfield"]["num_particles"])):
                            # Update particles with new type/count
                            if hasattr(self._actors['tcd'], 'streamline_data'):
                                streamline_polydata = self._actors['tcd'].streamline_data
                            else:
                                streamline_polydata = self._actors['tcd'].actor.GetMapper().GetInput()
                            
                            self._particle_animator.update_particle_type(
                                self._default["vfield"]["particle_type"], 
                                streamline_polydata, 
                                self._default["vfield"]["num_particles"])
                            
                            # Force render update to show new particles immediately
                            self.vtkWidget.GetRenderWindow().Render()
                    
                    if fieldprm._showbar and 'tcdbar' not in self._actors:
                        self._actors['tcdbar'] = cubetk.draw_colorbar(self._actors['tcd'].actor, "Norm(J)")
                        self.ren.AddActor2D(self._actors['tcdbar'].actor)
                    elif 'tcdbar' in self._actors:
                        self.ren.RemoveActor(self._actors['tcdbar'].actor)
                        del self._actors['tcdbar']
                    if fieldprm._direction and 'tcddir' not in self._actors:
                        tmp_cube = copy.deepcopy(self._fchk.get_vol("1"))
                        # self._actors['tcddir'] = cubetk.draw_cones_nogrid(tmp_cube, self._seeds)
                        self._actors['tcddir'] = cubetk.quiv3d(tmp_cube, 
                                               scale=self._default["quiver"]["scale"]/5,
                                               subsample_factor=100,
                                               glyphmode='cone',
                                               clip_bounds=self._default["vfield"]["clip_bounds"] if self._default["vfield"]["enable_clipping"] else None)
                        self.ren.AddActor(self._actors['tcddir'].actor)
                    elif 'tcddir' in self._actors:
                        self.ren.RemoveActor(self._actors['tcddir'].actor)
                        del self._actors['tcddir']

            if fieldprm._showellipse:
                # Remove old seeds if they exist
                if 'seeds' in self._actors:
                    self.ren.RemoveActor(self._actors['seeds'].actor)
                    del self._actors['seeds']
                    
                # Create seeds if they don't exist or weren't created during recalc
                if self._seeds is None:
                    if self._default["vfield"]["sampling_method"] == "ellipsoid":
                        self._seeds = self._fchk.sample_ellipse_space(self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalellipse"])
                    else:  # molecular volume
                        current_tcd = self._fchk.get_vol("1")
                        self._seeds = sample_molecular_volume(current_tcd, self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalevdw"])
                
                # Always show the seeds regardless of sampling method
                self._actors['seeds'] = cubetk.draw_ellipsoid(self._seeds)
                self.ren.AddActor(self._actors['seeds'].actor)
            elif 'seeds' in self._actors:
                self.ren.RemoveActor(self._actors['seeds'].actor)
                del self._actors['seeds']
                
        elif current_prop == "quiver":
            # Handle quiver setup dialog
            quiverprm = QuiverSetupDialog(scale=self._default["quiver"]["scale"],
                                         subsamp=self._default["quiver"]["subsample"],
                                         enable_clipping=self._default["quiver"]["enable_clipping"],
                                         clip_bounds=self._default["quiver"]["clip_bounds"],
                                         vtk_renderer=self.ren,
                                         vtk_render_window=self.vtkWidget.GetRenderWindow(),
                                         vtk_scene_bounds=_vtk_bounds,
                                         lower=self._default["quiver"]["lower"],
                                         upper=self._default["quiver"]["upper"])
            quiverprm.exec()
            # Update the default values
            self._default["quiver"]["scale"] = quiverprm._scale
            self._default["quiver"]["subsample"] = quiverprm._subsamp
            self._default["quiver"]["enable_clipping"] = quiverprm._enable_clipping
            self._default["quiver"]["clip_bounds"] = quiverprm._clip_bounds
            self._default["quiver"]["lower"] = quiverprm._lower
            self._default["quiver"]["upper"] = quiverprm._upper
            
            # Redraw quiver if it exists
            if 'tcd' in self._actors:
                self.showtcd()
        
        self._updatereder()

    def _enablefieldsetup(self):
        # Stop any running animation when changing property type
        if self._show_particles:
            self.stop_animation()
        
        current_prop = self.prop.currentText().lower()
        if current_prop in ["streamlines", "quiver"]:
            self._fieldsetup.setEnabled(True)
        else:
            self._fieldsetup.setEnabled(False)
            # Reset animation flag for non-streamline properties
            if current_prop != "streamlines":
                self._default["vfield"]["animate_particles"] = False
            
        if 'tcd' in self._actors:
            self.ren.RemoveActor(self._actors['tcd'].actor)
            self.showtcd()

    # File opening
    def open_cube(self):
        fname = QFileDialog.getOpenFileName(self, 'Select Cube file', '.','*.cube')[0]
        if len(fname) == 0:
            return None
        # to avoid refactoring for now, keep fchkfname
        self._fchkfname = os.path.basename(fname)
        try:
            cubdata = cube_parser(fname)
            self._fchk = fill_molecule_from_cubedata(cubdata)
            # self._fchk.add_vol('1', cubdata)
            self._menus['mol']['grps'].setEnabled(True)
            self._menus['mol']['transparent'].setEnabled(True)
            self._menus['mol']['settings'].setEnabled(True)
            self._menus['cubes']['aim'].setEnabled(True)
            self.etcdch.setEnabled(True)
            
            if cubdata.nval == 1:
                self._cubtype = 'scalar'
                self._fchk.add_vol('1', ScalarsCube(cubdata))
                self._isoline.setEnabled(True)
                self._fieldsetup.setEnabled(True)
            elif cubdata.nval == 3:
                self._cubtype = 'vector'
                self._fchk.add_vol('1', VecCubeData(cubdata))
                self._fieldsetup.setEnabled(True)
                self._menus['cubes']['dmt'].setEnabled(True)
            else:
                raise NoValidData("open_cube", "Cube file must contain either a single scalar or a 3D vector dataset")
        except Exception as err:
            print(err)

        self._cleanactors()
        self._actors = {}
        self._has_auto_fragment = False  # Reset auto-fragment flag when opening new file
        
        
        # Reset interface state to prevent issues when switching molecule types
        self._reset_interface_state()
        
        # for key in self._actors:
        #     self._actors[key] = None
        # update validator
        # Determine excluded atoms if hide auto group is enabled
        excluded_atoms = None
        if (self._default['molconfig']['hide_auto_group'] and 
            self._has_auto_fragment and 
            self._fchk._frags is not None and 
            len(self._fchk._frags) > 0):
            excluded_atoms = self._fchk._frags['indx'][-1]  # Last fragment is auto-generated
        self._actors['mol'] =  cubetk.fillmolecule_custom(
            self._fchk.atnum,
            self._fchk.crd,
            opacity=self._default['molconfig']['opacity'],
            bond_radius=self._default['molconfig']['bond_radius'],
            atom_radius_scale=self._default['molconfig']['atom_radius_scale'],
            tubes_mode=self._default['molconfig']['tubes_mode'],
            bond_tollerance=self._default['molconfig']['bond_tollerance'],
            excluded_atoms=excluded_atoms
        )
        self.ren.AddActor(self._actors['mol'].actor)
        self.ren.ResetCamera()
        # Create the ComboBox items based on cube types: scalar or vector
        self.prop.clear()
        if self._cubtype == 'scalar':
            self.prop.addItem('')  # Default empty item
            self.prop.addItem('Isosurface')
            # self.prop.addItem('Volume Rendering') # NYI
        elif self._cubtype == 'vector':
            self.prop.addItem('')
            self.prop.addItem('Streamlines')
            self.prop.addItem('Quiver')
            self.prop.addItem('MoE')
            self.prop.addItem('EoM')
            self.prop.addItem('EoE')
            self.prop.addItem('MoM')

    def open_aim(self):
        fname = QFileDialog.getOpenFileName(self, 'Select AIM cube file', '.','*.cube')[0]
        aimcube = cube_parser(fname)
        if aimcube.nval == 1:
            pass  # OK
        elif aimcube.nval == 2:
            print("Debug: Aberto's cube, only 2nd value kept for now")
            aimcube.nval = 1
            # Keep the second value and ensure proper 1D structure
            aimcube.cube = aimcube.cube[1, :]
        else:
            raise NoValidData("open_aim", "aim cube must contain a single scalar dataset")
        self._fchk.add_aim(aimcube)
        if self._fchk._frags is not None:
            self._menus['cubes']["saim"].setEnabled(True)
            self._menus['cubes']['dmt'].setEnabled(True)
            self._menus['etcddtm']['frags'].setEnabled(True)

    def opengroups(self):
        jname = QFileDialog.getOpenFileName(self, 'Select JSON file', '.','*.json')[0]
        try:
            jdata = read_json(jname)
            res = []
            if self._fchk is not None:
                natm = self._fchk.natoms
            else:
                natm = int(jdata["molecule"]["natoms"])
            
            # Process fragments from JSON
            for i in jdata["molecule"]["frags"]:
                if isinstance(i["fr_index"], str):
                    tmp = range_parse(i["fr_index"], natm, flatten=True)
                else:
                    tmp = i['fr_index']
                res.append([x-1 for x in tmp])
            
            # Check for atoms not included in any fragment
            all_listed_atoms = set()
            for frag in res:
                all_listed_atoms.update(frag)
            
            all_atoms = set(range(natm))
            unlisted_atoms = all_atoms - all_listed_atoms
            
            # Add unlisted atoms as an additional fragment if any exist
            self._has_auto_fragment = False
            if unlisted_atoms:
                unlisted_list = sorted(list(unlisted_atoms))
                res.append(unlisted_list)
                self._has_auto_fragment = True  # Flag to track auto-created fragment
                print(f"Added unlisted atoms to additional fragment: {[x+1 for x in unlisted_list]}")
            
            self._fchk.set_fragment(res)
            if self._fchk._aimdata is not None:
                self._menus['cubes']['saim'].setEnabled(True)
                self._menus['cubes']['frags'].setEnabled(True)

            # FIXME
            # self.__color = random_colors(len(res))
            # if not self.fname is None:
            #    pass
            #    self.acpbutton.setEnabled(True)

        except Exception as err:
            print(err)

    # App state functions
    def _setisoval(self):
        self._default['isoval']['iso'] = float(self.isoline.text())
        if 'tcd' in self._actors:
            try:
                for i, val  in enumerate([-self._default['isoval']['iso'],
                                          self._default['isoval']['iso']]):
                    self._actors['tcd'].filter.SetValue(i, val)
                self._actors['tcd'].filter.Update()
                self._updatereder()
            except AttributeError as err:
                print(err)
            

    def showtcdmt(self):
        # Controllare quelli checked e mostrarli
        # crd = np.self._fchk.crd
        flag = False
        veccrd = []
        veccmp = []
        vectyp = []
        coltyp = 4
        if 'tcddtm' in self._actors:
            self.ren.RemoveActor(self._actors['tcddtm'].actor)
        if self._menus['etcddtm']['tot'].isChecked() or self._menus['vtcddtm']['tot'].isChecked():
            flag = True
            veccrd.append(self._fchk.get_com())
            veccrd.append(self._fchk.get_com())
            vectyp.extend([coltyp, -coltyp])
            veccmp.extend(list(self._fchk.get_vol_integrals("1")))

        if self._menus['etcddtm']['frags'].isChecked() or self._menus['vtcddtm']['frags'].isChecked():
            flag = True
            fragindx = self._fchk.get_frag_indx()
            tmpvec = self._fchk.get_vol_integrals("1", tps='frags', cgs=False)
            for i, ind in enumerate(fragindx):
                veccrd.append(self._fchk.get_com(mask=list(ind)))
                veccrd.append(self._fchk.get_com(mask=list(ind)))
                veccmp.append(tmpvec[0][i, :])
                veccmp.append(tmpvec[1][i, :])
                vectyp.extend([coltyp, -coltyp])
        if flag:
            self._actors['tcddtm'] = cubetk.draw_vectors(np.array(veccrd),
                                                         np.array(veccmp),
                                                         np.array(vectyp)) 
            self.ren.AddActor(self._actors['tcddtm'].actor)
        self._updatereder()            


    def cleartcdmt(self):
        if 'tcddtm' in self._actors:
            self.ren.RemoveActor(self._actors['tcddtm'].actor)
            self._actors.pop('tcddtm')
        for key in self._menus['etcddtm']:
            self._menus['etcddtm'][key].setChecked(False)
 
        self._updatereder()

    def showtcd(self):
        if 'tcd' in self._actors:
            self.ren.RemoveActor(self._actors['tcd'].actor)
            del self._actors['tcd']
        if 'tcddir' in self._actors:
            self.ren.RemoveActor(self._actors['tcddir'].actor)
            del self._actors['tcddir']
        if 'tcdbar' in self._actors:
            self.ren.RemoveActor(self._actors['tcdbar'].actor)
            del self._actors['tcdbar']
        
        # Stop any running animation
        if self._show_particles:
            self.stop_animation()

        prop_cur = self.prop.currentText().lower()
        if prop_cur == "":
            # print(self._actors.keys())
            self._updatereder()
            return None
        tmp_cube = copy.deepcopy(self._fchk.get_vol("1"))
        tmp_iso = [-self._default['isoval']['iso'],
                   self._default['isoval']['iso']]
        if prop_cur == "streamlines":
            if self._seeds is None:
                if self._default["vfield"]["sampling_method"] == "ellipsoid":
                    self._seeds = self._fchk.sample_ellipse_space(self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalellipse"])
                else:  # molecular volume
                    current_tcd = self._fchk.get_vol("1")
                    self._seeds = sample_molecular_volume(current_tcd, self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalevdw"])
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            self._actors['tcd'] =  cubetk.fillstreamline(tmp_cube,
                                                         clipping=(self._default["vfield"]["vfmax"],
                                                                    self._default["vfield"]["vfmin"]),
                                                         minspeed=self._default["vfield"]["mspeed"],
                                                         seeds=self._seeds,
                                                         clip_bounds=self._default["vfield"]["clip_bounds"] if self._default["vfield"]["enable_clipping"] else None)
            if self._default["vfield"]["showbar"]:
                self._actors['tcdbar'] = cubetk.draw_colorbar(self._actors['tcd'].actor, "Norm(J)")
            if self._default["vfield"]["showdir"]:
                # attempt to draw direction with cones.
                # instead of using the seeds points, 
                # we can show hedehogs with cone gliphs
                # and a sparse sampling of the vector field
                # self._actors['tcddir'] = cubetk.draw_cones_nogrid(tmp_cube, self._seeds)
                self._actors['tcddir'] = cubetk.quiv3d(tmp_cube, 
                                               scale=self._default["quiver"]["scale"]/5,
                                               subsample_factor=100,
                                               glyphmode='cone',
                                               clip_bounds=self._default["vfield"]["clip_bounds"] if self._default["vfield"]["enable_clipping"] else None)
            
            # Add animated particles if enabled
            if self._default["vfield"]["animate_particles"]:
                # Get the raw streamline polydata (not the tube filter output)
                if hasattr(self._actors['tcd'], 'streamline_data'):
                    streamline_polydata = self._actors['tcd'].streamline_data
                else:
                    # Fallback to tube filter output if streamline_data not available
                    streamline_polydata = self._actors['tcd'].actor.GetMapper().GetInput()
                
                self._particle_animator = cubetk.create_streamline_particles(
                    self.ren, streamline_polydata, 
                    num_particles=self._default["vfield"]["num_particles"],
                    particle_type=self._default["vfield"]["particle_type"])
                self.start_animation()
            
        elif prop_cur == "quiver":
            mask_index = filtervecatom(tmp_cube, 0.3)
            tmp_cube.cube[:, mask_index] = 0
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            self._actors['tcd'] = cubetk.quiv3d(tmp_cube,
                                               lower=self._default["quiver"]["lower"],
                                               upper=self._default["quiver"]["upper"],
                                               scale=self._default["quiver"]["scale"],
                                               subsample_factor=self._default["quiver"]["subsample"],
                                               clip_bounds=self._default["quiver"]["clip_bounds"] if self._default["quiver"]["enable_clipping"] else None)
        elif prop_cur == "moe":
            self.isoline.setEnabled(True)
            vec = tmp_cube.integrate() 
            tmp_cube = tmp_cube.proj_on_vec(vec=vec, rot=True, cube=True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            print(f"Max: {np.max(tmp_cube.cube)}, Min: {np.min(tmp_cube.cube)}")
        elif prop_cur == "eom":
            self.isoline.setEnabled(True)
            vec = tmp_cube.rotorintegrate()
            tmp_cube = tmp_cube.proj_on_vec(vec=vec, rot=False, cube=True)
            tmp_cube.cube 
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            print(f"Max: {np.max(tmp_cube.cube)}, Min: {np.min(tmp_cube.cube)}")
        elif prop_cur == "eoe":
            self.isoline.setEnabled(True)
            vec = tmp_cube.integrate()
            tmp_cube = tmp_cube.proj_on_vec(vec=vec*2, rot=False, cube=True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            print(f"Max: {np.max(tmp_cube.cube)}, Min: {np.min(tmp_cube.cube)}")

        elif prop_cur == "mom":
            self.isoline.setEnabled(True)
            vec = tmp_cube.rotorintegrate()
            tmp_cube = tmp_cube.proj_on_vec(vec=vec, rot=True, cube=True)
                # tmp_cube.cube /= self._fchk.get_exeng(self._activest)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            print(f"Max: {np.max(tmp_cube.cube)}, Min: {np.min(tmp_cube.cube)}")
        elif prop_cur == "isosurface":
            self.isoline.setEnabled(True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            print(f"Max: {np.max(tmp_cube.cube)}, Min: {np.min(tmp_cube.cube)}")

        self.ren.AddActor(self._actors['tcd'].actor)
        if 'tcdbar' in self._actors:
            self.ren.AddActor2D(self._actors['tcdbar'].actor)
        if 'tcddir' in self._actors:
            self.ren.AddActor(self._actors['tcddir'].actor)
        
        self._updatereder()

    def configure_molecule(self):
        """Open dialog to configure molecule display settings"""
        config_dialog = MoleculeConfigDialog(
            parent=self,
            wireframe=self._default['molconfig']['wireframe'],
            opacity=self._default['molconfig']['opacity'],
            bond_radius=self._default['molconfig']['bond_radius'],
            atom_radius_scale=self._default['molconfig']['atom_radius_scale'],
            tubes_mode=self._default['molconfig']['tubes_mode'],
            bond_tollerance=self._default['molconfig']['bond_tollerance'],
            hide_auto_group=self._default['molconfig']['hide_auto_group'],
            has_auto_fragment=self._has_auto_fragment and self._fchk._frags is not None
        )

        
        config_dialog.exec()
        
        if config_dialog._okexit:
            # Update configuration
            self._default['molconfig']['wireframe'] = config_dialog._wireframe
            self._default['molconfig']['opacity'] = config_dialog._opacity
            self._default['molconfig']['bond_radius'] = config_dialog._bond_radius
            self._default['molconfig']['atom_radius_scale'] = config_dialog._atom_radius_scale
            self._default['molconfig']['tubes_mode'] = config_dialog._tubes_mode
            self._default['molconfig']['bond_tollerance'] = config_dialog._bond_tollerance
            self._default['molconfig']['hide_auto_group'] = config_dialog._hide_auto_group
            
            # Refresh molecule display
            self._refresh_molecule_display()

    def _refresh_molecule_display(self):
        """Refresh the molecule display with current settings"""
        if 'mol' in self._actors:
            # Remove current molecule
            self.ren.RemoveActor(self._actors['mol'].actor)
            
            # Determine excluded atoms if hide auto group is enabled
            excluded_atoms = None
            if (self._default['molconfig']['hide_auto_group'] and 
                self._has_auto_fragment and 
                self._fchk._frags is not None and 
                len(self._fchk._frags) > 0):
                excluded_atoms = self._fchk._frags['indx'][-1]  # Last fragment is auto-generated
            
            # Create new molecule with updated settings
            self._actors['mol'] = cubetk.fillmolecule_custom(
                self._fchk.atnum,
                self._fchk.crd,
                opacity=self._default['molconfig']['opacity'],
                bond_radius=self._default['molconfig']['bond_radius'],
                atom_radius_scale=self._default['molconfig']['atom_radius_scale'],
                tubes_mode=self._default['molconfig']['tubes_mode'],
                bond_tollerance=self._default['molconfig']['bond_tollerance'],
                excluded_atoms=excluded_atoms
            )
            
            # Add updated molecule to renderer
            self.ren.AddActor(self._actors['mol'].actor)
            self._updatereder()

    def toggle_mol_transparency(self):
        """Toggle molecule transparency between opaque and semi-transparent"""
        if 'mol' in self._actors:
            is_transparent = self._menus['mol']['transparent'].isChecked()
            alpha = 0.3 if is_transparent else 1.0
            
            # Apply transparency to all parts of the molecule
            mol_actor = self._actors['mol'].actor
            mol_actor.GetProperty().SetOpacity(alpha)
            
            # If there are multiple components (atoms, bonds, etc.), apply to all
            if hasattr(self._actors['mol'], 'sphere_actors'):
                for sphere_actor in self._actors['mol'].sphere_actors:
                    sphere_actor.GetProperty().SetOpacity(alpha)
            
            if hasattr(self._actors['mol'], 'cylinder_actors'):
                for cyl_actor in self._actors['mol'].cylinder_actors:
                    cyl_actor.GetProperty().SetOpacity(alpha)
                    
            self._updatereder()

    # AIM functions
    def show_aim_space(self):
        # Check the correct menu based on molecule type
        saim_checked = False
        saim_checked = self._menus['cubes']['saim'].isChecked()
            
        if saim_checked:
            tmp_cube = self._fchk.get_vol("1").get_frag_isosurf()
            _mask = vdw_boolean_mask(tmp_cube, thresh=1)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube = np.where(_mask, tmp_cube.cube, -1e10)  # Mask out values outside the vdW surface
            colors = self._fchk.get_frag_colors()
            grids = cubetk.fillcubeimage(tmp_cube, vec=False, aslist=True)
            
            # Determine how many fragments to display (skip auto-generated fragment if it exists)
            num_fragments = len(grids)
            if self._has_auto_fragment:
                num_fragments -= 1  # Skip the last fragment (auto-generated for missing atoms)
            
            for i in range(num_fragments):
                grd = grids[i]
                isoactor = cubetk._countur(grd, [1],
                                         active="scalar",
                                         colors=[colors[i]],
                                         opacity=0.1)
                self.ren.AddActor(isoactor.actor)
                self._actors[f"aimiso{i:d}"] = isoactor
        else:
            # Remove all existing AIM isosurfaces
            aimiso_keys = [key for key in self._actors.keys() if key.startswith("aimiso")]
            for key in aimiso_keys:
                self.ren.RemoveActor(self._actors[key].actor)
                del self._actors[key]
        self._updatereder()

    def open_scene(self):
        fname = QFileDialog.getOpenFileName(self, 'Select VTK Scene file',
                                            '.','*.json')[0]
        jsonfile = os.path.basename(fname)
        try:
            scene = read_json(fname)
            camera = self.ren.GetActiveCamera()
            camera.SetPosition(scene['Camera:Position'])
            camera.SetFocalPoint(scene['Camera:FocalPoint'])
            camera.SetViewUp(scene['Camera:ViewUp'])
            camera.SetViewAngle(scene['Camera:ViewAngle'])
            camera.SetClippingRange(scene['Camera:ClippingRange'])
            self.iren.Start()
            self._default = scene['Used parameters']
            # print("NYI")
        except FileNotFoundError as err:
            print(err)
        except KeyError:
            print(f"{jsonfile} not a VTK scene")
        
    def save_scene(self):
        jsondialog = SaveSceneDialog()
        jsondialog.exec()
        fname = jsondialog._fname
        camera = self.ren.GetActiveCamera()
        res = {}
        res['Camera:FocalPoint'] = camera.GetFocalPoint()
        res['Camera:Position'] = camera.GetPosition()
        res['Camera:ViewUp'] = camera.GetViewUp()
        res['Camera:ViewAngle'] = camera.GetViewAngle()
        res['Camera:ClippingRange'] = camera.GetClippingRange()
        res['Used parameters'] = self._default
        write_json(res, fname)

    def export_scene_pov(self):
        dlg = ExportPOVDialog(parent=self)
        if dlg.exec() == ExportPOVDialog.Accepted:
            if dlg.export(self.vtkWidget.GetRenderWindow()):
                if 'mol' in self._actors:
                    write_molecule_pov(self._actors['mol'], dlg._fname)

    def _batch_operations(self):
        """Performs batch operations on multiple files using a configuration JSON file.
        This function open the JSON file via a file dialog, reads the file and performs the operations.
        The structure of the JSON is:
        {
            "systems":[
            {
                "fname": "prefix_fname",
                "fchk": "path_to_fchk",
                "cubes":{"#nstate": "path_to_cube"},
                "aim": "path_to_aim_cube",
                "groups": "path_to_groups_json",
            }
            ],
            "operations":[{
                "type": "operation_type", # e.g. streamlines, quiver, moe, eom, eoe
                "show_dmt": true/false,
                "show_nm": true/false,
                "image_size": [width, height],
                "dpi": 300,
                "output_dir": "path_to_output_directory"
                }
            ],
            "settings": "path_to_settings_json"
        }
        """
        pass


def main():
    """Main entry point for the TCD visualization application.
    
    Creates a QApplication instance, initializes the TCDvis main window,
    and starts the event loop. This function is intended for use when
    running the script directly from the command line.
    
    For library usage, create a TCDvis instance directly instead:
        app = QApplication(sys.argv)
        window = TCDvis(moldata=your_data)
        window.show()
        app.exec()
    """
    app = QApplication(sys.argv)
    mainWin = TCDvis()
    mainWin.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


