"""Script to visualize the TCD data.
    It can display both electronic and vibrational TCD data.
    The script is based on PySide6 and VTK libraries.

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
from tcdlibx.calc.cube_manip import VecCubeData, VtcdData, cube_parser
from tcdlibx.graph.helpers import EleMolecule, VibMolecule, filtervecatom, sample_molecular_volume
import tcdlibx.graph.cube_graphvtk as cubetk
from tcdlibx.gui.dialogs import (
    SavePngDialog, SavePngSeriesDialog, StreamLineSetupDialog, TCDDialog, QuiverSetupDialog, SaveSceneDialog, NMConfigDialog, MoleculeConfigDialog
)
from tcdlibx.io.estp_io import PHYSFACT, get_elemol, get_vibmol
from tcdlibx.io.jsonio import read_json, write_json
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.var_tools import range_parse
from tcdlibx.utils.vtk_utils import QVTKRenderWindowInteractor, vtk

# Module configuration
DEBUG = True

class TCDvis(QMainWindow):
    """TCD Visualization Application.
    
    A PySide6/VTK-based application for visualizing Transition Current Density (TCD)
    data for both electronic and vibrational transitions.
    
    This class provides a complete GUI interface for:
    - Loading and visualizing molecular structure data (fchk files)
    - Displaying electronic and vibrational TCD data as streamlines, quiver plots, etc.
    - Animated particle visualization along streamlines
    - Interactive isosurface and vector field visualization
    - Export capabilities for images and data
    
    Args:
        moldata: Optional molecular data (EleMolecule or VibMolecule instance).
                If provided, the application will initialize with this data loaded.
    """
    
    def __init__(self, moldata: tp.Optional[tp.Union[EleMolecule, VibMolecule]] = None):
        """Initialize the TCD visualization application.
        
        Args:
            moldata: Optional molecular data to load on startup.
        """
        super().__init__()

        self._lastfname = 'tcdfigure.png'
        # self._cube = None
        self._fchk = moldata
        if isinstance(moldata, EleMolecule):
            self._moltype = 'ele'
        elif isinstance(moldata, VibMolecule):
            self._moltype = 'vib'
        else:
            self._moltype = None
        self._nstates = 1
        self._activest = 0
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
        
        self._default = {'isoval': {'iso': 0.01},
                         'vfield': {'vfmax': 1e2,
                                    'vfmin': 1e5,
                                    'mspeed': None,
                                    'npoints': 100,
                                    'scalellipse': 3.,
                                    'scalevdw': 2.0,
                                    'sampling_method': 'ellipsoid',
                                    'showdir': False,
                                    'showseeds': False,
                                    'conescale': .1,
                                    'showbar': False,
                                    'animate_particles': False,
                                    'num_particles': 15,
                                    'particle_type': 'sphere'},
                         'quiver': {'scale': 100,
                                   'subsample': 5},
                         'nmconfig': {'invert_phase': False,
                                     'scale_factor': 1.0,
                                     'color': (0.0, 0.0, 1.0)},
                         'molconfig': {'wireframe': False,
                                      'opacity': 1.0,
                                      'bond_radius': 0.03,
                                      'atom_radius_scale': 0.1,
                                      'tubes_mode': False,
                                      'bond_tollerance': 0.23,
                                      'hide_auto_group': False},}

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
        ## Open fchk
        openButton = QAction(QIcon(''), 'Open', self)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open a fchk file')
        openButton.triggered.connect(self.open_fchk)
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
        
        # NM buttons
        nmmenu = fileMol.addMenu('NM')
        nmmenu.setEnabled(False)
        self._menus['mol']['nm'] = {}
        nmButton = QAction(QIcon(''), 'Display NM', self)
        nmButton.setStatusTip('Display the NM vectors')
        nmButton.setCheckable(True)
        nmButton.triggered.connect(self.shownm)
        nmButton.setEnabled(True)
        nmmenu.addAction(nmButton)
        self._menus['mol']['nm']['menu'] = nmmenu
        self._menus['mol']['nm']['disp'] = nmButton
        nmConfig = QAction(QIcon(''), 'Configure NM', self)
        nmConfig.setStatusTip('Configure normal mode display options (phase, scale, color)')
        nmConfig.triggered.connect(self.configure_nm)
        nmConfig.setEnabled(True)
        nmmenu.addAction(nmConfig)
        self._menus['mol']['nm']['config'] = nmConfig
        # Ele. DMT
        fileEDmt = fileMol.addMenu('Ele. DTMs')
        fileEDmt.setEnabled(False) 
        self._menus['eledmt'] = {}
        self._menus['eledmt']['menu'] = fileEDmt
        eemtButton = QAction(QIcon(''), 'Ele.', self)
        eemtButton.setStatusTip('Display the electronic electric and magnetic DTMs from fchk')
        eemtButton.setCheckable(True)
        eemtButton.triggered.connect(self.showdmt)
        fileEDmt.addAction(eemtButton)
        self._menus['eledmt']['ele'] = eemtButton
        nomtButton = QAction(QIcon(''), 'Clear', self)
        nomtButton.setStatusTip('remove all DTM vectors')
        nomtButton.triggered.connect(self.cleardmt)
        fileEDmt.addAction(nomtButton)
        self._menus['mol']['edmt'] = fileEDmt
        # Vib. DMT
        fileVDmt = fileMol.addMenu('Vib. DTMs')
        fileVDmt.setEnabled(False) 
        self._menus['vibdmt'] = {}
        self._menus['vibdmt']['menu'] = fileVDmt
        dmtButton = QAction(QIcon(''), 'Total', self)
        dmtButton.setStatusTip('Display the electric and magnetic DTMs from fhck atomic tensors')
        dmtButton.setCheckable(True)
        dmtButton.triggered.connect(self.showdmt)
        fileVDmt.addAction(dmtButton)
        self._menus['vibdmt']['tot'] = dmtButton
        emtButton = QAction(QIcon(''), 'Ele.', self)
        emtButton.setStatusTip('Display the electronic components the electric and magnetic DTMs from fhck atomic tensors')
        emtButton.setCheckable(True)
        emtButton.triggered.connect(self.showdmt)
        fileVDmt.addAction(emtButton)
        self._menus['vibdmt']['ele'] = emtButton
        nmtButton = QAction(QIcon(''), 'Nuc.', self)
        nmtButton.setStatusTip('Display the nuclear components of electric and magnetic DTMs from fhck atomic tensors')
        nmtButton.setCheckable(True)
        nmtButton.triggered.connect(self.showdmt)
        fileVDmt.addAction(nmtButton)
        self._menus['vibdmt']['nuc'] = nmtButton
        nomtButton = QAction(QIcon(''), 'Clear', self)
        nomtButton.setStatusTip('remove all DTM vectors')
        nomtButton.triggered.connect(self.cleardmt)
        fileVDmt.addAction(nomtButton)
        self._menus['mol']['vibdmt'] = fileVDmt

        # ETCD
        self._menus['etcd'] = {}
        fileetcd = mainMenu.addMenu('ETCD')
        etcdButton = QAction(QIcon(''), 'Load ETCD', self)
        etcdButton.setStatusTip('Load a ETCD cube file')
        etcdButton.triggered.connect(self.open_cube)
        etcdButton.setEnabled(False)
        self._menus['etcd']['tcd'] = etcdButton
        fileetcd.addAction(etcdButton)
        aimButton = QAction(QIcon(''), 'Add AIM', self)
        aimButton.setStatusTip('Loads a cube file with AIM partition of the cube')
        aimButton.triggered.connect(self.open_aim)
        aimButton.setEnabled(False)
        self._menus['etcd']['aim'] = aimButton
        fileetcd.addAction(aimButton)
        saimButton = QAction(QIcon(''), 'Show AIM Frag Space', self)
        saimButton.setStatusTip('Shows the volume of each fragments in AIM partition scheme')
        saimButton.triggered.connect(self.show_aim_space)
        saimButton.setEnabled(False)
        saimButton.setCheckable(True)
        self._menus['etcd']['saim'] = saimButton
        fileetcd.addAction(saimButton)
        # ETCD dtm
        fileTCDmt = fileetcd.addMenu('ETCD DTMs')
        fileTCDmt.setEnabled(False)
        self._menus['etcd']['dmt'] = fileTCDmt
        self._menus['etcddtm'] = {}
        tcdmtButton = QAction(QIcon(''), 'Total', self)
        tcdmtButton.setStatusTip('Display the electric and magnetic DTMs from ETCD calculations')
        tcdmtButton.setCheckable(True)
        tcdmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(tcdmtButton)
        self._menus['etcddtm']['tot'] = tcdmtButton
        frtcmtButton = QAction(QIcon(''), 'Fragments Contr.', self)
        frtcmtButton.setStatusTip('Contributions of each fragment to electronic DTMs')
        frtcmtButton.setCheckable(True)
        frtcmtButton.setEnabled(False)
        frtcmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(frtcmtButton)
        self._menus['etcddtm']['frags'] = frtcmtButton
        nomtButton = QAction(QIcon(''), 'Clear', self)
        nomtButton.setStatusTip('remove all ETCD DTM vectors')
        nomtButton.triggered.connect(self.cleartcdmt)
        fileTCDmt.addAction(nomtButton)

        # VTCD
        self._menus['vtcd'] = {}
        filevtcd = mainMenu.addMenu('VTCD')
        vtcdButton = QAction(QIcon(''), 'Load VTCD', self)
        vtcdButton.setStatusTip('Load a VTCD cube file')
        vtcdButton.triggered.connect(self.open_cube)
        vtcdButton.setEnabled(False)
        self._menus['vtcd']['tcd'] = vtcdButton
        filevtcd.addAction(vtcdButton)
        aimButton = QAction(QIcon(''), 'Add AIM', self)
        aimButton.setStatusTip('Loads a cube file with AIM partition of the cube')
        aimButton.triggered.connect(self.open_aim)
        aimButton.setEnabled(False)
        self._menus['vtcd']['aim'] = aimButton
        filevtcd.addAction(aimButton)
        saimButton = QAction(QIcon(''), 'Show AIM Frag Space', self)
        saimButton.setStatusTip('Shows the volume of each fragment in AIM partition scheme')
        saimButton.triggered.connect(self.show_aim_space)
        saimButton.setEnabled(False)
        saimButton.setCheckable(True)
        self._menus['vtcd']['saim'] = saimButton
        filevtcd.addAction(saimButton)
        # VTCD dtm
        fileTCDmt = filevtcd.addMenu('VTCD DTMs')
        fileTCDmt.setEnabled(False)
        self._menus['vtcd']['dmt'] = fileTCDmt
        self._menus['vtcddtm'] = {}
        tcdmtButton = QAction(QIcon(''), 'Total', self)
        tcdmtButton.setStatusTip('Display the electric and magnetic DTMs from VTCD calculations')
        tcdmtButton.setCheckable(True)
        tcdmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(tcdmtButton)
        self._menus['vtcddtm']['tot'] = tcdmtButton
        frtcmtButton = QAction(QIcon(''), 'Fragments Contr.', self)
        frtcmtButton.setStatusTip('Contributions of each fragment to electronic DTMs')
        frtcmtButton.setCheckable(True)
        frtcmtButton.setEnabled(False)
        frtcmtButton.triggered.connect(self.showtcdmt)
        fileTCDmt.addAction(frtcmtButton)
        self._menus['vtcddtm']['frags'] = frtcmtButton
        nomtButton = QAction(QIcon(''), 'Clear', self)
        nomtButton.setStatusTip('remove all VTCD DTM vectors')
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
        self._trtext = QLabel("Transition number:")
        self.stline = QLineEdit()
        stval = QIntValidator(1, self._nstates)
        stval.setLocale(QLocale('English'))
        self.stline.setValidator(stval)
        self.stline.setText("1")
        self.stline.editingFinished.connect(self._setstate)
        self.stline.setEnabled(False)
        self.message2 = QLabel(f"of {self._nstates}")
        hlay.addWidget(self._trtext)
        hlay.addWidget(self.stline)
        hlay.addWidget(self.message2)
        hlay.addItem(QSpacerItem(100, 10, QSizePolicy.Expanding))
        # TCD types
        self.prop = QComboBox()
        self.prop.setGeometry(QRect(40, 40, 491, 31))
        self.prop.addItem('')
        self.prop.addItem('Streamlines')
        self.prop.addItem('Quiver')
        self.prop.addItem('MoE')
        self.prop.addItem('EoM')
        self.prop.addItem('EoE')
        self.prop.currentIndexChanged.connect(self._enablefieldsetup)
        grid.addWidget(self.prop, 0, 1)
        self.etcdch = QPushButton('Show ETCD field', self)
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
            self._enable_molmenu()
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
            self._updatenst()
            self.stline.setEnabled(True)

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
        # Reset state line to default state
        self.stline.setText("1")
        self.stline.setEnabled(False)
        self.stline.setStyleSheet("color: black;  background-color: white")
        
        # Reset TCD checkbox/control
        self.etcdch.setEnabled(False)
        
        # Reset isoline control
        self.isoline.setText(f"{self._default['isoval']['iso']:.6f}")
        self.isoline.setEnabled(False)
        
        # Reset property combo box to first item
        if hasattr(self, 'prop'):
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

    def _enable_molmenu(self):
        for val in self._menus['mol']:
            #print(val)
            if not val == 'nm' and 'dmt' not in val:
                self._menus['mol'][val].setEnabled(True)
                
        # Enable molecule-type specific menus
        if self._moltype == 'vib':
            self._menus['mol']['nm']['menu'].setEnabled(True)
            self._menus['vibdmt']['menu'].setEnabled(True)
        elif self._moltype == 'ele':
            self._menus['eledmt']['menu'].setEnabled(True)
            
        # Enable TCD menus based on molecule type
        if self._moltype == 'ele':
            self._menus['etcd']['tcd'].setEnabled(True)
        elif self._moltype == 'vib':
            self._menus['vtcd']['tcd'].setEnabled(True)
        # for val in self._menus['vtcd']:
        #    val.setEnabled(True)

    def _disable_molmenu(self):
        for val in self._menus['mol']:
            if isinstance(self._menus['mol'][val], dict):
                self._menus['mol'][val]['menu'].setEnabled(False)
            else:
                self._menus['mol'][val].setEnabled(False)
        for val in self._menus['etcd']:
            self._menus['etcd'][val].setEnabled(False)
        self._menus['etcddtm']['frags'].setEnabled(False)
        for val in self._menus['vtcd']:
            self._menus['vtcd'][val].setEnabled(False)
        self._menus['vtcddtm']['frags'].setEnabled(False)

    def _fieldstp(self):
        current_prop = self.prop.currentText().lower()
        
        if current_prop == "streamlines":
            # Handle streamlines setup dialog
            if self._default["vfield"]["mspeed"] is None:
                self._default["vfield"]["mspeed"] = self._fchk.get_tcd(self._activest)._maxnorm/1e4
            fieldprm = StreamLineSetupDialog(vfmax=self._default["vfield"]["vfmax"],
                                             vfmin=self._default["vfield"]["vfmin"],
                                             mspeed=self._default["vfield"]["mspeed"],
                                             maxval=self._fchk.get_tcd(self._activest)._maxnorm,
                                             nseeds=self._default["vfield"]["npoints"],
                                             scale=self._default["vfield"]["scalellipse"],
                                             direction=self._default["vfield"]["showdir"],
                                             showellipse=self._default["vfield"]["showseeds"],
                                             showbar=self._default["vfield"]["showbar"],
                                             animate_particles=self._default["vfield"]["animate_particles"],
                                             num_particles=self._default["vfield"]["num_particles"],
                                             particle_type=self._default["vfield"]["particle_type"],
                                             sampling_method=self._default["vfield"]["sampling_method"],
                                             scalevdw=self._default["vfield"]["scalevdw"],)
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
                    current_tcd = self._fchk.get_tcd(self._activest)
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
                        tmp_cube = copy.deepcopy(self._fchk.get_tcd(self._activest))
                        # self._actors['tcddir'] = cubetk.draw_cones_nogrid(tmp_cube, self._seeds)
                        self._actors['tcddir'] = cubetk.quiv3d(tmp_cube, 
                                               scale=self._default["quiver"]["scale"]/5,
                                               subsample_factor=100,
                                               glyphmode='cone')
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
                        current_tcd = self._fchk.get_tcd(self._activest)
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
                                         subsamp=self._default["quiver"]["subsample"])
            quiverprm.exec()
            # Update the default values
            self._default["quiver"]["scale"] = quiverprm._scale
            self._default["quiver"]["subsample"] = quiverprm._subsamp
            
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
            if self._activest in self._fchk.avail_tcd():
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
    def open_fchk(self):
        fname = QFileDialog.getOpenFileName(self, 'Select FCHK file', '.','*.fchk')[0]
        if len(fname) == 0:
            return None
        self._fchkfname = os.path.basename(fname)
        try:
            self._fchk = EleMolecule(get_elemol(fname))
            self._moltype = 'ele'
        except IndexError:
            try:
                self._fchk = VibMolecule(get_vibmol(fname))
                self._moltype = 'vib'
            # Fix properly the exceptions
            except Exception as err:
                print(err)
        except Exception as err:
            print(err)

        self._cleanactors()
        self._actors = {}
        self._has_auto_fragment = False  # Reset auto-fragment flag when opening new file
        
        # Reset active state to first state
        self._activest = 0
        
        # Reset interface state to prevent issues when switching molecule types
        self._reset_interface_state()
        
        # for key in self._actors:
        #     self._actors[key] = None
        self._updatenst()
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
        self._disable_molmenu()
        self._enable_molmenu()
        self.ren.ResetCamera()
        self.stline.setEnabled(True)

    def open_cube(self):
        texts = {'ele': ['ETCD', 'Transition'],
                 'vib': ['VTCD', 'NM']}
        cub = TCDDialog(maxval=self._fchk.ntrans, texts=texts[self._moltype])
        cub.exec()
        if cub._cube is None:
            return None
        if self._moltype == 'ele':
            cubdata = VecCubeData(cube_parser(cub._cube))
            self._menus['etcd']["aim"].setEnabled(True)
            self._menus['etcd']['dmt'].setEnabled(True)
            # Enable fragment contributions if AIM data and fragments are available
            if self._fchk._aimdata is not None and self._fchk._frags is not None:
                self._menus['etcddtm']['frags'].setEnabled(True)
        else:
            cubdata = VtcdData(cube_parser(cub._cube),
                           self._fchk._moldata['evec'][cub._vib-1],
                           self._fchk._moldata['freq'][cub._vib-1])
            self._menus['vtcd']["aim"].setEnabled(True)
            self._menus['vtcd']['dmt'].setEnabled(True)
            # Enable fragment contributions if AIM data and fragments are available
            if self._fchk._aimdata is not None and self._fchk._frags is not None:
                self._menus['vtcddtm']['frags'].setEnabled(True)
        self._fchk.add_tcd(cub._vib-1, cubdata)
        self.stline.setText(f"{cub._vib}")
        self._setstate()

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
            if self._moltype == 'ele':
                self._menus['etcd']["saim"].setEnabled(True)
            elif self._moltype == 'vib':
                self._menus['vtcd']["saim"].setEnabled(True)
            if self._activest in self._fchk.avail_tcd():
                if self._moltype == 'ele':
                    self._menus['etcd']['dmt'].setEnabled(True)
                    self._menus['etcddtm']['frags'].setEnabled(True)
                elif self._moltype == 'vib':
                    self._menus['vtcd']['dmt'].setEnabled(True)
                    self._menus['vtcddtm']['frags'].setEnabled(True)

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
            if (self._fchk._aimdata is not None and 
                self._activest in self._fchk.avail_tcd()):
                if self._moltype == 'ele':
                    self._menus['etcddtm']['frags'].setEnabled(True)
                elif self._moltype == 'vib':
                    self._menus['vtcddtm']['frags'].setEnabled(True)

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
            
    def _updatenst(self):
        self._nstates = self._fchk.ntrans
        if self._moltype == 'vib':
            self._trtext.setText("Normal mode number:")
        elif self._moltype == 'ele':
            self._trtext.setText("Transition number:") 
        stval = QIntValidator(1, self._nstates)
        stval.setLocale(QLocale('English'))
        self.stline.setValidator(stval)
        self.message2.setText(f"of {self._nstates}")

    def _setstate(self):
        self._activest = int(self.stline.text()) - 1
        
        # Stop any running particle animation and reset animation flag
        if self._show_particles:
            self.stop_animation()
        self._default["vfield"]["animate_particles"] = False
        
        # check and enable/disable vfield setup button
        self._enablefieldsetup()
        if 'tcd' in self._actors:
            self.ren.RemoveActor(self._actors['tcd'].actor) 
        self.cleartcdmt()
        if self._activest in self._fchk.avail_tcd():
            self.stline.setStyleSheet("color: white;  background-color: black")
            self.etcdch.setEnabled(True)
            self.isoline.setEnabled(True)
            if self._moltype == 'ele':
                self._menus['etcd']['dmt'].setEnabled(True)
                # Enable fragment contributions if AIM data and fragments are available
                if self._fchk._aimdata is not None and self._fchk._frags is not None:
                    self._menus['etcddtm']['frags'].setEnabled(True)
            elif self._moltype == 'vib':
                self._menus['vtcd']['dmt'].setEnabled(True)
                # Enable fragment contributions if AIM data and fragments are available
                if self._fchk._aimdata is not None and self._fchk._frags is not None:
                    self._menus['vtcddtm']['frags'].setEnabled(True)
        else:
            self.stline.setStyleSheet("color: black;  background-color: white")
            self.etcdch.setEnabled(False)
            self.isoline.setEnabled(False)
            if self._moltype == 'ele':
                self._menus['etcd']['dmt'].setEnabled(False)
            elif self._moltype == 'vib':
                self._menus['vtcd']['dmt'].setEnabled(False)
        if 'mfpdtm' in self._actors:
            self.showdmt()
        
        # Refresh TCD DTM visualization if it's currently displayed
        if 'tcddtm' in self._actors:
            self.showtcdmt()
        
        # Refresh NM visualization if it's currently displayed (only for vibrational molecules)
        if self._moltype == 'vib' and self._menus['mol']['nm']['disp'].isChecked():
            self.shownm()

    def showdmt(self):
        # Controllare quelli checked e mostrarli
        # crd = np.self._fchk.crd
        flag = False
        veccrd = []
        veccmp = []
        vectyp = []
        coltyp = {'tot': 3,
                  'ele': 2,
                  'nuc': 1}
        if 'mfpdtm' in self._actors:
            self.ren.RemoveActor(self._actors['mfpdtm'].actor)
        if self._moltype == 'ele':
            mkey = 'eledmt'
        elif self._moltype == 'vib':
            mkey = 'vibdmt'
        for key in self._menus[mkey]:
            if key != 'menu':
                if self._menus[mkey][key].isChecked():
                    flag = True
                    veccrd.append(self._fchk.get_com())
                    veccrd.append(self._fchk.get_com())
                    tmp_dip = self._fchk.get_dtm(self._activest, tps=key, cgs=False)
                    veccmp.append(tmp_dip[0])
                    veccmp.append(tmp_dip[1])
                    vectyp.append(coltyp[key])
                    vectyp.append(-coltyp[key])
                    if DEBUG:
                        print("From fchk")
                        print(f"EDTM: {tmp_dip[0][0]:10.5f}{tmp_dip[0][1]:10.5f}{tmp_dip[0][2]:10.5f}")
                        print(f"MDTM: {tmp_dip[1][0]:10.5f}{tmp_dip[1][1]:10.5f}{tmp_dip[1][2]:10.5f}")
        if flag:
            self._actors['mfpdtm'] = cubetk.draw_vectors(np.array(veccrd),
                                                         np.array(veccmp),
                                                         np.array(vectyp)) 
            self.ren.AddActor(self._actors['mfpdtm'].actor)
        self._updatereder()

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
            veccmp.extend(list(self._fchk.get_tcd_dtm(self._activest, cgs=False)))
            if DEBUG:
                tmp_dip = self._fchk.get_tcd_dtm(self._activest, cgs=False)
                print("From Cube")
                print(f"EDTM: {tmp_dip[0][0]:10.5f}{tmp_dip[0][1]:10.5f}{tmp_dip[0][2]:10.5f}")
                print(f"MDTM: {tmp_dip[1][0]:10.5f}{tmp_dip[1][1]:10.5f}{tmp_dip[1][2]:10.5f}")

        if self._menus['etcddtm']['frags'].isChecked() or self._menus['vtcddtm']['frags'].isChecked():
            flag = True
            fragindx = self._fchk.get_frag_indx()
            tmpvec = self._fchk.get_tcd_dtm(self._activest, tps='frags', cgs=False)
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

    def cleardmt(self):
        if 'mfpdtm' in self._actors:
            self.ren.RemoveActor(self._actors['mfpdtm'].actor)
            self._actors.pop('mfpdtm')
        for key in self._menus['eledmt']:
            if key != 'menu':
                self._menus['eledmt'][key].setChecked(False)
        for key in self._menus['vibdmt']:
            if key != 'menu':
                self._menus['vibdmt'][key].setChecked(False)
        self._updatereder()

    def cleartcdmt(self):
        if 'tcddtm' in self._actors:
            self.ren.RemoveActor(self._actors['tcddtm'].actor)
            self._actors.pop('tcddtm')
        for key in self._menus['etcddtm']:
            self._menus['etcddtm'][key].setChecked(False)
        for key in self._menus['vtcddtm']:
            self._menus['vtcddtm'][key].setChecked(False)
 
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
        tmp_cube = copy.deepcopy(self._fchk.get_tcd(self._activest))
        tmp_iso = [-self._default['isoval']['iso'],
                   self._default['isoval']['iso']]
        if prop_cur == "streamlines":
            if self._seeds is None:
                if self._default["vfield"]["sampling_method"] == "ellipsoid":
                    self._seeds = self._fchk.sample_ellipse_space(self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalellipse"])
                else:  # molecular volume
                    current_tcd = self._fchk.get_tcd(self._activest)
                    self._seeds = sample_molecular_volume(current_tcd, self._default["vfield"]["npoints"], scale=self._default["vfield"]["scalevdw"])
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            self._actors['tcd'] =  cubetk.fillstreamline(tmp_cube,
                                                         clipping=(self._default["vfield"]["vfmax"],
                                                                    self._default["vfield"]["vfmin"]),
                                                         minspeed=self._default["vfield"]["mspeed"],
                                                         seeds=self._seeds)
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
                                               glyphmode='cone')
            
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
                                               scale=self._default["quiver"]["scale"],
                                               subsample_factor=self._default["quiver"]["subsample"])
        elif prop_cur == "moe":
            if self._moltype == 'ele':
                vec = tmp_cube.integrate() / self._fchk.get_exeng(self._activest)
                tmp_cube = tmp_cube.proj_on_vec(vec=vec, rot=True, cube=True)
            else:
                tmp_cube = tmp_cube.proj_on_vec("moe", nucl=False, cube=True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            if DEBUG:
                print(f"Integrated scalar: {tmp_cube.integrate():.5f}" )
                tmp = self._fchk.get_tcd_dtm(self._activest, cgs=False)
                print(f"From Vec cube: {np.dot(tmp[0], tmp[1]):.5f}")
                tmp = self._fchk.get_dtm(self._activest, cgs=False)
                print(f"From FCHK: {np.dot(tmp[0], tmp[1]):.5f}")
        elif prop_cur == "eom":
            if self._moltype == 'ele':
                vec = tmp_cube.rotorintegrate()
                tmp_cube = tmp_cube.proj_on_vec(vec=vec, rot=False, cube=True)
                tmp_cube.cube /= self._fchk.get_exeng(self._activest)
            else:
                tmp_cube = tmp_cube.proj_on_vec("eom", nucl=False, cube=True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            if DEBUG:
                print(f"Integrated scalar: {tmp_cube.integrate():.5f}" )
                tmp = self._fchk.get_tcd_dtm(self._activest, cgs=False)
                print(f"From Vec cube: {np.dot(tmp[0], tmp[1]):.5f}")
                tmp = self._fchk.get_dtm(self._activest, cgs=False)
                print(f"From FCHK: {np.dot(tmp[0], tmp[1]):.5f}")
        elif prop_cur == "eoe":
            vec = tmp_cube.integrate()
            if self._moltype == 'ele':
                vec /= self._fchk.get_exeng(self._activest)
                tmp_cube = tmp_cube.proj_on_vec(vec=vec*2, rot=False, cube=True)
            else:
                tmp_cube = tmp_cube.proj_on_vec("eoe", nucl=False, cube=True)
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang
            tmp_cube.cube /= self._fchk.get_transeng(self._activest)
            tmp_cube.cube /= PHYSFACT.bohr2ang**3
            self._actors['tcd'] = cubetk.countur(tmp_cube, tmp_iso)
            if DEBUG:
                print(f"Integrated scalar: {tmp_cube.integrate():.5f}" )
                tmp = self._fchk.get_tcd_dtm(self._activest, cgs=False)
                print(f"From Vec cube: {np.dot(tmp[0], tmp[0]):.5f}")
                tmp = self._fchk.get_dtm(self._activest, cgs=False)
                print(f"From FCHK: {np.dot(tmp[0], tmp[0]):.5f}")

        self.ren.AddActor(self._actors['tcd'].actor)
        if 'tcdbar' in self._actors:
            self.ren.AddActor2D(self._actors['tcdbar'].actor)
        if 'tcddir' in self._actors:
            self.ren.AddActor(self._actors['tcddir'].actor)
        
        self._updatereder()

    def configure_nm(self):
        """Open dialog to configure normal mode display options"""
        config_dialog = NMConfigDialog(
            parent=self,
            invert_phase=self._default['nmconfig']['invert_phase'],
            scale_factor=self._default['nmconfig']['scale_factor'],
            color=self._default['nmconfig']['color']
        )
        
        config_dialog.exec()
        
        if config_dialog._okexit:
            # Update configuration
            self._default['nmconfig']['invert_phase'] = config_dialog._invert_phase
            self._default['nmconfig']['scale_factor'] = config_dialog._scale_factor
            self._default['nmconfig']['color'] = config_dialog._color
            
            # Refresh display if NM is currently shown
            if self._menus['mol']['nm']['disp'].isChecked():
                self.shownm()

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

    def shownm(self):
        if 'nm' in self._actors:
            self.ren.RemoveActor(self._actors['nm'].actor)
            del self._actors['nm']
        if self._menus['mol']['nm']['disp'].isChecked():
            # Apply configuration settings
            phase_mult = -1 if self._default['nmconfig']['invert_phase'] else 1
            scale = self._default['nmconfig']['scale_factor']
            color = self._default['nmconfig']['color']
            
            self._actors['nm'] = cubetk.draw_nm3d(
                self._fchk.crd,
                self._fchk.get_evec(self._activest) * phase_mult,
                self._fchk.atnum,
                scale=scale,
                color=color
            )
            self.ren.AddActor(self._actors['nm'].actor)
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
        if self._moltype == 'ele':
            saim_checked = self._menus['etcd']['saim'].isChecked()
        elif self._moltype == 'vib':
            saim_checked = self._menus['vtcd']['saim'].isChecked()
            
        if saim_checked:
            ind = self._fchk.avail_tcd()[0]
            tmp_cube = self._fchk.get_tcd(ind).get_frag_isosurf()
            tmp_cube.loc2wrd *=  PHYSFACT.bohr2ang 
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


