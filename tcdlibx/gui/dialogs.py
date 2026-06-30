import os
import typing as tp
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PySide6.QtCore import QRegularExpression, QLocale, Qt, QObject, QEvent
from PySide6.QtGui import QRegularExpressionValidator, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QCheckBox, QComboBox
from PySide6.QtWidgets import QLineEdit, QFileDialog, QPushButton, QMessageBox, QColorDialog, QSlider, QRadioButton, QButtonGroup, QFrame
from tcdlibx.utils.var_tools import fuzzy_equal
from tcdlibx.graph.cube_graphvtk import create_clip_plane_actors, move_clip_plane


class _FocusFilter(QObject):
    """Event filter that triggers *callback* when the watched widget receives keyboard focus."""
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.FocusIn:
            self._callback()
        return False


class _ClipPlaneMixin:
    """Mixin that adds semi-transparent VTK clip-plane preview actors to clipping dialogs.

    Both *QuiverSetupDialog* and *StreamLineSetupDialog* inherit from this class.
    The mixin expects the dialog to expose ``_xmin_input / _xmax_input / _ymin_input /
    _ymax_input / _zmin_input / _zmax_input`` ``QLineEdit`` widgets and a
    ``_preview_planes_checkbox`` ``QCheckBox`` (created by
    :meth:`_setup_plane_preview_ui`).  Call order in ``__init__``:

    1. ``_init_clip_planes(renderer, render_window, scene_bounds)``
    2. Build the six bound ``QLineEdit`` widgets.
    3. ``_setup_plane_preview_ui(enable_clipping)``  (returns the checkbox widget)
    """

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_clip_planes(self, vtk_renderer, vtk_render_window, vtk_scene_bounds):
        """Store VTK objects and initialise the plane-state containers."""
        self._vtk_renderer = vtk_renderer
        self._vtk_render_window = vtk_render_window
        self._vtk_scene_bounds = vtk_scene_bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
        self._plane_actors = {}   # key -> vtkActor
        self._plane_sources = {}  # key -> vtkPlaneSource
        self._planes_active = False

    def _setup_plane_preview_ui(self, enable_clipping):
        """Create the 'Preview clipping planes' checkbox and wire up all signals.

        Returns the ``QCheckBox`` so the caller can add it to a layout.
        """
        has_vtk = (
            self._vtk_renderer is not None
            and self._vtk_scene_bounds is not None)
        self._preview_planes_checkbox = QCheckBox("Preview clipping planes")
        self._preview_planes_checkbox.setEnabled(enable_clipping and has_vtk)
        self._preview_planes_checkbox.stateChanged.connect(self._toggle_preview_planes)

        # Focus filters: when the user clicks into an axis' input the planes
        # for that axis become visible.
        self._x_focus_filter = _FocusFilter(lambda: self._on_axis_focus('x'), self)
        self._y_focus_filter = _FocusFilter(lambda: self._on_axis_focus('y'), self)
        self._z_focus_filter = _FocusFilter(lambda: self._on_axis_focus('z'), self)
        self._xmin_input.installEventFilter(self._x_focus_filter)
        self._xmax_input.installEventFilter(self._x_focus_filter)
        self._ymin_input.installEventFilter(self._y_focus_filter)
        self._ymax_input.installEventFilter(self._y_focus_filter)
        self._zmin_input.installEventFilter(self._z_focus_filter)
        self._zmax_input.installEventFilter(self._z_focus_filter)

        # Live position updates while the user types
        self._xmin_input.textChanged.connect(lambda _t: self._update_plane_position('xmin'))
        self._xmax_input.textChanged.connect(lambda _t: self._update_plane_position('xmax'))
        self._ymin_input.textChanged.connect(lambda _t: self._update_plane_position('ymin'))
        self._ymax_input.textChanged.connect(lambda _t: self._update_plane_position('ymax'))
        self._zmin_input.textChanged.connect(lambda _t: self._update_plane_position('zmin'))
        self._zmax_input.textChanged.connect(lambda _t: self._update_plane_position('zmax'))

        # Remove actors from the renderer whenever the dialog closes
        self.finished.connect(lambda _: self._cleanup_planes())

        return self._preview_planes_checkbox

    # ------------------------------------------------------------------
    # VTK plane management
    # ------------------------------------------------------------------

    def _get_plane_value(self, key):
        """Return the float coordinate for *key*; fall back to the scene edge if empty."""
        input_map = {
            'xmin': self._xmin_input, 'xmax': self._xmax_input,
            'ymin': self._ymin_input, 'ymax': self._ymax_input,
            'zmin': self._zmin_input, 'zmax': self._zmax_input,
        }
        default_map = {
            'xmin': self._vtk_scene_bounds[0], 'xmax': self._vtk_scene_bounds[1],
            'ymin': self._vtk_scene_bounds[2], 'ymax': self._vtk_scene_bounds[3],
            'zmin': self._vtk_scene_bounds[4], 'zmax': self._vtk_scene_bounds[5],
        }
        text = input_map[key].text().strip()
        try:
            return float(text) if text else default_map[key]
        except ValueError:
            return default_map[key]

    def _create_clip_planes(self):
        """Create six semi-transparent plane actors (hidden) and add to the renderer."""
        if not self._vtk_renderer or not self._vtk_scene_bounds:
            return
        initial = {key: self._get_plane_value(key)
                   for key in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')}
        self._plane_actors, self._plane_sources = create_clip_plane_actors(
            self._vtk_scene_bounds, initial)
        for actor in self._plane_actors.values():
            self._vtk_renderer.AddActor(actor)
        self._planes_active = True
        if self._vtk_render_window:
            self._vtk_render_window.Render()

    def _on_axis_focus(self, axis):
        """Show the two planes for *axis*, hide the rest."""
        if not self._preview_planes_checkbox.isChecked() or not self._planes_active:
            return
        for key, actor in self._plane_actors.items():
            actor.SetVisibility(key[0] == axis)
        if self._vtk_render_window:
            self._vtk_render_window.Render()

    def _update_plane_position(self, key):
        """Move the plane *key* to the coordinate currently typed in its input field."""
        if not self._planes_active or key not in self._plane_sources:
            return
        move_clip_plane(self._plane_sources[key], key,
                        self._get_plane_value(key), self._vtk_scene_bounds)
        if self._vtk_render_window:
            self._vtk_render_window.Render()

    def _toggle_preview_planes(self):
        """Show / hide clip planes in response to the preview checkbox."""
        if self._preview_planes_checkbox.isChecked():
            if not self._planes_active:
                self._create_clip_planes()
            self._on_axis_focus('x')
        else:
            for actor in self._plane_actors.values():
                actor.VisibilityOff()
            if self._vtk_render_window:
                self._vtk_render_window.Render()

    def _cleanup_planes(self):
        """Remove all clip-plane actors from the renderer (called on dialog close)."""
        if self._vtk_renderer:
            for actor in self._plane_actors.values():
                self._vtk_renderer.RemoveActor(actor)
            if self._planes_active and self._vtk_render_window:
                self._vtk_render_window.Render()
        self._plane_actors.clear()
        self._plane_sources.clear()
        self._planes_active = False

    def _update_clipping_toggle(self, is_enabled):
        """Helper called from ``_toggle_clipping`` to keep preview checkbox in sync."""
        has_vtk = (
            self._vtk_renderer is not None
            and self._vtk_scene_bounds is not None)
        self._preview_planes_checkbox.setEnabled(is_enabled and has_vtk)
        if not is_enabled:
            self._preview_planes_checkbox.setChecked(False)


class SavePngDialog(QDialog):
    def __init__(self, parent=None, fname="tcdfigure.png", figure_size=None):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        self._dpi = 150  # Default DPI
        self._folder = os.path.dirname(fname) if os.path.dirname(fname) else "."
        self._figure_size = figure_size  # (width, height) of the displayed figure
        self._width = None
        self._height = None
        
        # PNG name input
        nameLabel = QLabel('PNG name:', self)
        self.pngname = QLineEdit(self)
        png_valid = QRegularExpressionValidator()
        pattern = r'.*\.png'
        regexp = QRegularExpression(pattern)
        png_valid.setRegularExpression(regexp)
        self.pngname.setValidator(png_valid)
        self.pngname.setText(os.path.basename(self._fname))
        self.pngname.textEdited.connect(self._setcheck)

        # DPI input
        dpiLabel = QLabel('DPI:', self)
        self.dpiInput = QLineEdit(self)
        dpi_validator = QIntValidator(50, 1200, self)  # DPI range 50-1200
        self.dpiInput.setValidator(dpi_validator)
        self.dpiInput.setText(str(self._dpi))
        self.dpiInput.textEdited.connect(self._update_dpi)
        
        # File size options
        sizeLabel = QLabel('Image size:', self)
        self.sizeCombo = QComboBox(self)
        self.sizeCombo.addItem("1 column (8.25 cm width)", "1col")
        self.sizeCombo.addItem("2 column (17.8 cm width)", "2col")
        self.sizeCombo.addItem("Custom", "custom")
        self.sizeCombo.currentIndexChanged.connect(self._on_size_changed)
        
        # Custom width input (initially hidden)
        customLabel = QLabel('Custom width (px):', self)
        self.customWidthInput = QLineEdit(self)
        width_validator = QIntValidator(100, 10000, self)  # Width range 100-10000 px
        self.customWidthInput.setValidator(width_validator)
        # Set default custom width to 1 column at current DPI
        default_width = int((8.25 / 2.54) * self._dpi)
        self.customWidthInput.setText(str(default_width))
        self.customWidthInput.textEdited.connect(self._update_custom_size)
        
        # Initially hide custom width controls
        customLabel.setVisible(False)
        self.customWidthInput.setVisible(False)
        self._customLabel = customLabel
        
        # Set default size based on DPI
        self._update_size_from_dpi()
        
        # Folder selection
        folderLabel = QLabel('Save in:', self)
        self.folderPath = QLineEdit(self)
        self.folderPath.setText(self._folder)
        self.folderPath.setReadOnly(True)
        self.browseButton = QPushButton('Browse...', self)
        self.browseButton.clicked.connect(self._browse_folder)

        self.setWindowTitle("Save PNG file")

        # Layout setup
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(nameLabel, 0, 0)
        self.grid_layout.addWidget(self.pngname, 0, 1, 1, 2)
        self.grid_layout.addWidget(dpiLabel, 1, 0)
        self.grid_layout.addWidget(self.dpiInput, 1, 1, 1, 2)
        self.grid_layout.addWidget(sizeLabel, 2, 0)
        self.grid_layout.addWidget(self.sizeCombo, 2, 1, 1, 2)
        self.grid_layout.addWidget(self._customLabel, 3, 0)
        self.grid_layout.addWidget(self.customWidthInput, 3, 1, 1, 2)
        self.grid_layout.addWidget(folderLabel, 4, 0)
        self.grid_layout.addWidget(self.folderPath, 4, 1)
        self.grid_layout.addWidget(self.browseButton, 4, 2)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getpng)
        self.buttonBox.rejected.connect(self.reject)

        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.grid_layout)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _getpng(self):
        self._fname = os.path.join(self._folder, self.pngname.text())
        self._dpi = int(self.dpiInput.text()) if self.dpiInput.text() else 150
        
        # Recalculate size based on current DPI setting
        self._update_size_from_dpi()
        
        # Check if file already exists
        if os.path.exists(self._fname):
            reply = QMessageBox.question(
                self,
                'File Exists',
                f'The file "{os.path.basename(self._fname)}" already exists.\n\nDo you want to overwrite it?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self._okexit = False
                return
        
        self._okexit = True

    def _setcheck(self):
        if self.pngname.hasAcceptableInput():
            self.okbtn.setEnabled(True)
        else:
            self.okbtn.setEnabled(False)
    
    def _update_dpi(self):
        try:
            self._dpi = int(self.dpiInput.text())
            self._update_size_from_dpi()
        except ValueError:
            self._dpi = 150
    
    def _update_size_from_dpi(self):
        """Calculate pixel dimensions based on physical size and DPI"""
        current_data = self.sizeCombo.currentData()
        if current_data == "1col":
            # 1 column: 8.25 cm = 3.248 inches
            width_inches = 8.25 / 2.54
            self._width = int(width_inches * self._dpi)
        elif current_data == "2col":
            # 2 column: 17.8 cm = 7.008 inches
            width_inches = 17.8 / 2.54
            self._width = int(width_inches * self._dpi)
        elif current_data == "custom":
            try:
                self._width = int(self.customWidthInput.text())
            except ValueError:
                # Default to 1 column at current DPI
                width_inches = 8.25 / 2.54
                self._width = int(width_inches * self._dpi)
        
        self._update_height()
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self._folder)
        if folder:
            self._folder = folder
            self.folderPath.setText(folder)
    
    def _on_size_changed(self, index):
        """Handle size combo box changes"""
        current_data = self.sizeCombo.currentData()
        is_custom = current_data == "custom"
        
        # Show/hide custom width controls
        self._customLabel.setVisible(is_custom)
        self.customWidthInput.setVisible(is_custom)
        
        # Update width and height based on current selection and DPI
        self._update_size_from_dpi()
    
    def _update_custom_size(self):
        """Update size when custom width is edited"""
        if self.sizeCombo.currentData() == "custom":
            try:
                self._width = int(self.customWidthInput.text())
                self._update_height()
            except ValueError:
                pass
    
    def _update_height(self):
        """Calculate height based on width and figure proportions"""
        if self._figure_size and self._width:
            # Calculate height maintaining aspect ratio
            fig_width, fig_height = self._figure_size
            if fig_width > 0:
                aspect_ratio = fig_height / fig_width
                self._height = int(self._width * aspect_ratio)
            else:
                self._height = int(self._width * 0.75)  # Default 4:3 ratio
        else:
            # Default aspect ratio if no figure size is provided
            self._height = int(self._width * 0.75)  # Default 4:3 ratio
    
    def get_size(self):
        """Return the selected width and height in pixels"""
        return self._width, self._height


class SavePngSeriesDialog(QDialog):
    def __init__(self, parent=None, fname="tcd-gif-XXX.png"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        nameLabel = QLabel('template name for png name:', self)
        self.pngname = QLineEdit(self)
        png_valid = QRegularExpressionValidator()
        # regexp = QRegularExpression()
        # FIXME improve the regexp
        pattern = r'.*XXX.*\.png'
        regexp = QRegularExpression(pattern)
        png_valid.setRegularExpression(regexp)
        # png_valid.setLocale(QLocale('English'))
        self.pngname.setValidator(png_valid)
        self.pngname.setText(self._fname)
        self.pngname.textEdited.connect(self._setcheck)

        self.setWindowTitle("Insert PNG files template name")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.hlay = QHBoxLayout()
        self.hlay.addWidget(nameLabel)
        self.hlay.addWidget(self.pngname)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getpng)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.hlay)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _getpng(self):
        self._fname = self.pngname.text()
        self._okexit = True

    def _setcheck(self):
        if self.pngname.hasAcceptableInput():
            self.okbtn.setEnabled(True)
        else:
            self.okbtn.setEnabled(False)
            #self.buttonBox.

class SaveSceneDialog(QDialog):
    def __init__(self, parent=None, fname="vtkscene.json"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        self._folder = os.path.dirname(fname) if os.path.dirname(fname) else "."
        
        # JSON name input
        nameLabel = QLabel('Scene JSON name:', self)
        self.jsoname = QLineEdit(self)
        json_valid = QRegularExpressionValidator()
        # regexp = QRegularExpression()
        # FIXME improve the regexp
        pattern = r'.*\.json'
        regexp = QRegularExpression(pattern)
        json_valid.setRegularExpression(regexp)
        self.jsoname.setValidator(json_valid)
        self.jsoname.setText(os.path.basename(self._fname))
        self.jsoname.textEdited.connect(self._setcheck)
        
        # Folder selection
        folderLabel = QLabel('Save in:', self)
        self.folderPath = QLineEdit(self)
        self.folderPath.setText(self._folder)
        self.folderPath.setReadOnly(True)
        self.browseButton = QPushButton('Browse...', self)
        self.browseButton.clicked.connect(self._browse_folder)

        self.setWindowTitle("Save JSON Scene file")

        # Layout setup
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(nameLabel, 0, 0)
        self.grid_layout.addWidget(self.jsoname, 0, 1, 1, 2)
        self.grid_layout.addWidget(folderLabel, 1, 0)
        self.grid_layout.addWidget(self.folderPath, 1, 1)
        self.grid_layout.addWidget(self.browseButton, 1, 2)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getjson)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.grid_layout)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _getjson(self):
        self._fname = os.path.join(self._folder, self.jsoname.text())
        
        # Check if file already exists
        if os.path.exists(self._fname):
            reply = QMessageBox.question(
                self,
                'File Exists',
                f'The file "{os.path.basename(self._fname)}" already exists.\n\nDo you want to overwrite it?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self._okexit = False
                return
        
        self._okexit = True

    def _setcheck(self):
        if self.jsoname.hasAcceptableInput():
            self.okbtn.setEnabled(True)
        else:
            self.okbtn.setEnabled(False)
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self._folder)
        if folder:
            self._folder = folder
            self.folderPath.setText(folder)


class ExportPOVDialog(QDialog):
    """Dialog for exporting the current VTK scene to a POV-Ray file via vtkPOVExporter."""

    def __init__(self, parent=None, fname="scene.pov"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        self._folder = os.path.dirname(fname) if os.path.dirname(fname) else "."

        self.setWindowTitle("Export POV-Ray Scene")

        nameLabel = QLabel("POV-Ray filename:", self)
        self.povname = QLineEdit(self)
        pov_valid = QRegularExpressionValidator(
            QRegularExpression(r'.*\.pov'), self)
        self.povname.setValidator(pov_valid)
        self.povname.setText(os.path.basename(self._fname))
        self.povname.textEdited.connect(self._setcheck)

        folderLabel = QLabel("Save in:", self)
        self.folderPath = QLineEdit(self)
        self.folderPath.setText(self._folder)
        self.folderPath.setReadOnly(True)
        self.browseButton = QPushButton("Browse...", self)
        self.browseButton.clicked.connect(self._browse_folder)

        grid = QGridLayout()
        grid.addWidget(nameLabel, 0, 0)
        grid.addWidget(self.povname, 0, 1, 1, 2)
        grid.addWidget(folderLabel, 1, 0)
        grid.addWidget(self.folderPath, 1, 1)
        grid.addWidget(self.browseButton, 1, 2)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.okbtn = self.buttonBox.button(QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._getfname)
        self.buttonBox.rejected.connect(self.reject)

        vlay = QVBoxLayout()
        vlay.addLayout(grid)
        vlay.addWidget(self.buttonBox)
        self.setLayout(vlay)

    def _getfname(self):
        self._fname = os.path.join(self._folder, self.povname.text())
        if os.path.exists(self._fname):
            reply = QMessageBox.question(
                self,
                "File Exists",
                f'The file "{os.path.basename(self._fname)}" already exists.\n\nOverwrite?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self._okexit = False
                return
        self._okexit = True

    def _setcheck(self):
        self.okbtn.setEnabled(self.povname.hasAcceptableInput())

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self._folder)
        if folder:
            self._folder = folder
            self.folderPath.setText(folder)

    def export(self, render_window) -> bool:
        """Run vtkPOVExporter on *render_window* using the confirmed filename.

        Returns True on success, False if the dialog was cancelled or the
        export failed.
        """
        if not self._okexit:
            return False
        try:
            from vtkmodules.vtkIOExport import vtkPOVExporter
        except ImportError:
            QMessageBox.critical(self, "Missing module",
                                 "vtkmodules.vtkIOExport is not available in this VTK build.")
            return False
        exporter = vtkPOVExporter()
        exporter.SetRenderWindow(render_window)
        exporter.SetFileName(self._fname)
        exporter.Write()
        return True


class NMConfigDialog(QDialog):
    def __init__(self, parent=None, invert_phase=False, scale_factor=1.0, color=(1.0, 0.0, 0.0)):
        super().__init__(parent)
        
        self._invert_phase = invert_phase
        self._scale_factor = scale_factor
        self._color = color
        self._okexit = False
        
        self.setWindowTitle("Normal Mode Configuration")
        
        # Phase inversion checkbox
        self.phaseInvert = QCheckBox('Invert Phase', self)
        self.phaseInvert.setChecked(self._invert_phase)
        
        # Scale factor input
        scaleLabel = QLabel('Scale Factor:', self)
        self.scaleInput = QLineEdit(self)
        scale_validator = QDoubleValidator(0.1, 10.0, 2, self)  # Range 0.1-10.0, 2 decimals
        self.scaleInput.setValidator(scale_validator)
        self.scaleInput.setText(str(self._scale_factor))
        
        # Color selection
        colorLabel = QLabel('Vector Color:', self)
        self.colorButton = QPushButton('Select Color', self)
        self.colorButton.clicked.connect(self._select_color)
        self._update_color_button()
        
        # Layout setup
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.phaseInvert, 0, 0, 1, 2)
        self.grid_layout.addWidget(scaleLabel, 1, 0)
        self.grid_layout.addWidget(self.scaleInput, 1, 1)
        self.grid_layout.addWidget(colorLabel, 2, 0)
        self.grid_layout.addWidget(self.colorButton, 2, 1)
        
        # Dialog buttons
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._get_config)
        self.buttonBox.rejected.connect(self.reject)
        
        # Main layout
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.grid_layout)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)
    
    def _select_color(self):
        """Open color dialog to select vector color"""
        # Convert RGB float tuple to QColor
        from PySide6.QtGui import QColor
        current_color = QColor()
        current_color.setRgbF(self._color[0], self._color[1], self._color[2])
        
        color = QColorDialog.getColor(current_color, self, "Select Vector Color")
        if color.isValid():
            new_color = (color.redF(), color.greenF(), color.blueF())
            self._color = new_color
            self._update_color_button()
    
    def _update_color_button(self):
        """Update the color button to show the selected color"""
        r, g, b = [int(c * 255) for c in self._color]
        self.colorButton.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); color: white;")
    
    def _get_config(self):
        """Get configuration values when OK is pressed"""
        self._invert_phase = self.phaseInvert.isChecked()
        try:
            self._scale_factor = float(self.scaleInput.text())
        except ValueError:
            self._scale_factor = 1.0
        # Note: _color is already updated in _select_color method when user changes color
        self._okexit = True


class CubeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Cube Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._fname = ""
        self._label = ""
        self._val = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel("Cube label:")
        self.nmline = QLineEdit()
        self.nmline.setText("Cube1")
        self.nmline.editingFinished.connect(self._setval)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.nmline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setval)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setval(self):
        self._label = str(self.nmline.text())


class TCDDialog(QDialog):
    def __init__(self, parent=None, maxval=1, texts: tp.List[str] = ["VTCD", "NM"]):
        super().__init__(parent)

        self.setWindowTitle(f"{texts[0]} Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._maxval = maxval
        self._fname = ""
        self._val = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel(f"{texts[1]} number:")
        self.nmline = QLineEdit()
        nmval = QIntValidator(1, self._maxval)
        nmval.setLocale(QLocale('English'))
        self.nmline.setValidator(nmval)
        self.nmline.setText("1")
        self.nmline.editingFinished.connect(self._setval)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.nmline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setval)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setval(self):
        self._vib = int(self.nmline.text())


class VTCDDialog(QDialog):
    def __init__(self, parent=None, nvib=1):
        super().__init__(parent)

        self.setWindowTitle("VTCD Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._nvib = nvib
        self._fname = ""
        self._vib = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel("NM number:")
        self.nmline = QLineEdit()
        nmval = QIntValidator(1, self._nvib)
        nmval.setLocale(QLocale('English'))
        self.nmline.setValidator(nmval)
        self.nmline.setText("1")
        self.nmline.editingFinished.connect(self._setvib)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.nmline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvib)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setvib(self):
        self._vib = int(self.nmline.text())



class ETCDDialog(QDialog):
    def __init__(self, parent=None, nstate=1):
        super().__init__(parent)

        self.setWindowTitle("ETCD Loader")

        # widget =  QWidget(self)
        # self.setCentralWidget(widget)
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        self._nstate = nstate
        self._fname = ""
        self._state = None
        self._cube = None
        self._cubefname = "No file"

        message = QLabel("Transition number:")
        self.stateline = QLineEdit()
        stval = QIntValidator(1, self._nstate)
        stval.setLocale(QLocale('English'))
        self.stateline.setValidator(stval)
        self.stateline.setText("1")
        self.stateline.editingFinished.connect(self._setstate)

        openbutton = QPushButton('OpenCube', self)
        openbutton.clicked.connect(self.open)
 
        self.editline = QLineEdit()
        self.editline.setText("{}".format(self._cubefname))
        self.editline.setEnabled(False)
        # grid.addWidget(message)
        grid.addWidget(message, 0, 0)
        grid.addWidget(self.stateline, 0, 1)
        grid.addWidget(openbutton, 0, 2)
        grid.addWidget(self.editline, 0, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setstate)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def open(self):
        self._cube = QFileDialog.getOpenFileName(self, 'Select cube file', '.','*.cube')[0]
        self._cubefname = os.path.basename(self._cube)
        self.editline.setText("{}".format(self._cubefname))
 
    def _setstate(self):
        self._state = int(self.stateline.text())

class EditDoubleLine():
    """
    Edit a double value in a dialog
    return a QHBox layout with a QLabel and a QLineEdit
    """
    def __init__(self, text: str, default: float, validator: QDoubleValidator) -> None:
        self._hlay = QHBoxLayout()
        message = QLabel(text)
        self._hlay.addWidget(message)
        self._line = QLineEdit()
        self._line.setValidator(validator)
        self._line.setText(f"{default:12.6E}")
        self._hlay.addWidget(self._line)
        self._line.editingFinished.connect(self._setedit)
        self._edit = False
        self._default = default

    def _getvalue(self) -> float:
        return float(self._line.text())

    def _setedit(self):
        if not fuzzy_equal(self._getvalue(), self._default, tol=1E-10):
            self._edit = True

    @property
    def edit(self):
        return self._edit


class EditIntLine():
    """
    Edit a integer value in a dialog
    return a QHBox layout with a QLabel and a QLineEdit
    """
    def __init__(self, text: str, default: int, validator: QIntValidator) -> None:
        self._hlay = QHBoxLayout()
        message = QLabel(text)
        self._hlay.addWidget(message)
        self._line = QLineEdit()
        self._line.editingFinished.connect(self._setedit)
        self._line.setValidator(validator)
        self._line.setText(f"{default}")
        self._hlay.addWidget(self._line)
        self._edit = False
        self._default = default

    def _getvalue(self) -> int:
        return int(self._line.text())

    def _setedit(self):
        if self._getvalue() != self._default:
            self._edit = True

    @property
    def edit(self):
        return self._edit

class QuiverSetupDialog(QDialog, _ClipPlaneMixin):
    """Dialog for setting up quiver plot parameters

    Args:
        QDialog (_type_): _description_
    """

    def __init__(self,
                 scale: float,
                 subsamp: int,
                 lower: float = 0.0001,
                 upper: float = 0.01,
                 enable_clipping: bool = False,
                 clip_bounds: tp.Optional[tp.Dict[str, tp.Optional[float]]] = None,
                 show_rotor: bool = False,
                 vtk_renderer=None,
                 vtk_render_window=None,
                 vtk_scene_bounds: tp.Optional[tp.Tuple[float, ...]] = None,
                 parent: tp.Optional[tp.Union[QDialog, None]] = None) -> None:
        """ initialize the dialog. Requires a dictionary with the parameters,
            the maximum norm value in the field and optionally a parent dialog

        Args:
            scale (float): scaling factor for arrows
            subsamp (int): Subsampling of the grid for arrows
            lower (float): lower bound to filter arrows (vectors below this norm are hidden)
            upper (float): upper bound to filter arrows (vectors above this norm are clamped)
            enable_clipping (bool): whether spatial clipping is enabled
            clip_bounds (tp.Optional[tp.Dict[str, tp.Optional[float]]]): clipping bounds
            parent (tp.Optional[tp.Union[QDialog, None]]): Not required
        """
        super().__init__(parent)
        self._scale = scale
        self._subsamp = subsamp
        self._enable_clipping = enable_clipping
        self._clip_bounds = clip_bounds or {}
        self._show_rotor = show_rotor
        self._init_clip_planes(vtk_renderer, vtk_render_window, vtk_scene_bounds)
        self._lower = lower
        self._upper = upper
        self.setWindowTitle("Quiver Setup Dialog")
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)

        message = QLabel("Scaling factor for arrows:")
        self._scalemol = EditDoubleLine("Scaling factor", scale, QDoubleValidator(0.1, 10.0, 2))
        grid.addWidget(message, 0, 0)
        grid.addLayout(self._scalemol._hlay, 1, 0)

        message = QLabel("Subsamplig")
        self._subsampline = EditIntLine("Subsampling", subsamp, QIntValidator(1, 500))
        grid.addWidget(message, 2, 0)
        grid.addLayout(self._subsampline._hlay, 3, 0)

        _bnd_validator = QDoubleValidator()
        _bnd_validator.setLocale(QLocale.c())
        message = QLabel("Lower bound (hide vectors below this norm):")
        self._lowerline = EditDoubleLine("Lower", lower, _bnd_validator)
        grid.addWidget(message, 4, 0)
        grid.addLayout(self._lowerline._hlay, 5, 0)

        _bnd_validator2 = QDoubleValidator()
        _bnd_validator2.setLocale(QLocale.c())
        message = QLabel("Upper bound (clamp vectors above this norm):")
        self._upperline = EditDoubleLine("Upper", upper, _bnd_validator2)
        grid.addWidget(message, 6, 0)
        grid.addLayout(self._upperline._hlay, 7, 0)

        # Rotor display checkbox
        self._show_rotor_checkbox = QCheckBox("Show Rotor (r x J)")
        self._show_rotor_checkbox.setChecked(show_rotor)
        grid.addWidget(self._show_rotor_checkbox, 8, 0, 1, 2)

        # Spatial clipping section
        self._clipping_checkbox = QCheckBox("Enable Spatial Clipping")
        self._clipping_checkbox.setChecked(enable_clipping)
        self._clipping_checkbox.stateChanged.connect(self._toggle_clipping)
        grid.addWidget(self._clipping_checkbox, 9, 0, 1, 2)
        
        # Create clipping controls group (initially hidden)
        self._clipping_frame = QFrame()
        self._clipping_frame.setFrameStyle(QFrame.StyledPanel)
        clipping_layout = QGridLayout(self._clipping_frame)
        
        # X bounds
        x_label = QLabel("X Range (Bohr):")
        clipping_layout.addWidget(x_label, 0, 0)
        
        self._xmin_input = QLineEdit()
        self._xmin_input.setPlaceholderText("xmin (leave empty for no limit)")
        self._xmin_input.setText(str(self._clip_bounds.get('xmin', '')) if self._clip_bounds.get('xmin') is not None else '')
        clipping_layout.addWidget(self._xmin_input, 0, 1)
        
        self._xmax_input = QLineEdit()
        self._xmax_input.setPlaceholderText("xmax (leave empty for no limit)")
        self._xmax_input.setText(str(self._clip_bounds.get('xmax', '')) if self._clip_bounds.get('xmax') is not None else '')
        clipping_layout.addWidget(self._xmax_input, 0, 2)
        
        # Y bounds
        y_label = QLabel("Y Range (Bohr):")
        clipping_layout.addWidget(y_label, 1, 0)
        
        self._ymin_input = QLineEdit()
        self._ymin_input.setPlaceholderText("ymin (leave empty for no limit)")
        self._ymin_input.setText(str(self._clip_bounds.get('ymin', '')) if self._clip_bounds.get('ymin') is not None else '')
        clipping_layout.addWidget(self._ymin_input, 1, 1)
        
        self._ymax_input = QLineEdit()
        self._ymax_input.setPlaceholderText("ymax (leave empty for no limit)")
        self._ymax_input.setText(str(self._clip_bounds.get('ymax', '')) if self._clip_bounds.get('ymax') is not None else '')
        clipping_layout.addWidget(self._ymax_input, 1, 2)
        
        # Z bounds
        z_label = QLabel("Z Range (Bohr):")
        clipping_layout.addWidget(z_label, 2, 0)
        
        self._zmin_input = QLineEdit()
        self._zmin_input.setPlaceholderText("zmin (leave empty for no limit)")
        self._zmin_input.setText(str(self._clip_bounds.get('zmin', '')) if self._clip_bounds.get('zmin') is not None else '')
        clipping_layout.addWidget(self._zmin_input, 2, 1)
        
        self._zmax_input = QLineEdit()
        self._zmax_input.setPlaceholderText("zmax (leave empty for no limit)")
        self._zmax_input.setText(str(self._clip_bounds.get('zmax', '')) if self._clip_bounds.get('zmax') is not None else '')
        clipping_layout.addWidget(self._zmax_input, 2, 2)
        
        # Add double validators for coordinate inputs
        coord_validator = QDoubleValidator()
        coord_validator.setLocale(QLocale('English'))
        for input_field in [self._xmin_input, self._xmax_input, self._ymin_input, 
                           self._ymax_input, self._zmin_input, self._zmax_input]:
            input_field.setValidator(coord_validator)
        
        # Add clipping frame to main layout
        self.vlay.addWidget(self._clipping_frame)
        
        # Set initial visibility
        self._clipping_frame.setVisible(enable_clipping)

        # Clip-plane preview checkbox (enabled only when clipping and VTK are available)
        self.vlay.addWidget(self._setup_plane_preview_ui(enable_clipping))

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvals)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _setvals(self):
        self._scale = self._scalemol._getvalue()
        self._subsamp = self._subsampline._getvalue()
        self._lower = self._lowerline._getvalue()
        self._upper = self._upperline._getvalue()
        self._show_rotor = self._show_rotor_checkbox.isChecked()
        
        # Handle spatial clipping bounds
        self._enable_clipping = self._clipping_checkbox.isChecked()
        if self._enable_clipping:
            # Parse coordinate inputs, handling empty strings as None
            def parse_float_or_none(text):
                try:
                    return float(text) if text.strip() else None
                except (ValueError, AttributeError):
                    return None
            
            self._clip_bounds = {
                'xmin': parse_float_or_none(self._xmin_input.text()),
                'xmax': parse_float_or_none(self._xmax_input.text()),
                'ymin': parse_float_or_none(self._ymin_input.text()),
                'ymax': parse_float_or_none(self._ymax_input.text()),
                'zmin': parse_float_or_none(self._zmin_input.text()),
                'zmax': parse_float_or_none(self._zmax_input.text())
            }
        else:
            self._clip_bounds = {}
    
    def _toggle_clipping(self):
        """Toggle visibility of clipping controls"""
        is_enabled = self._clipping_checkbox.isChecked()
        self._clipping_frame.setVisible(is_enabled)
        self._enable_clipping = is_enabled
        self._update_clipping_toggle(is_enabled)



class MoleculeConfigDialog(QDialog):
    def __init__(self, parent=None, wireframe=False, opacity=1.0, bond_radius=0.03, atom_radius_scale=0.03, tubes_mode=False, bond_tollerance=0.23, hide_auto_group=False, has_auto_fragment=False):
        super().__init__(parent)
        
        self._wireframe = wireframe
        self._opacity = opacity
        self._bond_radius = bond_radius
        self._atom_radius_scale = atom_radius_scale
        self._tubes_mode = tubes_mode
        self._bond_tollerance = bond_tollerance
        self._hide_auto_group = hide_auto_group
        self._has_auto_fragment = has_auto_fragment
        self._okexit = False
        
        self.setWindowTitle("Molecule Display Settings")
        
        # Tubes mode checkbox (replaces wireframe checkbox)
        self.tubesCheck = QCheckBox('Tubes Mode (atoms = bond size)', self)
        self.tubesCheck.setChecked(self._tubes_mode)
        self.tubesCheck.stateChanged.connect(self._toggle_tubes_mode)
        
        # Opacity slider and input
        opacityLabel = QLabel('Opacity:', self)
        self.opacitySlider = QSlider(Qt.Horizontal, self)
        self.opacitySlider.setRange(1, 100)  # 1-100 for 0.01-1.0
        self.opacitySlider.setValue(int(self._opacity * 100))
        self.opacitySlider.valueChanged.connect(self._update_opacity_input)
        
        self.opacityInput = QLineEdit(self)
        opacity_validator = QDoubleValidator()
        opacity_validator.setLocale(QLocale.c())
        opacity_validator.setRange(0.01, 1.0, 2)
        self.opacityInput.setValidator(opacity_validator)
        self.opacityInput.setText(str(self._opacity))
        self.opacityInput.editingFinished.connect(self._update_opacity_slider)
        
        # Bond radius input
        bondRadiusLabel = QLabel('Bond Radius:', self)
        self.bondRadiusInput = QLineEdit(self)
        bond_validator = QDoubleValidator()
        bond_validator.setLocale(QLocale.c())
        bond_validator.setRange(0.001, 3.0, 3)
        self.bondRadiusInput.setValidator(bond_validator)
        self.bondRadiusInput.setText(str(self._bond_radius))
        self.bondRadiusInput.textChanged.connect(self._sync_tubes_mode)
        
        # Atom radius scale input
        atomRadiusLabel = QLabel('Atom Radius Scale:', self)
        self.atomRadiusInput = QLineEdit(self)
        atom_validator = QDoubleValidator()
        atom_validator.setLocale(QLocale.c())
        atom_validator.setRange(0.001, 3.0, 3)
        self.atomRadiusInput.setValidator(atom_validator)
        self.atomRadiusInput.setText(str(self._atom_radius_scale))
        
        # Bond tolerance input
        bondToleranceLabel = QLabel('Bond Tolerance:', self)
        self.bondToleranceInput = QLineEdit(self)
        tolerance_validator = QDoubleValidator()
        tolerance_validator.setLocale(QLocale.c())
        tolerance_validator.setRange(0.1, 2.0, 2)
        self.bondToleranceInput.setValidator(tolerance_validator)
        self.bondToleranceInput.setText(str(self._bond_tollerance))
        
        # Wireframe preset button
        self.wireframeButton = QPushButton('Wireframe Preset (0.01)', self)
        self.wireframeButton.clicked.connect(self._apply_wireframe_preset)
        
        # Store controls that should be disabled in tubes mode
        self.atom_controls = [atomRadiusLabel, self.atomRadiusInput]
        
        # Layout setup
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.tubesCheck, 0, 0, 1, 2)
        self.grid_layout.addWidget(opacityLabel, 1, 0)
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacitySlider)
        opacity_layout.addWidget(self.opacityInput)
        self.grid_layout.addLayout(opacity_layout, 1, 1)
        
        self.grid_layout.addWidget(bondRadiusLabel, 2, 0)
        self.grid_layout.addWidget(self.bondRadiusInput, 2, 1)
        self.grid_layout.addWidget(atomRadiusLabel, 3, 0)
        self.grid_layout.addWidget(self.atomRadiusInput, 3, 1)
        self.grid_layout.addWidget(bondToleranceLabel, 4, 0)
        self.grid_layout.addWidget(self.bondToleranceInput, 4, 1)
        
        # Hide auto group checkbox
        self.hideAutoGroupCheck = QCheckBox('Hide auto group', self)
        self.hideAutoGroupCheck.setChecked(self._hide_auto_group)
        self.hideAutoGroupCheck.setEnabled(self._has_auto_fragment)
        self.grid_layout.addWidget(self.hideAutoGroupCheck, 5, 0, 1, 2)
        
        self.grid_layout.addWidget(self.wireframeButton, 6, 0, 1, 2)
        
        # Dialog buttons
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._get_config)
        self.buttonBox.rejected.connect(self.reject)
        
        # Main layout
        self.vlay = QVBoxLayout()
        self.vlay.addLayout(self.grid_layout)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)
        
        # Initialize tubes mode state
        self._toggle_tubes_mode()
    
    def _toggle_tubes_mode(self):
        """Enable/disable atom controls based on tubes checkbox and sync values"""
        tubes_enabled = self.tubesCheck.isChecked()
        self._tubes_mode = tubes_enabled
        
        # Enable/disable atom controls
        for control in self.atom_controls:
            control.setEnabled(not tubes_enabled)
        
        # If tubes mode is enabled, sync atom radius with bond radius
        if tubes_enabled:
            self._sync_tubes_mode()
    
    def _sync_tubes_mode(self):
        """Sync atom radius with bond radius when in tubes mode"""
        if self._tubes_mode and self.tubesCheck.isChecked():
            bond_radius_text = self.bondRadiusInput.text()
            # Only sync if there's valid text and it's not empty
            if bond_radius_text and bond_radius_text.strip():
                try:
                    # Validate that it's a valid number
                    float(bond_radius_text)
                    self.atomRadiusInput.setText(bond_radius_text)
                except ValueError:
                    # If invalid number, don't update atom radius
                    pass
    
    def _apply_wireframe_preset(self):
        """Apply wireframe preset values (0.01 for both bond and atom)"""
        self.bondRadiusInput.setText("0.01")
        self.atomRadiusInput.setText("0.01")
        # If tubes mode is on, keep them synced
        if self.tubesCheck.isChecked():
            self._sync_tubes_mode()
    
    def _update_opacity_input(self):
        """Update opacity input when slider changes"""
        value = self.opacitySlider.value() / 100.0
        self.opacityInput.setText(f"{value:.2f}")
    
    def _update_opacity_slider(self):
        """Update opacity slider when input changes"""
        try:
            value = float(self.opacityInput.text())
            self.opacitySlider.setValue(int(value * 100))
        except ValueError:
            pass
    
    def _get_config(self):
        """Get configuration values when OK is pressed"""
        self._wireframe = False  # No longer used, keeping for compatibility
        self._tubes_mode = self.tubesCheck.isChecked()
        
        try:
            self._opacity = float(self.opacityInput.text())
        except ValueError:
            self._opacity = 1.0
        
        try:
            self._bond_radius = float(self.bondRadiusInput.text())
        except ValueError:
            self._bond_radius = 0.03
            
        try:
            self._atom_radius_scale = float(self.atomRadiusInput.text())
        except ValueError:
            self._atom_radius_scale = 0.03
            
        try:
            self._bond_tollerance = float(self.bondToleranceInput.text())
        except ValueError:
            self._bond_tollerance = 0.23
            
        self._hide_auto_group = self.hideAutoGroupCheck.isChecked()
            
        self._okexit = True


class StreamLineSetupDialog(QDialog, _ClipPlaneMixin):
    """Dialog for setting up stream lines with spatial clipping options

    Args:
        QDialog (_type_): _description_
    """

    def __init__(self,
                 vfmax: float,
                 vfmin: float,
                 mspeed: float,
                 maxval: float,
                 nseeds: int,
                 scale: float,
                 direction: bool,
                 showellipse: bool,
                 showbar: bool = False,
                 animate_particles: bool = False,
                 num_particles: int = 15,
                 particle_type: str = "sphere",
                 sampling_method: str = "ellipsoid",
                 scalevdw: float = 1.0,
                 enable_clipping: bool = False,
                 clip_bounds: tp.Optional[tp.Dict[str, tp.Optional[float]]] = None,
                 vtk_renderer=None,
                 vtk_render_window=None,
                 vtk_scene_bounds: tp.Optional[tp.Tuple[float, ...]] = None,
                 parent: tp.Optional[tp.Union[QDialog, None]] = None,
                 ) -> None:
        """ initialize the dialog. Requires a dictionary with the parameters,
            the maximum norm value in the field and optionally a parent dialog

        Args:
            params (tp.Dict[str, tp.Any]): dictionary with the parameters
            maxval (float): maximum norm value in the field
            enable_clipping (bool): whether spatial clipping is enabled
            clip_bounds (tp.Optional[tp.Dict[str, tp.Optional[float]]]): clipping bounds
            parent (tp.Optional[tp.Union[QDialog, None]]): Not required
        """
        super().__init__(parent)
        self._vfmax = vfmax
        self._vfmin = vfmin
        self._mspeed = mspeed
        self._nseeds = nseeds
        self._scale = scale
        self._direction = direction
        self._showellipse = showellipse
        self._showbar = showbar
        self._animate_particles = animate_particles
        self._num_particles = num_particles
        self._particle_type = particle_type
        self._sampling_method = sampling_method
        self._scalevdw = scalevdw
        self._enable_clipping = enable_clipping
        self._clip_bounds = clip_bounds or {}
        self._init_clip_planes(vtk_renderer, vtk_render_window, vtk_scene_bounds)
        self._recalseeds = False
        self._redrawstream = False
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")
        self.setWindowTitle("Stream Line Setup Dialog")
        self.vlay = QVBoxLayout()
        message = QLabel(f"Max norm value:{maxval:12.6E}")
        self.vlay.addWidget(message)
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        message = QLabel("""Bound values as fraction
 of the max val:""")
        grid.addWidget(message,0, 0)
        message = QLabel("Upper denominator:")
        self.isoline = QLineEdit()
        maxvalid = QDoubleValidator()
        # validator for upper bound
        maxvalid.setRange(1, 1E12) # FIXME this shit
        maxvalid.setLocale(QLocale('English'))
        self._upmes = EditDoubleLine("Upper Bound", vfmax, maxvalid)
        grid.addLayout(self._upmes._hlay, 1, 0)
        minvalid = QDoubleValidator()
        minvalid.setLocale(QLocale('English'))
        minvalid.setRange(self._upmes._getvalue(),1E12)
        self._domes = EditDoubleLine("Lower bound", vfmin, minvalid)
        grid.addLayout(self._domes._hlay, 2, 0)
        self._shdir = QCheckBox("Show direction")
        self._shdir.setChecked(direction)
        self._shdir.stateChanged.connect(self._setdir)
        self._shell = QCheckBox("Show seeds")
        self._shell.setChecked(showellipse)
        self._shell.stateChanged.connect(self._setell)
        self._shbar = QCheckBox("Show ColorBar")
        self._shbar.setChecked(showbar)
        self._shbar.stateChanged.connect(self._setbar)
        self._animate = QCheckBox("Animate Particles")
        self._animate.setChecked(animate_particles)
        self._animate.stateChanged.connect(self._setanimate)
        grid.addWidget(self._shdir, 3, 0)
        grid.addWidget(self._shell, 4, 0)
        grid.addWidget(self._shbar, 5, 0)
        grid.addWidget(self._animate, 6, 0)

        # Second column
        message = QLabel("""Minimum speed for streamlines integration""")
        grid.addWidget(message,0 , 1)
        spdvalid = QDoubleValidator()
        spdvalid.setLocale(QLocale('English'))
        spdvalid.setRange(1E-12, maxval)
        self._spmes = EditDoubleLine("Minimum speed", mspeed, spdvalid)
        grid.addItem(self._spmes._hlay, 1, 1)
        seedvalid = QIntValidator()
        seedvalid.setLocale(QLocale('English'))
        # BUG limit hardcoded
        seedvalid.setRange(1, 500)
        self._seedline = EditIntLine("Number of seeds", nseeds, seedvalid)
        grid.addItem(self._seedline._hlay, 2, 1)
        # Spatial clipping section
        self._clipping_checkbox = QCheckBox("Enable Spatial Clipping")
        self._clipping_checkbox.setChecked(enable_clipping)
        self._clipping_checkbox.stateChanged.connect(self._toggle_clipping)
        grid.addWidget(self._clipping_checkbox, 7, 0, 1, 2)
        
        # Create clipping controls group (initially hidden)
        self._clipping_frame = QFrame()
        self._clipping_frame.setFrameStyle(QFrame.StyledPanel)
        clipping_layout = QGridLayout(self._clipping_frame)
        
        # X bounds
        x_label = QLabel("X Range (Bohr):")
        clipping_layout.addWidget(x_label, 0, 0)
        
        self._xmin_input = QLineEdit()
        self._xmin_input.setPlaceholderText("xmin (leave empty for no limit)")
        self._xmin_input.setText(str(self._clip_bounds.get('xmin', '')) if self._clip_bounds.get('xmin') is not None else '')
        clipping_layout.addWidget(self._xmin_input, 0, 1)
        
        self._xmax_input = QLineEdit()
        self._xmax_input.setPlaceholderText("xmax (leave empty for no limit)")
        self._xmax_input.setText(str(self._clip_bounds.get('xmax', '')) if self._clip_bounds.get('xmax') is not None else '')
        clipping_layout.addWidget(self._xmax_input, 0, 2)
        
        # Y bounds
        y_label = QLabel("Y Range (Bohr):")
        clipping_layout.addWidget(y_label, 1, 0)
        
        self._ymin_input = QLineEdit()
        self._ymin_input.setPlaceholderText("ymin (leave empty for no limit)")
        self._ymin_input.setText(str(self._clip_bounds.get('ymin', '')) if self._clip_bounds.get('ymin') is not None else '')
        clipping_layout.addWidget(self._ymin_input, 1, 1)
        
        self._ymax_input = QLineEdit()
        self._ymax_input.setPlaceholderText("ymax (leave empty for no limit)")
        self._ymax_input.setText(str(self._clip_bounds.get('ymax', '')) if self._clip_bounds.get('ymax') is not None else '')
        clipping_layout.addWidget(self._ymax_input, 1, 2)
        
        # Z bounds
        z_label = QLabel("Z Range (Bohr):")
        clipping_layout.addWidget(z_label, 2, 0)
        
        self._zmin_input = QLineEdit()
        self._zmin_input.setPlaceholderText("zmin (leave empty for no limit)")
        self._zmin_input.setText(str(self._clip_bounds.get('zmin', '')) if self._clip_bounds.get('zmin') is not None else '')
        clipping_layout.addWidget(self._zmin_input, 2, 1)
        
        self._zmax_input = QLineEdit()
        self._zmax_input.setPlaceholderText("zmax (leave empty for no limit)")
        self._zmax_input.setText(str(self._clip_bounds.get('zmax', '')) if self._clip_bounds.get('zmax') is not None else '')
        clipping_layout.addWidget(self._zmax_input, 2, 2)
        
        # Add double validators for coordinate inputs
        coord_validator = QDoubleValidator()
        coord_validator.setLocale(QLocale('English'))
        for input_field in [self._xmin_input, self._xmax_input, self._ymin_input, 
                           self._ymax_input, self._zmin_input, self._zmax_input]:
            input_field.setValidator(coord_validator)
        
        # Add clipping frame to main layout
        self.vlay.addWidget(self._clipping_frame)
        
        # Set initial visibility
        self._clipping_frame.setVisible(enable_clipping)
        
        # Clip-plane preview checkbox (enabled only when clipping and VTK are available)
        self.vlay.addWidget(self._setup_plane_preview_ui(enable_clipping))

        scaleval = QDoubleValidator()
        scaleval.setLocale(QLocale('English'))
        scaleval.setRange(.2, 10.)
        self._scalemol = EditDoubleLine("Ellipsoid scaling factor", scale, scaleval)
        grid.addItem(self._scalemol._hlay, 3, 1)
        
        # Sampling method selection
        sampling_group_label = QLabel("Sampling method:")
        grid.addWidget(sampling_group_label, 4, 1)
        
        self._sampling_group = QButtonGroup()
        self._ellipsoid_radio = QRadioButton("Ellipsoid")
        self._molvolume_radio = QRadioButton("Mol. volume")
        
        if sampling_method == "ellipsoid":
            self._ellipsoid_radio.setChecked(True)
        else:
            self._molvolume_radio.setChecked(True)
            
        self._sampling_group.addButton(self._ellipsoid_radio, 0)
        self._sampling_group.addButton(self._molvolume_radio, 1)
        self._ellipsoid_radio.toggled.connect(self._update_sampling_method)
        
        sampling_hlay = QHBoxLayout()
        sampling_hlay.addWidget(self._ellipsoid_radio)
        sampling_hlay.addWidget(self._molvolume_radio)
        grid.addLayout(sampling_hlay, 5, 1)
        
        # VDW scaling factor
        scalevdwval = QDoubleValidator()
        scalevdwval.setLocale(QLocale('English'))
        scalevdwval.setRange(.2, 5.)
        self._scalevdw_edit = EditDoubleLine("VDW radius scaling", scalevdw, scalevdwval)
        grid.addItem(self._scalevdw_edit._hlay, 6, 1)
        particlevalid = QIntValidator()
        particlevalid.setLocale(QLocale('English'))
        particlevalid.setRange(1, 50)  # Reasonable range for particles
        self._particleline = EditIntLine("Number of particles", num_particles, particlevalid)
        grid.addItem(self._particleline._hlay, 7, 1)  # Move down to accommodate new controls
        # Set initial state of particle count field based on animation checkbox
        self._particleline._line.setEnabled(animate_particles)
        
        # Add particle type selection
        particle_type_label = QLabel("Particle type:")
        self._particle_type_combo = QComboBox()
        self._particle_type_combo.addItems(["sphere", "cone"])
        self._particle_type_combo.setCurrentText(particle_type)
        self._particle_type_combo.setEnabled(animate_particles)
        hlay_particle_type = QHBoxLayout()
        hlay_particle_type.addWidget(particle_type_label)
        hlay_particle_type.addWidget(self._particle_type_combo)
        grid.addItem(hlay_particle_type, 8, 1)
        
        self._genseeds = QPushButton('Resample seeds', self)
        self._genseeds.clicked.connect(self._setresample)
        hlay_tmp = QHBoxLayout()
        hlay_tmp.addWidget(self._genseeds)
        grid.addItem(hlay_tmp, 9, 1)

        # Enable/disable controls based on sampling method
        self._update_sampling_method()

        # add text to the dialog
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvals)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

    def _update_sampling_method(self):
        """Update UI controls based on selected sampling method"""
        ellipsoid_selected = self._ellipsoid_radio.isChecked()
        self._scalemol._line.setEnabled(ellipsoid_selected)
        self._scalevdw_edit._line.setEnabled(not ellipsoid_selected)
        
        # Update sampling method
        self._sampling_method = "ellipsoid" if ellipsoid_selected else "molvolume"
        
        # Update resample button text
        if ellipsoid_selected:
            self._genseeds.setText('Resample ellipsoid')
        else:
            self._genseeds.setText('Resample mol. volume')
    
    def _setvals(self):
        self._vfmax = self._upmes._getvalue()
        if self._upmes.edit:
            self._redrawstream = True
        self._vfmin = self._domes._getvalue()
        if self._domes.edit:
            self._redrawstream = True
        self._mspeed = self._spmes._getvalue()
        if self._spmes.edit:
            self._redrawstream = True
        self._nseeds = self._seedline._getvalue()
        if self._seedline.edit:
            self._recalseeds = True
            self._redrawstream = True
        self._scale = self._scalemol._getvalue()
        if self._scalemol.edit:
            self._recalseeds = True
            self._redrawstream = True
        scalevdw_value = self._scalevdw_edit._getvalue()
        if self._scalevdw_edit.edit:
            self._recalseeds = True
            self._redrawstream = True
        self._scalevdw = scalevdw_value
        self._num_particles = self._particleline._getvalue()
        if self._particleline.edit:
            self._redrawstream = True
        self._particle_type = self._particle_type_combo.currentText()
        self._sampling_method = "ellipsoid" if self._ellipsoid_radio.isChecked() else "molvolume"
        
        # Handle spatial clipping bounds
        self._enable_clipping = self._clipping_checkbox.isChecked()
        if self._enable_clipping:
            # Parse coordinate inputs, handling empty strings as None
            def parse_float_or_none(text):
                try:
                    return float(text) if text.strip() else None
                except (ValueError, AttributeError):
                    return None
            
            self._clip_bounds = {
                'xmin': parse_float_or_none(self._xmin_input.text()),
                'xmax': parse_float_or_none(self._xmax_input.text()),
                'ymin': parse_float_or_none(self._ymin_input.text()),
                'ymax': parse_float_or_none(self._ymax_input.text()),
                'zmin': parse_float_or_none(self._zmin_input.text()),
                'zmax': parse_float_or_none(self._zmax_input.text())
            }
        else:
            self._clip_bounds = {}
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")

    def _setresample(self):
        self._recalseeds = True
        self._redrawstream = True
        self._nseeds = self._seedline._getvalue()
        self._scale = self._scalemol._getvalue()
        # print(f"vfmax:{self._vfmax} vfmin:{self._vfmin} mspeed:{self._mspeed} nseeds:{self._nseeds} scale:{self._scale}")

    def _setdir(self):
        self._direction = self._shdir.isChecked()

    def _setell(self):
        self._showellipse = self._shell.isChecked()

    def _setbar(self):
        self._showbar = self._shbar.isChecked()

    def _setanimate(self):
        self._animate_particles = self._animate.isChecked()
        # Enable/disable particle count field based on animation checkbox
        self._particleline._line.setEnabled(self._animate_particles)
        # Enable/disable particle type combo box based on animation checkbox
        self._particle_type_combo.setEnabled(self._animate_particles)
    
    def _toggle_clipping(self):
        """Toggle visibility of clipping controls"""
        is_enabled = self._clipping_checkbox.isChecked()
        self._clipping_frame.setVisible(is_enabled)
        self._enable_clipping = is_enabled
        self._update_clipping_toggle(is_enabled)

