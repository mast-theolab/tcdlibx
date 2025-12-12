import os
import typing as tp
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PySide6.QtCore import QRegularExpression, QLocale, Qt
from PySide6.QtGui import QRegularExpressionValidator, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QCheckBox, QComboBox
from PySide6.QtWidgets import QLineEdit, QFileDialog, QPushButton, QMessageBox, QColorDialog, QSlider, QRadioButton, QButtonGroup
from tcdlibx.utils.var_tools import fuzzy_equal

class SavePngDialog(QDialog):
    def __init__(self, parent=None, fname="tcdfigure.png"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        self._dpi = 150  # Default DPI
        self._folder = os.path.dirname(fname) if os.path.dirname(fname) else "."
        
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
        self.grid_layout.addWidget(folderLabel, 2, 0)
        self.grid_layout.addWidget(self.folderPath, 2, 1)
        self.grid_layout.addWidget(self.browseButton, 2, 2)

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
        except ValueError:
            self._dpi = 150
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self._folder)
        if folder:
            self._folder = folder
            self.folderPath.setText(folder)


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

class QuiverSetupDialog(QDialog):
    """Dialog for setting up quiver plot parameters

    Args:
        QDialog (_type_): _description_
    """

    def __init__(self,
                 scale: float,
                 subsamp: int,
                 parent: tp.Optional[tp.Union[QDialog, None]] = None) -> None:
        """ initialize the dialog. Requires a dictionary with the parameters,
            the maximum norm value in the field and optionally a parent dialog

        Args:
            scale (float): scaling factor for arrows
            nseeds (int): Subsampling of the grid for arrows
            parent (tp.Optional[tp.Union[QDialog, None]]): Not required
        """
        super().__init__(parent)
        self._scale = scale
        self._subsamp = subsamp
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


class MoleculeConfigDialog(QDialog):
    def __init__(self, parent=None, wireframe=False, opacity=1.0, bond_radius=0.03, atom_radius_scale=0.03, tubes_mode=False):
        super().__init__(parent)
        
        self._wireframe = wireframe
        self._opacity = opacity
        self._bond_radius = bond_radius
        self._atom_radius_scale = atom_radius_scale
        self._tubes_mode = tubes_mode
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
        self.grid_layout.addWidget(self.wireframeButton, 4, 0, 1, 2)
        
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
            
        self._okexit = True


class StreamLineSetupDialog(QDialog):
    """Dialog for setting up stream lines

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
                 parent: tp.Optional[tp.Union[QDialog, None]] = None,
                 ) -> None:
        """ initialize the dialog. Requires a dictionary with the parameters,
            the maximum norm value in the field and optionally a parent dialog

        Args:
            params (tp.Dict[str, tp.Any]): dictionary with the parameters
            maxval (float): maximum norm value in the field
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
        self._shell = QCheckBox("Show ellipsoids")
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
        # Ellipsoid scaling factor
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
        self._scalevdw = EditDoubleLine("VDW radius scaling", scalevdw, scalevdwval)
        grid.addItem(self._scalevdw._hlay, 6, 1)
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
        self._scalevdw._line.setEnabled(not ellipsoid_selected)
        
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
        scalevdw_value = self._scalevdw._getvalue()
        if self._scalevdw.edit:
            self._recalseeds = True
            self._redrawstream = True
        self._scalevdw = scalevdw_value
        self._num_particles = self._particleline._getvalue()
        if self._particleline.edit:
            self._redrawstream = True
        self._particle_type = self._particle_type_combo.currentText()
        self._sampling_method = "ellipsoid" if self._ellipsoid_radio.isChecked() else "molvolume"
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

