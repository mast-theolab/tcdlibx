import os
import typing as tp
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtCore import QRegularExpression, QLocale
from PySide6.QtGui import QRegularExpressionValidator, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QCheckBox, QComboBox
from PySide6.QtWidgets import QLineEdit, QFileDialog, QGridLayout, QPushButton
from tcdlibx.utils.var_tools import fuzzy_equal

class SavePngDialog(QDialog):
    def __init__(self, parent=None, fname="tcdfigure.png"):
        super().__init__(parent)
        self._fname = fname
        self._okexit = False
        nameLabel = QLabel('png name:', self)
        self.pngname = QLineEdit(self)
        png_valid = QRegularExpressionValidator()
        # regexp = QRegularExpression()
        # FIXME improve the regexp
        pattern = r'.*\.png'
        regexp = QRegularExpression(pattern)
        png_valid.setRegularExpression(regexp)
        # png_valid.setLocale(QLocale('English'))
        self.pngname.setValidator(png_valid)
        self.pngname.setText(self._fname)
        self.pngname.textEdited.connect(self._setcheck)

        self.setWindowTitle("Save PNG file")

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


class TensorSetupDialog(QDialog):
    """Dialog for setting up tensor visualization parameters

    Args:
        QDialog: Qt dialog base class
    """

    def __init__(self,
                 sphere_radius: float = 0.5,
                 nb_sphere_samples: int = 50,
                 vector_scale: float = 2.0,
                 opacity: float = 0.8,
                 color_scheme: str = 'weight_mag',
                 atom_filter: tp.Optional[str] = None,
                 show_spheres: bool = False,
                 parent: tp.Optional[tp.Union[QDialog, None]] = None) -> None:
        """Initialize the tensor setup dialog.

        Args:
            sphere_radius: Radius of the visualization spheres
            nb_sphere_samples: Number of fibonacci samples on each sphere
            vector_scale: Scale factor for the vectors
            opacity: Opacity of the visualization
            color_scheme: Color scheme for vectors ('magnitude', 'weight_mag', 'uniform')
            atom_filter: Comma-separated atomic symbols to filter (e.g., "C,N,O")
            show_spheres: Whether to show pale yellow spheres at tensor positions
            parent: Parent dialog (optional)
        """
        super().__init__(parent)
        
        # Store initial values
        self._sphere_radius = sphere_radius
        self._nb_sphere_samples = nb_sphere_samples
        self._vector_scale = vector_scale
        self._opacity = opacity
        self._color_scheme = color_scheme
        self._atom_filter = atom_filter
        self._show_spheres = show_spheres
        
        self.setWindowTitle("Tensor Visualization Setup")
        
        # Main layout
        self.vlay = QVBoxLayout()
        grid = QGridLayout()
        self.vlay.addLayout(grid)
        
        # Sphere radius
        radius_valid = QDoubleValidator()
        radius_valid.setLocale(QLocale('English'))
        radius_valid.setRange(0.1, 5.0)
        self._radius_line = EditDoubleLine("Sphere radius", sphere_radius, radius_valid)
        grid.addLayout(self._radius_line._hlay, 0, 0)
        
        # Number of sphere samples
        samples_valid = QIntValidator()
        samples_valid.setLocale(QLocale('English'))
        samples_valid.setRange(10, 200)
        self._samples_line = EditIntLine("Number of sphere samples", nb_sphere_samples, samples_valid)
        grid.addLayout(self._samples_line._hlay, 1, 0)
        
        # Vector scale
        scale_valid = QDoubleValidator()
        scale_valid.setLocale(QLocale('English'))
        scale_valid.setRange(0.1, 20.0)
        self._scale_line = EditDoubleLine("Vector scale factor", vector_scale, scale_valid)
        grid.addLayout(self._scale_line._hlay, 2, 0)
        
        # Opacity
        opacity_valid = QDoubleValidator()
        opacity_valid.setLocale(QLocale('English'))
        opacity_valid.setRange(0.1, 1.0)
        self._opacity_line = EditDoubleLine("Opacity", opacity, opacity_valid)
        grid.addLayout(self._opacity_line._hlay, 3, 0)
        
        # Color scheme
        color_label = QLabel("Color scheme:")
        self._color_combo = QComboBox()
        self._color_combo.addItems(['magnitude', 'weight_mag', 'uniform'])
        self._color_combo.setCurrentText(color_scheme)
        self._color_combo.currentTextChanged.connect(self._setcolorscheme)
        
        color_hlay = QHBoxLayout()
        color_hlay.addWidget(color_label)
        color_hlay.addWidget(self._color_combo)
        grid.addLayout(color_hlay, 4, 0)
        
        # Atom filter
        filter_label = QLabel("Atom filter (comma-separated):")
        self._filter_line = QLineEdit()
        self._filter_line.setPlaceholderText("e.g., C,N,O (leave empty for all atoms)")
        if atom_filter:
            self._filter_line.setText(atom_filter)
        self._filter_line.textChanged.connect(self._setatomfilter)
        
        filter_hlay = QHBoxLayout()
        filter_hlay.addWidget(filter_label)
        filter_hlay.addWidget(self._filter_line)
        grid.addLayout(filter_hlay, 5, 0)
        
        # Show spheres checkbox
        self._show_spheres_checkbox = QCheckBox("Show pale yellow spheres at tensor positions")
        self._show_spheres_checkbox.setChecked(show_spheres)
        self._show_spheres_checkbox.stateChanged.connect(self._setshowspheres)
        grid.addWidget(self._show_spheres_checkbox, 6, 0)
        
        # Dialog buttons
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvals)
        self.buttonBox.rejected.connect(self.reject)
        
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)
    
    def _setvals(self):
        """Set the values from the dialog controls."""
        self._sphere_radius = self._radius_line._getvalue()
        self._nb_sphere_samples = self._samples_line._getvalue()
        self._vector_scale = self._scale_line._getvalue()
        self._opacity = self._opacity_line._getvalue()
        # color_scheme is already set by the combo box callback
    
    def _setcolorscheme(self):
        """Set the color scheme from the combo box."""
        self._color_scheme = self._color_combo.currentText()
    
    def _setatomfilter(self):
        """Set the atom filter from the text input."""
        self._atom_filter = self._filter_line.text().strip()
        if not self._atom_filter:
            self._atom_filter = None
    
    def _setshowspheres(self):
        """Set the show spheres flag from the checkbox."""
        self._show_spheres = self._show_spheres_checkbox.isChecked()
    
    @property
    def sphere_radius(self) -> float:
        """Get the sphere radius value."""
        return self._sphere_radius
    
    @property
    def nb_sphere_samples(self) -> int:
        """Get the number of sphere samples."""
        return self._nb_sphere_samples
    
    @property
    def vector_scale(self) -> float:
        """Get the vector scale factor."""
        return self._vector_scale
    
    @property
    def opacity(self) -> float:
        """Get the opacity value."""
        return self._opacity
    
    @property
    def color_scheme(self) -> str:
        """Get the color scheme."""
        return self._color_scheme
    
    @property
    def atom_filter(self) -> tp.Optional[tp.List[str]]:
        """Get the atom filter as a list of atomic symbols."""
        if self._atom_filter is None or self._atom_filter == "":
            return None
        # Split by comma and strip whitespace from each symbol
        symbols = [symbol.strip() for symbol in self._atom_filter.split(',')]
        # Filter out empty strings
        symbols = [symbol for symbol in symbols if symbol]
        return symbols if symbols else None
    
    @property
    def show_spheres(self) -> bool:
        """Get the show spheres flag."""
        return self._show_spheres


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
        scaleval = QDoubleValidator()
        scaleval.setLocale(QLocale('English'))
        scaleval.setRange(.2, 10.)
        self._scalemol = EditDoubleLine("Ellipsoid scaling factor", scale, scaleval)
        grid.addItem(self._scalemol._hlay, 3, 1)
        particlevalid = QIntValidator()
        particlevalid.setLocale(QLocale('English'))
        particlevalid.setRange(1, 50)  # Reasonable range for particles
        self._particleline = EditIntLine("Number of particles", num_particles, particlevalid)
        grid.addItem(self._particleline._hlay, 6, 1)  # Align with animation checkbox
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
        grid.addItem(hlay_particle_type, 7, 1)
        
        self._genseeds = QPushButton('Resample the ellissoide', self)
        self._genseeds.clicked.connect(self._setresample)
        hlay_tmp = QHBoxLayout()
        hlay_tmp.addWidget(self._genseeds)
        grid.addItem(hlay_tmp, 4, 1)

        # add text to the dialog
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.accepted.connect(self._setvals)
        self.buttonBox.rejected.connect(self.reject)

        # self.vlay.addWidget(message)
        self.vlay.addWidget(self.buttonBox)
        self.setLayout(self.vlay)

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
        self._num_particles = self._particleline._getvalue()
        if self._particleline.edit:
            self._redrawstream = True
        self._particle_type = self._particle_type_combo.currentText()
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

