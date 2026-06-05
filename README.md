# tcdlibx

A Python library for the manipulation and visualization of volumetric vector-field
datasets arising from quantum-chemical calculations, with a focus on
**Transition Current Densities (TCD)** for vibrational and electronic transitions.

## Features

- Read and manipulate `.cube` and `.fchk` (Gaussian) files containing TCD data
- Compute integrated electric and magnetic transition dipole moments from TCD grids
- Fragment-based decomposition of TCD contributions (with AIM basin support)
- Interactive 3-D visualization via a PySide6/VTK GUI (`tcdvis`):
  - Isosurfaces, streamlines, quiver plots and particle animations
  - Molecular geometry overlay
  - Export scenes to PNG and POV-Ray (`.pov`)
- 2-D quiver-plot scripts for quick inspection

## Dependencies

### Required
- [`numpy`](https://numpy.org)
- [`matplotlib`](https://matplotlib.org)
- [`estampes`](https://github.com/mast-theolab/estampes) *(install separately, see below)*

### Optional
- `vtk` — required for the GUI and 3-D visualization
- `PySide6` — required for the GUI
- `numba` — accelerates some numerical routines

## Installation

First install `estampes` from its repository:

```bash
pip install git+https://github.com/mast-theolab/estampes.git
```

Then install `tcdlibx`:

```bash
pip install . -e

```

`numba` can be added independently:

```bash
pip install numba
```

## Command-line tools

| Command | Description |
|---------|-------------|
| `tcdvis` | Interactive 3-D visualization GUI |
| `plot_vtcd2d` | 2-D quiver plot of a vibrational TCD |
| `plot_etcd2d` | 2-D quiver plot of an electronic TCD |
| `calc_vtcd` | Compute vibrational TCD from `.fchk` and `.cube` files |

## Quick start

```bash
# Launch the interactive GUI
tcdvis
```

Once open, use **File → Open** to load a `.fchk` file and then load the
corresponding TCD `.cube` file(s) through the interface.


## Citing

If you use `tcdlibx` in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE) for details.

