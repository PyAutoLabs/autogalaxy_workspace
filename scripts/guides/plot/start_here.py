"""
Plots: Start Here
=================

This example introduces the PyAutoGalaxy plotting API.

Plotting is performed by standalone functions in the `autogalaxy.plot` module, imported as `aplt`:

 - `aplt.plot_array()` — plot any 2D array (data, galaxy images, residuals, etc.).
 - `aplt.plot_grid()` — plot a 2D grid of (y,x) coordinates.
 - `aplt.subplot_imaging_dataset()`, `aplt.subplot_galaxies()`, etc. — multi-panel subplots for standard objects.

The API follows a simple pattern: quantities are computed from PyAutoGalaxy objects via their
methods (e.g. `galaxy.image_2d_from(grid=grid)`) and the resulting array or grid is passed to a
plotting function. This means anything the library can compute, you can plot, without needing a
dedicated plotting class for every object.

Figure appearance (titles, colormaps, log10 scaling, output to disk) is customized by passing
keyword arguments directly to the plotting functions, with project-wide defaults set via config
files.

__Contents__

- **Dataset:** Load the galaxy dataset and set up the galaxy used throughout this example.
- **plot_array:** The fundamental function for plotting any 2D array.
- **plot_grid:** Plot 2D (y,x) coordinate grids.
- **Customization:** Titles, colormaps, log10 scaling and value limits via keyword arguments.
- **Output:** Save figures to disk in one or more formats, with control over path and filename.
- **Config Defaults:** Project-wide default appearance via `config/visualize/`.
- **Overlays:** Overlay positions and grids using the `positions=` and `grid=` keyword arguments.
- **subplot_* Functions:** Multi-panel overviews of datasets and galaxies.
- **Where To Next:** Object-by-object figures, fit plotting per dataset type and search results.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load an example imaging dataset and set up objects used throughout this example.
"""
dataset_path = Path("dataset") / "imaging" / "simple"

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

data_path = dataset_path / "data.fits"
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

"""
__plot_array__

The fundamental plotting function is `aplt.plot_array()`, which displays any 2D `Array2D`.

We can plot the raw data array loaded from a .fits file.
"""
aplt.plot_array(array=data, title="Data")

"""
We can also plot quantities computed from a galaxy, such as its image.

This is the pattern used throughout PyAutoGalaxy: compute the quantity you want via the object's
method, then pass it to `plot_array`.
"""
aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Galaxy Image")

"""
__plot_grid__

The `aplt.plot_grid()` function displays a 2D grid of (y,x) coordinates, for example the uniform
grid the galaxy's image is computed on.
"""
aplt.plot_grid(grid=grid, title="Uniform Grid")

"""
__Customization__

Each plotting function accepts direct keyword arguments for customization:

 - `title`: The figure title string.
 - `colormap`: The matplotlib colormap name (e.g. "jet", "hot", "gray").
 - `use_log10`: If True, the colormap is plotted in log10 scale.
 - `vmin` / `vmax`: The minimum and maximum values of the colormap scale.

The colormap accepts any valid matplotlib colormap name.
"""
aplt.plot_array(array=data, title="Jet Colormap", colormap="jet")
aplt.plot_array(array=data, title="Hot Colormap", colormap="hot")
aplt.plot_array(array=data, title="Gray Colormap", colormap="gray")

"""
A galaxy's light spans several orders of magnitude and is often easier to interpret in log10
space, which `use_log10=True` provides.
"""
aplt.plot_array(
    array=galaxy.image_2d_from(grid=grid),
    title="Galaxy Image (Log10)",
    use_log10=True,
)

"""
The `vmin` / `vmax` keywords fix the colormap limits, which is useful for comparing figures on the
same scale.
"""
aplt.plot_array(array=data, title="Data (Fixed Scale)", vmin=0.0, vmax=1.0)

"""
__Output__

By default (with no `output_path` input), figures are displayed on screen.

To save a figure to disk instead, pass `output_path` (a directory) and `output_format`. The file
is saved as `{output_path}/{title}.{output_format}` by default; pass `output_filename` to choose
the filename explicitly.
"""
aplt.plot_array(
    array=data,
    title="Image",
    output_path=Path("output") / "plot",
    output_filename="example",
    output_format="png",
)

"""
Multiple formats can be specified as a list to save the same figure in each format at once, for
example a .png for quick inspection alongside a .pdf for publication.
"""
aplt.plot_array(
    array=data,
    title="Image",
    output_path=Path("output") / "plot",
    output_filename="example",
    output_format=["png", "pdf"],
)

"""
__Config Defaults__

When no explicit keyword is passed to a plotting function the default value is read from the
config files in:

  autogalaxy_workspace/config/visualize/

Key entries in `config/visualize/general.yaml` include:

 - `colormap`: The default colormap of all 2D plots.
 - `ticks` -> `number_of_ticks_2d`: The number of ticks on each spatial axis.
 - `colorbar` -> `labelsize` / `labelsize_subplot`: The font size of colorbar tick labels.
 - `contour` -> `total_contours`: The contour levels drawn over log10 plots.
 - `units` -> `cb_unit`: The unit label of the colorbar.
 - `subplot_shape_to_figsize_factor`: The scaling of subplot figure sizes.

This allows the default appearance to be controlled project-wide without changing code. To change
these defaults, edit the YAML config file and restart the Python session (or Jupyter kernel).

The separate `config/visualize/plots.yaml` file controls which figures are output automatically
during a model-fit — see the `__Visualizer__` documentation at the end of `scripts/imaging/plot.py`.

__Overlays__

Overlays are added to plots using keyword arguments, for example `positions=` (scatter points,
e.g. light-profile centres) and `grid=` (a grid of coordinates drawn over the figure).
"""
positions = ag.Grid2DIrregular(values=[(0.0, 0.0)])

aplt.plot_array(
    array=galaxy.image_2d_from(grid=grid),
    title="Galaxy Image with Centre",
    positions=positions,
)

"""
The full range of overlays is documented in `scripts/guides/plot/visuals.py`.

__subplot_* Functions__

For standard objects (datasets, galaxies), dedicated subplot functions produce multi-panel
overviews automatically.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

galaxies = ag.Galaxies(galaxies=[galaxy])

aplt.subplot_galaxies(galaxies=galaxies, grid=grid)

"""
__Where To Next__

- **Object-by-object figures** (light profiles, galaxies, 1D radial profiles, PDF error regions):
  `scripts/guides/plot/plotters.py`.

- **Fit plotting** (e.g. `FitImaging` residuals, chi-squared maps and fit subplots) is documented
  per dataset type in each dataset package's `plot.py` example, e.g. `scripts/imaging/plot.py` and
  `scripts/interferometer/plot.py`. Each of these also documents the `Visualizer`, which outputs
  these figures automatically during a model-fit.

- **Non-linear search results** (corner plots and search-specific visualization):
  `scripts/guides/plot/searches.py`.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
