"""
Plots: Start Here
=================

This example illustrates the API for plotting.

__Contents__

- **Dataset**: Load an example image used to illustrate plotting.
- **Figures**: Plot the image using `plot_array`.
- **Customization**: Customize the appearance of the figure using direct arguments.
- **Log10**: Plot quantities in log10 space.
- **Configs**: Customize the appearance of figures using the config files.
- **Subplots**: Plot multiple images using subplot functions.
- **Visuals**: Add visuals to the figure, such as a mask or light profile centres.
- **Searches**: Visualize the results of a search.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

First, lets load an example image of a galaxy as an `Array2D`.
"""
dataset_path = Path("dataset") / "imaging" / "simple"

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

data_path = dataset_path / "data.fits"
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
__Figures__

We plot an array by passing it to `plot_array`.

The `autogalaxy.workspace.*.plot.plotters` example illustrates every type of object that can be plotted,
for example datasets, galaxies, fits, etc.
"""
aplt.plot_array(array=data, title="Image")

"""
__Plot Customization__

You can customize matplotlib properties by passing arguments directly to `plot_array`.

Common options:

 - `title=`: Figure title.
 - `figsize=`: Figure dimensions as `(width, height)`.
 - `colormap=`: Matplotlib colormap name, e.g. `"jet"` or `"gray"`.
 - `xlabel=`, `ylabel=`: Axis labels.
 - `vmin=`, `vmax=`: Color scale limits.
"""
aplt.plot_array(
    array=data,
    title="Image",
)

"""
__Log10__

Many of the quantities we plot are often clearer in log10 space, for example the image of a galaxy.

Pass `use_log10=True` to plot in log10 space.
"""
aplt.plot_array(array=data, title="Image", use_log10=True)

"""
__Configs__

All matplotlib defaults can be customized via the config files, such that those values are used every time.

Checkout the `mat_wrap.yaml` file in `autogalaxy_workspace/config/visualize/mat_wrap`.

All default matplotlib values are here. There are a lot of entries, so lets focus on whats important for displaying
figures:

 - mat_wrap.yaml -> Figure -> figure: -> figsize
 - mat_wrap.yaml -> YLabel -> figure: -> fontsize
 - mat_wrap.yaml -> XLabel -> figure: -> fontsize
 - mat_wrap.yaml -> TickParams -> figure: -> labelsize
 - mat_wrap.yaml -> YTicks -> figure: -> labelsize
 - mat_wrap.yaml -> XTicks -> figure: -> labelsize

__Subplots__

In addition to plotting individual figures, **PyAutoGalaxy** can also plot subplots showing all components of an
object simultaneously.

For example, `subplot_imaging_dataset` plots the data, noise-map and PSF of an `Imaging` dataset together.

Other subplot functions include `subplot_fit_imaging`, `subplot_galaxies`, `subplot_of_light_profiles`, etc.

__Visuals__

Visuals can be added to any figure by passing them as keyword arguments to `plot_array`.

For example, we can plot a mask on the image above by passing `mask=mask`.

The `autogalaxy.workspace.*.plot.visuals` example illustrates every visual overlay argument,
for example `mask=`, `grid=`, `positions=`, `lines=`, `vector_yx=`, `border=`, `patches=`, etc.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)

aplt.plot_array(array=data, title="Image")

"""
__Customize Visuals With Config__

The default appearance of figures is provided in the `config/visualize/` config files, which you should
checkout now. For example, tick label sizes and colorbar styling are controlled there.

__Searches__

Model-fits using a non-linear search (e.g. Nautilus, Emcee) produce search-specific visualization.

The `searches` example illustrates how to perform this visualization for every search (e.g. via the `corner_anesthetic` function).

Finish.
"""
