"""
Plots: Mat Plot
===============

This example illustrates the API for customizing the appearance of figures using arguments passed directly to
`plot_array` (and related functions like `plot_grid`).

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting.

__Contents__

- **Setup**: Set up all objects used to illustrate plotting.
- **Output**: Control where and how figures are saved.
- **Figure Size and Aspect**: Control the figure dimensions.
- **Labels**: Customize the title, x-label and y-label.
- **Color Map**: Customize the colormap, color scale limits and log10 normalization.
- **Log10**: Plot quantities in log10 space.
- **Contours**: Overlay contour lines on a figure.
- **Ticks / Colorbar / Legend / Annotate / Text / Axis**: Config-file-only styling.

__Setup__

To illustrate plotting, we require standard objects like a grid and dataset.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

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
        [sys.executable, "scripts/guides/plot/simulator.py"],
        check=True,
    )

data_path = dataset_path / "data.fits"
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
__Output__

We can control where figures are saved using `output_path=`, `output_filename=` and `output_format=`.

Below, we save the figure as a `.png` file named `example.png` in the `plot/plots` folder of the workspace.
"""
aplt.plot_array(
    array=data,
    title="Image",
    output_path=Path("notebooks") / "plot" / "plots",
    output_filename="example",
    output_format="png",
)

"""
We can specify a list of output formats so the figure is saved to all of them.
"""
aplt.plot_array(
    array=data,
    title="Image",
    output_path=Path("notebooks") / "plot" / "plots",
    output_filename="example",
    output_format=["png", "pdf"],
)

"""
To display a figure on screen rather than saving it, pass `output_format="show"`.
"""
aplt.plot_array(array=data, title="Image", output_format="show")

"""
__Figure Size and Aspect__

We can control the figure size using `figsize=` and the image aspect ratio using `aspect=`.
"""
aplt.plot_array(array=data, title="Image")

aplt.plot_array(array=data, title="Image")

"""
__Labels__

We can set the title, y-label and x-label using `title=`, `ylabel=` and `xlabel=`.
"""
aplt.plot_array(
    array=data,
    title="This is the title",
)

"""
__Color Map__

We can customize the colormap using `colormap=`, `vmin=` and `vmax=`.
"""
aplt.plot_array(array=data, title="Image", colormap="jet", vmin=0.0, vmax=1.0)

aplt.plot_array(array=data, title="Image", colormap="hot", vmin=0.0, vmax=2.0)

aplt.plot_array(array=data, title="Image", colormap="twilight")

"""
__Log10__

Many quantities are easier to interpret in log10 space. Pass `use_log10=True`.
"""
aplt.plot_array(array=data, title="Image", use_log10=True)

"""
__Contours__

Contour lines can be overlaid on a figure using `contours=N`, where `N` is the number of contour levels.

We first create a light profile image to use for the contour example.
"""
light = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.1,
    effective_radius=1.0,
    sersic_index=4.0,
)

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

image = light.image_2d_from(grid=grid)

aplt.plot_array(array=image, title="Image with Contours")

aplt.plot_array(array=image, title="Image with Fewer Contours")

"""
__Ticks / Colorbar / Legend / Annotate / Text / Axis__

The following aspects of figure appearance are controlled via the config files in
`autogalaxy_workspace/config/visualize/mat_wrap/` rather than via runtime Python arguments:

 - Tick label sizes, fonts and formatting (`TickParams`, `YTicks`, `XTicks` in `mat_wrap.yaml`)
 - Colorbar styling (`Colorbar`, `ColorbarTickParams` in `mat_wrap.yaml`)
 - Legend appearance (`Legend` in `mat_wrap.yaml`)
 - Annotations (`Annotate` in `mat_wrap.yaml`)
 - Text overlays (`Text` in `mat_wrap.yaml`)
 - Axis extent (`Axis` in `mat_wrap.yaml`)
 - Scatter marker styling for overlays like origins, positions and borders
   (`OriginScatter`, `MaskScatter`, `BorderScatter`, etc. in `mat_wrap_2d.yaml`)

To change these defaults, edit the corresponding YAML config file and restart the Python session
(or Jupyter kernel) for the changes to take effect.

Finish.
"""
