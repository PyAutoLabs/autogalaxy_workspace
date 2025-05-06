"""
Plots: MultiYX1DPlotter
=========================

This example illustrates how to plot multi 1D figure lines on the same plot.

It uses the specific example of plotting a `LightProfile`'s 1D image using multiple `LightProfilePlotter`'s.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
First, lets create two simple `LightProfile`'s which we'll plot the 1D images of on the same figure.
"""
light_0 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=1.0,
    sersic_index=2.0,
)

light_1 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=2.0,
    sersic_index=2.0,
)

"""
We also need the 2D grid the `LightProfile`'s are evaluated on.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
We now pass the light profiles and grid to a `LightProfilePlotter` and create a `MultiYX1DPlotter` which will be
used to plot both of their images in 1D on the same figure.
"""
mat_plot = aplt.MatPlot1D(yx_plot=aplt.YXPlot(plot_axis_type="semilogy"))

light_profile_plotter_0 = aplt.LightProfilePlotter(
    light_profile=light_0, grid=grid, mat_plot_1d=mat_plot
)
light_profile_plotter_1 = aplt.LightProfilePlotter(
    light_profile=light_1, grid=grid, mat_plot_1d=mat_plot
)

"""
We use these plotters to create a `MultiYX1DPlotter` which plot both of their images in 1D on the same figure.
"""
multi_plotter = aplt.MultiYX1DPlotter(
    plotter_list=[light_profile_plotter_0, light_profile_plotter_1]
)

"""
We now use the multi plotter to plot the images, where:

 - `func_name`: he name of the `LightProfilePlotter` function we call, in this case `figures_1d`.
 - `figure_name`: the name of the function's boolean input we set to True such that it plots, in this case `image`.
 
The input therefore corresponds to us writing `light_profile_plotter.figures_1d(image=True)` for each plotter.
"""
multi_plotter.figure_1d(func_name="figures_1d", figure_name="image")
