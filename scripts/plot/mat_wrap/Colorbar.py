"""
Plots: Colorbar
===============

This example illustrates how to customize the Colorbar in PyAutoGalaxy figures and subplots.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
First, lets load an example Hubble Space Telescope image of a real galaxy as an `Array2D`.
"""
dataset_path = path.join("dataset", "imaging", "complex")
data_path = path.join(dataset_path, "data.fits")
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
We can customize the colorbar using the `Colorbar` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.colorbar.html
"""
cb = aplt.Colorbar(
    fraction=0.047,
    shrink=5.0,
    aspect=1.0,
    pad=0.01,
    anchor=(0.0, 0.5),
    panchor=(1.0, 0.0),
)

mat_plot = aplt.MatPlot2D(colorbar=cb)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The labels of the `Colorbar` can also be customized. 

This uses the `cb.ax.set_yticklabels` to manually override the tick locations and labels:
 
 https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html
 
The input parameters of both the above methods can be passed into the `Colorbar` object.
"""
cb = aplt.Colorbar(manual_tick_labels=[1.0, 2.0], manual_tick_values=[0.0, 0.25])


mat_plot = aplt.MatPlot2D(colorbar=cb)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The colorbar can be disabled by setting the `Colorbar` object to `False`.
"""
mat_plot = aplt.MatPlot2D(colorbar=False)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
