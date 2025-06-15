"""
Plots: Legend
=============

This example illustrates how to customize the Matplotlib legend of a PyAutoGalaxy figures and subplot.

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
y = ag.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)
x = ag.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)

"""
We can customize the legend using the `Legend` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html
"""
legend = aplt.Legend(include_2d=True, loc="upper left", fontsize=10, ncol=2)

mat_plot = aplt.MatPlot1D(legend=legend)

array_plotter = aplt.Array1DPlotter(y=y, x=x, mat_plot_1d=mat_plot)
array_plotter.figure_1d()

"""
Finish.
"""
