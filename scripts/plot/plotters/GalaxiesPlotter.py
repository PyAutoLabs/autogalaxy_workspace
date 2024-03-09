"""
Plots: GalaxiesPlotter
===================

This example illustrates how to plot a `Plane` using a `GalaxiesPlotter`.

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
__Grid__

First, lets create `Grid2D`.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Plane__

We create a `Plane` representing a `Galaxy` with a `LightProfile`.
"""
bulge = ag.lp.Sersic(
    centre=(0.1, 0.1),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
    intensity=0.3,
    effective_radius=1.0,
    sersic_index=2.5,
)

galaxy = ag.Galaxy(redshift=1.0, bulge=bulge)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
__Figures__

We can plot the galaxies by passing it and our `grid to a` GalaxiesPlotter` and calling various `figure_*` methods.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Include__

A `Plane` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
mask = ag.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, radius=2.0
)
masked_grid = ag.Grid2D.from_mask(mask=mask)

include = aplt.Include2D(origin=True, mask=True, light_profile_centres=True)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=masked_grid, include_2d=include
)
galaxies_plotter.figures_2d(image=True)

"""
__Log10__

A galaxy's light and mass profiles are often clearer in log10 space, which inputting `use_log10=True` into 
the `MatPlot2D` object will do.

The same image can be set up manually via the `CMap`, `Contour` and `Colorbar` objects, but given this is a common
use-case, the `use_log10` input is provided for convenience.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=masked_grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
galaxies_plotter.figures_2d(image=True, convergence=True, potential=True)

"""
Finish.
"""
