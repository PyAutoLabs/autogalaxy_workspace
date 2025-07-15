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
__Galaxies__

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
__Visuals__

A galaxy consists of light and mass profiles, and their centres can be extracted and plotted over the image. 
The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter` and `MassProfilesCentreScatter`,
describes how to plot these visuals over images.

If the galaxy has a mass profile, it also has critical curves and caustics. The `visuals.ipynb` notebook, under the 
sections `CriticalCurvesLine` and `CausticsLine`, describes how to plot these visuals over images.

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
