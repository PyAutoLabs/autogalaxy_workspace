"""
Plots: GalaxyPlotter
====================

This example illustrates how to plot a `Galaxy` using a `GalaxyPlotter`.

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
__Galaxy__

First, lets create a `Galaxy` with multiple `LightProfile`'s and a `MassProfile`.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, -0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=2.0,
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

"""
__Grid__

We also need the 2D grid the `Galaxy`'s `Profile`'s are evaluated on.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the galaxy and grid to a `GalaxyPlotter` and call various `figure_*` methods to plot different attributes.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
galaxy_plotter.figures_2d(image=True)
galaxy_plotter.figures_1d(image=True)

"""
We can plot decomposed 1D profiles, which display a property of the galaxy in addition to those of its individual light 
and mass profiles. 

For the 1D plot of each profile, the 1D grid of (x) coordinates is centred on the profile and aligned with the 
major-axis. This means that if the galaxy consists of multiple profiles with different centres or angles the 1D plots 
are defined in a common way and appear aligned on the figure.
"""
galaxy_plotter.figures_1d_decomposed(image=True)

"""
__Subplots__

The `GalaxyPlotter` also has subplot method that plot each individual `Profile` in 2D as well as a 1D plot showing all
`Profiles` together.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
__Include__

A `Galaxy` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
include = aplt.Include2D(origin=True, mask=True, light_profile_centres=True)

mask = ag.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, radius=2.0
)
masked_grid = ag.Grid2D.from_mask(mask=mask)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=masked_grid, include_2d=include)
galaxy_plotter.figures_2d(image=True)


"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light and mass models estimated via a 
model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of a model-fit (the 
database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `einstein_radius1 with errors, which are plotted as vertical lines.

Below, we manually input two `Galaxy` objects with ligth and mass profiles that clearly show these errors on the figure.
"""
bulge_0 = ag.lp.Sersic(intensity=4.0, effective_radius=0.4, sersic_index=3.0)

disk_0 = ag.lp.Exponential(intensity=2.0, effective_radius=1.4)

galaxy_0 = ag.Galaxy(redshift=0.5, bulge=bulge_0, disk=disk_0)

bulge_1 = ag.lp.Sersic(intensity=4.0, effective_radius=0.8, sersic_index=3.0)

disk_1 = ag.lp.Exponential(intensity=2.0, effective_radius=1.8)

galaxy_1 = ag.Galaxy(redshift=0.5, bulge=bulge_1, disk=disk_1)

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=[galaxy_0, galaxy_1], grid=grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(image=True)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(image=True)

"""
Finish.
"""
