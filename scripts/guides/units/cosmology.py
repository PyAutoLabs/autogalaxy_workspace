"""
Units and Cosmology
===================

This tutorial illustrates how to perform unit conversions from **PyAutoGalaxy**'s internal units (e.g. arc-seconds,
electrons per second, dimensionless mass units) to physical units (e.g. kiloparsecs, magnitudes, solar masses).

This is used on a variety of important cosmological quantities for example the effective radii of galaxies.

__Errors__

To produce errors on unit converted quantities, you`ll may need to perform marginalization over samples of these
converted quantities (see `results/examples/samples.ipynb`).
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt


"""
__Galaxy__

We set up a simple galaxy and grid which will illustrate the unit conversion functionality. 
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
)

"""
__Arcsec to Kiloparsec__

Distance quantities are in arcseconds natively, because this means that known redshifts are not required in order to 
perform certain calculations.

By assuming redshifts for galaxies we can convert their quantities from arcseconds to kiloparsecs.

Below, we compute the effective radii of the galaxy in kiloparsecs. To do this, we assume a cosmology (internally
this uses the AstroPy Cosmology module) which allows us to compute the conversion factor `kpc_per_arcsec`.
"""
cosmology = ag.cosmo.Planck15()

kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=galaxy.redshift)
effective_radius_kpc = galaxy.bulge.effective_radius * kpc_per_arcsec

"""
This `kpc_per_arcsec` can be used as a conversion factor between arcseconds and kiloparsecs when plotting images of
galaxies.

We compute this value and plot the image in converted units of kiloparsecs.

This passes the plotting modules `Units` object a `ticks_convert_factor` and manually specified the new units of the
plot ticks.
"""
units = aplt.Units(ticks_convert_factor=kpc_per_arcsec, ticks_label=" kpc")

mat_plot = aplt.MatPlot2D(units=units)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid, mat_plot_2d=mat_plot)
galaxy_plotter.figures_2d(image=True)


"""
__Brightness Units / Luminosity__

When plotting the image of a galaxy, each pixel value is plotted in electrons / second, which is the unit values
displayed in the colorbar. 

A conversion factor between electrons per second and another unit can be input when plotting images of galaxies.

Below, we pass the exposure time of the image, which converts the units of the image from `electrons / second` to
electrons. 
"""
exposure_time_seconds = 2000.0
units = aplt.Units(
    colorbar_convert_factor=exposure_time_seconds, colorbar_label=" seconds"
)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid, mat_plot_2d=mat_plot)
galaxy_plotter.figures_2d(image=True)

"""
The luminosity of a galaxy is the total amount of light it emits, which is computed by integrating the light profile.
This integral is performed over the entire light profile, or within a specified radius.

Lets compute the luminosity of the galaxy in the default internal **PyAutoGalaxy** units of `electrons / second`.
Below, we compute the luminosity to infinite radius, which is the total luminosity of the galaxy, but one could
easily compute the luminosity within a specified radius instead.
"""
luminosity = galaxy.luminosity_within_circle_from(radius=np.inf)
print("Luminosity (electrons / second) = ", luminosity)

"""
From a luminosity in `electrons / second`, we can convert it to other units, such as `Jansky` or `erg / second`. 
This can also be used to compute the magnitude of the galaxy, which is the apparent brightness of the galaxy in a
given bandpass.

This functionality is not currently implemented in **PyAutoGalaxy**, but would be fairly simple for you to do
yourself (e.g. using the `astropy` package). If you want to contribute to **PyAutoGalaxy**, this would be a great
first issue to tackle, so please get in touch on SLACK!
"""
