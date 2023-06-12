"""
Results: Units and Cosmology
============================

This tutorial illustrates how to perform unit conversions of modeling results from **PyAutoGalaxy**'s internal
units (e.g. arc-seconds, electrons per second, dimensionless mass units) to physical units (e.g. kiloparsecs,
magnitudes, solar masses).

This is used on a variety of important lens model cosmological quantities for example the effective radii of the
galaxies in the model.

__Errors__

To produce errors on unit converted quantities, you`ll may need to perform marginalization over samples of these
converted quantities (see `results/examples/samples.ipynb`).
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using dynesty. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk]",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Arcsec to Kiloparsec__

The majority of distance quantities in **PyAutoGalaxy** are in arcseconds, because this means that known redshifts are
not required in order to compose the model.

By assuming redshifts for the galaxy  galaxies we can convert their quantities from arcseconds to kiloparsecs.

Below, we compute the effective radii of the galaxy in kiloparsecs. To do this, we assume an AstroPy cosmology which 
allows us to compute the conversion factor `kpc_per_arcsec`.
"""
plane = result.max_log_likelihood_plane

galaxy = plane.galaxies[0]

cosmology = ag.cosmo.Planck15()

kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=galaxy.redshift)
effective_radius_kpc = galaxy.bulge.effective_radius * kpc_per_arcsec

"""
This `kpc_per_arcsec` can be used as a conversion factor between arcseconds and kiloparsecs when plotting images of
galaxies.

Below, we compute this value and plot the image in converted units of kiloparsecs.

This passes the plotting modules `Units` object a `ticks_convert_factor` and manually specified the new units of the
plot ticks.
"""
units = aplt.Units(ticks_convert_factor=kpc_per_arcsec, ticks_label=" kpc")

mat_plot = aplt.MatPlot2D(units=units)

plane_plotter = aplt.PlanePlotter(plane=plane, grid=dataset.grid, mat_plot_2d=mat_plot)
plane_plotter.figures_2d(image=True)


"""
__Brightness Units / Luminosity__

When plotting the image of a galaxy, each pixel value is also plotted in electrons / second, which is the unit values
displayed in the colorbar. 

A conversion factor between electrons per second and another unit can be input when plotting images of galaxies.

Below, we pass the exposure time of the image, which converts the units of the image from `electrons / second` to
electrons. 

Note that this input `ticks_convert_factor_values` is the same input parameter used above to convert mass plots like the 
convergence to physical units.
"""
exposure_time_seconds = 2000.0
units = aplt.Units(
    colorbar_convert_factor=exposure_time_seconds, colorbar_label=" seconds"
)

galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=galaxy, grid=dataset.grid, mat_plot_2d=mat_plot
)
galaxy_plotter.figures_2d(image=True)

"""
The luminosity of a galaxy is the total amount of light it emits, which is computed by integrating the light profile.
This integral is performed over the entire light profile, or within a specified radius.

Lets compute the luminosity of the galaxy in the default internal **PyAutoGalaxy** units of `electrons / second`.
Below, we compute the luminosity to infinite radius, which is the total luminosity of the galaxy, but one could
easily compute the luminosity within a specified radius instead.
"""
galaxy = plane.galaxies[0]

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
