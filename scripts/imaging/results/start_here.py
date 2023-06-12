"""
Results: Start Here
===================

This script is the starting point for investigating the results of modeling and it provides
an overview of the modeling API.

After reading this script, the `examples` folder provides more detailed examples for analysing the different aspects of
performing modeling results outlined here.

__Model__

We begin by fitting a quick model to a simple dataset, which we will use to illustrate the modeling
results API.

If you are not familiar with the modeling API and process, checkout the `autogalaxy_workspace/examples/modeling`
folder for examples.
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
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
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
    nlive=100,
    walks=10,
    number_of_cores=1,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
__Samples__

The result's `Samples` object contains the complete set of non-linear search dynesty samples, where each sample 
corresponds to a set of model parameters that were evaluated and accepted. 

The examples script `autogalaxy_workspace/*/imaging/results/examples/samples.py` provides a detailed description of 
this object, including:

 - Extracting the maximum likelihood model.
 - Using marginalized PDFs to estimate errors on the model parameters.
 - Deriving errors on derived quantities, such as the Einstein radius.

Below, is an example of how to use the `Samples` object to estimate the mass model parameters which are 
the median of the probability distribution function and its errors at 3 sigma confidence intervals.
"""
samples = result.samples

median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance.galaxies.galaxy.bulge)
print()

ue3_instance = samples.values_at_upper_sigma(sigma=3.0)
le3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance.galaxies.galaxy.bulge, "\n")
print(le3_instance.galaxies.galaxy.bulge, "\n")

"""
__Plane__

The result's maximum likelihood `Plane` object contains everything necessary to perform calculations with the model
like retrieving the images of each galaxy.

The examples script `autogalaxy_workspace/*/imaging/results/examples/plane.py` provides a detailed 
description of this object, including:

 - Producing individual images of the galaxies in the plane.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

Below, is an example of how to use the `Plane` object to calculate the image of the galaxies.
"""
plane = result.max_log_likelihood_plane

image = plane.image_2d_from(grid=dataset.grid)

"""
__Fits__

The result's maximum likelihood `FitImaging` object contains everything necessary to inspect the model fit to the 
data.

The examples script `autogalaxy_workspace/*/imaging/results/examples/fits.py` provides a detailed description of this 
object, including:

 - How to inspect the residuals, chi-squared, likelihood and other quantities.
 - Outputting resulting images (e.g. the galaxy reconstruction) to hard-disk.
 - Refitting the data with other models from the `Samples` object, to investigate how sensitive the fit is to
   different models.

Below, is an example of how to use the `FitImaging` object to output the galaxy reconstruction to print the 
chi-squared and log likelihood values.
"""
fit = result.max_log_likelihood_fit

print(fit.chi_squared)
print(fit.log_likelihood)

"""
__Galaxies__

The result's maximum likelihood `Galaxy` objects contain everything necessary to inspect the individual properties of
the galaxies.

The examples script `autogalaxy_workspace/*/imaging/results/examples/galaxies.py` provides a detailed description 
of this object, including:

 - How to plot individual galaxy images, such as the galaxy's images.
 - Plotting the individual light profiles and mass profiles of the galaxies.
 - Making one dimensional profiles of the galaxies, such as their light and mass profiles as a function of radius.
 
Below, is an example of how to use the `Galaxy` objects to plot the galaxy's image.
"""
plane = result.max_log_likelihood_plane

galaxy = plane.galaxies[0]
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Cosmological Quantities__

The maximum likelihood model includes cosmological quantities, which can be computed via the result.

The examples script `autogalaxy_workspace/*/imaging/results/examples/cosmological_quantities.py` provides a detailed 
description of this object, including:

 - Calculating the Einstein radius of the galaxy.
 - Converting quantities like the Einstein radius or effective radius from arcseconds to kiloparsecs.
 - Computing the Einstein mass of the galaxy in solar masses.
 
Below, is an example of how to convert the effective radius of the galaxy from arcseconds to kiloparsecs.
"""
plane = result.max_log_likelihood_plane

cosmology = ag.cosmo.Planck15()

galaxy = plane.galaxies[0]
galaxy_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=galaxy.redshift)
galaxy_effective_radius_kpc = galaxy.bulge.effective_radius * galaxy_kpc_per_arcsec

"""
__Linear Light Profiles / Basis Objects__

A model can be fitted using a linear light profile, which is a light profile whose `intensity` parameter is 
sovled for via linear algebra.

This includes Basis objects such as a Multi-Gaussian expansion of Shapelets.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality.

The example script `autogalaxy_workspace/*/imaging/results/examples/linear.py` provides a detailed description of 
this functionality, including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.
 
The fit above did not use a pixelization, so we omit a example of the API below.

__Pixelization__

The model can reconstruct the galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autogalaxy_workspace/*/imaging/results/examples/pixelizations.py` provides a detailed description 
of inspecting the results of a fit using a pixelization, including:

 - Producing galaxy reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the galaxy's image using the pixelization.

The fit above did not use a pixelization, so we omit a example of the API below.
"""
