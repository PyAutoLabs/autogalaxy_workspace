"""
Results: Start Here
===================

This script is the starting point for investigating the results of modeling and it provides
an overview of the modeling API.

The majority of results are dataset independent, meaning that the same API can be used to inspect the results of any
model. Therefore, for the majority of results we refer you to the `autogalaxy_workspace/imaging/results` package,
which details the API which can be copy and pasted for interferometer fits.

The `examples` folder here does provide specific examples of how to inspects the results of fits using
interferometer datasets.

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

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0, sub_size=1
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

settings_dataset = ag.SettingsInterferometer(transformer_class=ag.TransformerNUFFT)

bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("interferometer", "modeling"),
    name="light[bulge_disk]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = ag.AnalysisInterferometer(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
__Samples__

The result's `Samples` object contains the complete set of non-linear search nautilus samples, where each sample 
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

The result's maximum likelihood `FitInterferometer` object contains everything necessary to inspect the model 
fit to the  data.

The examples script `autogalaxy_workspace/*/imaging/results/examples/fits.py` provides a detailed description of this 
object, including:

 - How to inspect the residuals, chi-squared, likelihood and other quantities.
 - Outputting resulting images (e.g. the source reconstruction) to hard-disk.
 - Refitting the data with other models from the `Samples` object, to investigate how sensitive the fit is to
   different models.

This script uses a `FitImaging` object, but the API for the majority of quantities are identical for an 
interferometer fit.

Below, is an example of how to use the `FitImaging` object to output the source reconstruction to print the 
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

 - How to plot individual galaxy images, such as the source galaxy's image-plane and source-plane images.
 - Plotting the individual light profiles and mass profiles of the galaxies.
 - Making one dimensional profiles of the galaxies, such as their light and mass profiles as a function of radius.
 
Below, is an example of how to use the `Galaxy` objects to plot the source galaxy's source-plane image.
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
 
Below, is an example of how to convert the effective radius of the source galaxy from arcseconds to kiloparsecs.
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

The model can reconstruct the source galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autogalaxy_workspace/*/imaging/results/examples/pixelizations.py` provides a detailed description 
of inspecting the results of a fit using a pixelization, including:

 - Producing source reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the source galaxy's image using the pixelization.

The fit above did not use a pixelization, so we omit a example of the API below.
"""
