"""
Results: Fits
=============

This tutorial inspects the model's fit to the data using the `FitImaging` object inferred by the non-linear search, 
for example visualizing and interpreting its results.

This includes inspecting the residuals, chi-squared and other goodness-of-fit quantities.

__Plot Module__

This example uses the **PyAutoGalaxy** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/imaging/results/examples/data_structure.ipynb` example.

__Other Models__

This tutorial does not use a pixelized source reconstruction or linear light profiles, which have their own dediciated
functionality that interfacts with the `FitImaging` object.

This is described in the dedicated example scripts `results/examples/linear.py` and `results/examples/pixelizaiton.py`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

Note that the model that is fitted has two galaxies, as opposed to just one like usual!
"""
dataset_name = "sersic_x2"
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

bulge_0 = af.Model(ag.lp.Sersic)
bulge_0.centre = (0.0, -1.0)

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_0)

bulge_1 = af.Model(ag.lp.Sersic)
bulge_1.centre = (0.0, 1.0)

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_1)

model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]__x2",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Fit__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit` which we can visualize.
"""
fit = result.max_log_likelihood_fit

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

These use the `slim` and `native` API discussed in the previous results tutorial.

There is a `model_image`, which is the image of the plane we inspected in the previous tutorial blurred with the 
imaging data's PSF. 

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the 
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_image.slim)
print(fit.model_image.native)

"""
There are numerous ndarrays showing the goodness of fit: 

 - `residual_map`: Residuals = (Data - Model_Data).
 - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
 - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
"""
print(fit.residual_map.slim)
print(fit.residual_map.native)

print(fit.normalized_residual_map.slim)
print(fit.normalized_residual_map.native)

print(fit.chi_squared_map.slim)
print(fit.chi_squared_map.native)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.
 - `noise_normalization`: The normalizing noise term in the likelihood function 
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).
 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Galaxy Quantities__

The `FitImaging` object has specific quantities which break down each image of each galaxy:

 - `model_images_of_galaxies_list`: Model-images of each individual galaxy, which in this example is a model image of 
 the two galaxies in the model. Both images are convolved with the imaging's PSF.
 
 - `subtracted_images_of_galaxies_list`: Subtracted images of each individual galaxy, which are the data's image with
 all other galaxy's model-images subtracted. For example, the first subtracted image has the second galaxy's model image
 subtracted and therefore is of only the right galaxy's emission.
"""
print(fit.model_images_of_galaxies_list[0].slim)
print(fit.model_images_of_galaxies_list[1].slim)

print(fit.subtracted_images_of_galaxies_list[0].slim)
print(fit.subtracted_images_of_galaxies_list[1].slim)

"""
__Unmasked Quantities__

All of the quantities above are computed using the mask which was used to fit the data.

The `FitImaging` can also compute the unmasked blurred image of each plane.
"""
print(fit.unmasked_blurred_image.native)
print(fit.unmasked_blurred_image_of_galaxies_list[0].native)
print(fit.unmasked_blurred_image_of_galaxies_list[1].native)

"""
__Mask__

We can use the `Mask2D` object to mask regions of one of the fit's maps and estimate quantities of it.

Below, we estimate the average absolute normalized residuals within a 1.0" circular mask, which would inform us of
how accurate the model fit is in the central regions of the data.
"""
mask = ag.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

normalized_residuals = fit.normalized_residual_map.apply_mask(mask=mask)

print(np.mean(np.abs(normalized_residuals.slim)))

"""
__Pixel Counting__

An alternative way to quantify residuals like the galaxy light residuals is pixel counting. For example, we could sum
up the number of pixels whose chi-squared values are above 10 which indicates a poor fit to the data.

Whereas computing the mean above the average level of residuals, pixel counting informs us how spatially large the
residuals extend. 
"""
mask = ag.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

chi_squared_map = fit.chi_squared_map.apply_mask(mask=mask)

print(np.sum(chi_squared_map > 10.0))

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection. 

For example, one could output the galaxy subtracted image of the second galaxy to a .fits file such that
we could fit this image again with an independent modeling script.
"""
galaxy_subtracted_image_2d = fit.subtracted_images_of_galaxies_list[1]
galaxy_subtracted_image_2d.output_to_fits(
    file_path=path.join(dataset_path, "galaxy_subtracted_data.fits"), overwrite=True
)

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

plane = ag.Plane(galaxies=instance.galaxies)

fit = ag.FitImaging(dataset=dataset, plane=plane)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Wrap Up__

In this tutorial, we saw how to inspect the quality of a model fit using the fit imaging object.

If you are modeling galaxies using interferometer data we cover the corresponding fit object in tutorial 6.
"""
