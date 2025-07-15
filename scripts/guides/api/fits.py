"""
Fits
====

This guide shows how to fit data using the `FitImaging` object, including visualizing and interpreting its results.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/guides/data_structures.ipynb` guide.

__Other Models__

This tutorial does not use a pixelized source reconstruction or linear light profiles, which have their own dediciated
functionality that interfacts with the `FitImaging` object.

This is described in the dedicated example scripts `modeling/features/linear_light_profiles.py`
and `modeling/features/pixelizaiton.py`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Loading Data__

We we begin by loading the galaxy dataset `simple__sersic` from .fits files, which is the dataset we will use to 
demonstrate fitting.
"""
dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
We can use the `ImagingPlotter` to plot the image, noise-map and psf of the dataset.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True, noise_map=True, psf=True)

"""
The `ImagingPlotter` also contains a subplot which plots all these properties simultaneously.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Grid__

When calculating the amount of emission in each image pixel from galaxies, a two dimensional line integral of all of 
the emission within the area of that pixel should be performed. However, for complex models this can be difficult 
to analytically compute and can lead to slow run times.

Instead, a high resolution grid of (y,x) coordinates is used, where the line integral of all the flux in a pixel
is computed by summing up the flux values at these (y,x) coordinates. The code below uses the same over sampling
size in every pixel for simplicity, but adaptive over sampling can also be used, which adapts the over sampling
to the bright regions of the image but uses computationally faster lower valkues in the outer regions.
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4,
)


"""
__Mask__

We next mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

To do this we can use a ``Mask2D`` object, which for this example we'll create as a 3.0" circle.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

"""
We now combine the imaging dataset with the mask.

Here, the mask is also used to compute the `Grid2D` we used in the previous overview to compute the light profile 
emission, where this grid has the mask applied to it.
"""
dataset = dataset.apply_mask(mask=mask)

grid_plotter = aplt.Grid2DPlotter(grid=dataset.grid)
grid_plotter.figure_2d()

"""
Here is what our image looks like with the mask applied, where PyAutoGalaxy has automatically zoomed around the mask
to make the galaxyed source appear bigger.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
__Fitting__

Following the previous overview, we can make a collection of galaxies from light profiles and individual galaxy objects..

The combination of light profiles below is the same as those used to generate the simulated dataset we loaded above.

It therefore produces galaxies whose image looks exactly like the dataset. As discussed in the previous overview, 
galaxies can be extended to include additional light profiles and galaxy objects, for example if you wanted to fit data
with multiple galaxies.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, -1.0),
        ell_comps=(0.25, 0.1),
        intensity=0.1,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 1.0),
        ell_comps=(0.0, 0.1),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.figures_2d(image=True)

"""
We now use the `FitImaging` object to fit the galaxies to the dataset. 

The fit performs the necessary tasks to create the `model_image` we fit the data with, such as blurring the
image of the galaxies with the imaging data's Point Spread Function (PSF). We can see this by comparing the galaxies 
image (which isn't PSF convolved) and the fit`s model image (which is).

[For those not familiar with Astronomy data, the PSF describes how the observed emission of the galaxy is blurred by
the telescope optics when it is observed. It mimicks this blurring effect via a 2D convolution operation].
"""
fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d(model_image=True)

"""
The fit creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset`s `image`.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

we'll plot all 3 of these, alongside a subplot containing them all, which also shows the data,
model image and individual galaxies in the fit.

For a good model where the model image and galaxies are representative of the galaxy system the
residuals, normalized residuals and chi-squared are minimized:
"""
fit_plotter.figures_2d(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)
fit_plotter.subplot_fit()

"""
The overall quality of the fit is quantified with the `log_likelihood` (the **HowToGalaxy** tutorials explains how
this is computed).
"""
print(fit.log_likelihood)

"""
__Bad Fit__

In contrast, a bad model will show features in the residual-map and chi-squared map.

We can produce such an image by using different galaxies. In the example below, we 
change the centre of the galaxies from (0.0, -1.0) to (0.0, -1.05), and from (0.0, 1.0) to (0.0, 1.05) which leads to 
residuals appearing in the centre of the fit.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, -1.05),
        ell_comps=(0.25, 0.1),
        intensity=0.1,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 1.05),
        ell_comps=(0.0, 0.1),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

fit_bad = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
A new fit using these galaxies shows residuals, normalized residuals and chi-squared which are non-zero. 
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit_bad)

fit_plotter.figures_2d(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)
fit_plotter.subplot_fit()

"""
We also note that its likelihood decreases.
"""
print(fit.log_likelihood)

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

These use the `slim` and `native` API discussed in the previous results tutorial.

There is a `model_data`, which is the image of the galaxies we inspected in the previous tutorial blurred with the 
imaging data's PSF. 

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the 
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_data.slim)
print(fit.model_data.native)

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

The `FitImaging` can also compute the unmasked blurred image of the galaxies.
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
    file_path=Path(dataset_path, "galaxy_subtracted_data.fits"), overwrite=True
)

"""
__Modeling Results__

Modeling uses a non-linear search to fit a model of galaxies to a dataset.

It is illustrated in the `modeling` packages of `autogalaxy_workspace`.

Modeling results have some specific functionality and use cases, which are described in the `results` packages of
`autogalaxy_workspace`,  in particular the `galaxies_fit.py` example script which describes: 

 - `Max Likelihood`: Extract and plot the galaxy models which maximize the likelihood of the fit.
 - `Samples`, Extract the samples of the non-linear search and inspect specific parameter values.
 - `Errors`: Makes plots that quantify the errors on the inferred galaxy properties.
 - `Refitting` Refit specific models from the modeling process to the dataset. 

__Wrap Up__

In this tutorial, we saw how to inspect the quality of a model fit using the fit imaging object.

If you are modeling galaxies using interferometer data we cover the corresponding fit object in tutorial 6.
"""
