"""
Fits
====

This guide shows how to fit data using the `FitInterferometer` object, including visualizing and interpreting its results.

References
----------

This example uses functionality described fully in other examples in the `guides` package:

- `guides/plot`: Using Plotter objects to plot and customize figures.
- `guides/units`: The source code unit conventions (e.g. arc seconds for distances and how to convert to physical units).
- `guides/data_structures`: The bespoke data structures used to store 1D and 2d arrays.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image is evaluated using.
"""
mask_radius = 3.5

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Loading Data__

We we begin by loading the dataset `simple` from .fits files, which is the dataset
we will use to demonstrate fitting.

This includes the method used to Fourier transform the real-space image to the uv-plane and compare
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for
interferometer datasets containing ~1-10 million visibilities.

This dataset was simulated using the `interferometer/simulator` example, read through that to have a better
understanding of how the data this exam fits was generated. The simulation uses the `TransformerDFT` to map
the real-space image to the uv-plane.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

"""
The `InterferometerPlotter` contains a subplot which plots all the key properties of the dataset simultaneously.

This includes the observed visibility data, RMS noise map and other information.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
Visibility data is in uv space, making it hard to interpret by eye.

The dirty images of the interferometer dataset can plotted, which use the transformer of the interferometer
to map the visibilities, noise-map or other quantity to a real-space image.
"""
dataset_plotter.subplot_dirty_images()

"""
__Fitting__

Following the previous overview example, we can make a galaxy from a collection of light profiles.

The combination of light profiles below is the same as those used to generate the simulated
dataset we loaded above.

It therefore produces a galaxy whose image looks exactly like the dataset.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Because the galaxy's light profiles are the same used to make the dataset, its image is nearly the same as the
observed image.

We can plot the image of the galaxies to confirm this, noting that its images are always in real space
(not Fourier space like the interferometer dataset) and therefore they can be directly visualized.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.set_title("Galaxies Image")
galaxies_plotter.figures_2d(image=True)

"""
However, the galaxy image is not what we observe in the interferometer dataset, because we observe the image as
visibilities in the uv-plane.

To compare directly to the data, we therefore need to Fourier transform the galaxy image to the uv-plane.

We do this by creating a `FitInterferometer` object, which performs this Fourier transform as part of the fitting
procedure.

The code plots the result of this, by using the `model_data` of the fit, which performs this Fourier transform
on the galaxy image above and plots the result visibilities in uv-space.
"""
fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d(model_data=True)

"""
The visibilities are again hard to interpret by eye, so we can plot the dirty image of the fit's model data. This
dirty image is the Fourier transform of the fit's model data (therefore the Fourier transform of the galaxy image) and
can be compared directly to the image of the galaxies above (albeit it still has the interferometer's PSF/dirty beam
convolved with it).
"""
fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d(dirty_image=True)

"""
The fit does a lot more than just Fourier transform the galaxy image it also creates the following:

 - The `residual_map`: The `model_data` visibilities subtracted from the observed dataset`s `data` visibilities.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

For a good galaxy model where the model and galaxies are representative of the dataset
residuals, normalized residuals and chi-squareds are minimized:
"""
fit_plotter.figures_2d(
    residual_map_real=True,
    residual_map_imag=True,
    normalized_residual_map_real=True,
    normalized_residual_map_imag=True,
    chi_squared_map_real=True,
    chi_squared_map_imag=True,
)

"""
A subplot can be plotted which contains all of the above quantities, as well as other information contained in the
galaxies such as the image and a normalized residual map where the colorbar
goes from 1.0 sigma to -1.0 sigma, to highlight regions where the fit is poor.
"""
fit_plotter.subplot_fit()

"""
Once again, dirty images are often easier to interpret, so we can plot a subplot of the dirty images of the data, model
data, residuals and chi-squared.
"""
fit_plotter.subplot_fit_dirty_images()

"""
The fit also provides us with a ``log_likelihood``, a single value quantifying how good the galaxies fitted the dataset.

Galaxy modeling, described in the next overview example, effectively tries to maximize this log likelihood value.
"""
print(fit.log_likelihood)

"""
__Bad Fit__

A bad galaxy model will show features in the residual-map and chi-squared map.

We can produce such an image by creating galaxies with different light profiles. In the example below, we
change the centre of the galaxy from (0.0, 0.0) to (0.05, 0.05), which leads to residuals appearing
in the fit.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
A new fit using this galaxies shows residuals, normalized residuals and chi-squared which are non-zero.
"""
fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
We also note that its likelihood decreases.
"""
print(fit.log_likelihood)

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

There is a `model_data`, which is the image-plane visibilities of the galaxies.

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_data)

"""
There are numerous ndarrays showing the goodness of fit:

 - `residual_map`: Residuals = (Data - Model_Data).
 - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
 - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
"""
print(fit.residual_map.slim)
print(fit.normalized_residual_map.slim)
print(fit.chi_squared_map.slim)

"""
There are `dirty` variants of the above maps, which transform the visibilities, residual-map, chi squared and other
values to to real-space images using the interferometer's transformer.

These real space images can be mapped between their `slim` and `native` representations (see the
`guides/data_structures` example for more information on these terms).
"""
print(fit.dirty_image.slim)  # Data
print(fit.dirty_model_image.slim)
print(fit.dirty_residual_map.slim)
print(fit.dirty_normalized_residual_map.slim)
print(fit.dirty_chi_squared_map.slim)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.

 - `noise_normalization`: The normalizing noise term in the likelihood function
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).

 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].

These sum other both the real and imaginary components of the visibilities to give a single value for each quantity.
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Galaxy Quantities__

The `FitInterferometer` object has specific quantities which break down each image of each galaxy:

 - `galaxy_model_visibilities_dict`: A dictionary which maps each galaxy in the galaxies to its model visibilities.

 - `galaxy_model_image_dict`: A dictionary which maps the model images of each galaxy.

These are not the dirty images, but instead the images of each galaxy that come from the galaxies object
(e.g. simply evaluating the galaxies image on the interferometer's real-space grid).
"""
print(fit.galaxy_model_visibilities_dict[galaxy].slim)
print(fit.galaxy_model_image_dict[galaxy].slim)

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection.

For example, one could output the galaxy model image to a .fits file such that
we could fit this image again with an independent pipeline.
"""
galaxy_model_image = fit.galaxy_model_image_dict[galaxy]
galaxy_model_image.output_to_fits(
    file_path=dataset_path / "galaxy_model_image.fits", overwrite=True
)

"""
Finish.
"""
