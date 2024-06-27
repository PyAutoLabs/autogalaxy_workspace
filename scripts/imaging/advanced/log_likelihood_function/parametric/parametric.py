"""
__Log Likelihood Function: Parametric__

This script provides a step-by-step guide of the **PyAutoGalaxy** `log_likelihood_function` which is used to fit
`Imaging` data with parametric light profiles (a Sersic bulge and Exponential disk).

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoGalaxy**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a linear run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

In order to perform a likelihood evaluation, we first load a dataset.

This example fits a simulated galaxy where the imaging resolution is 0.1 arcsecond-per-pixel resolution.
"""
dataset_path = path.join("dataset", "imaging", "simple")

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
This guide uses **PyAutoGalaxy**'s in-built visualization tools for plotting. 

For example, using the `ImagingPlotter` the imaging dataset we perform a likelihood evaluation on is plotted.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
a likelihood evaluation.

We define a 2D circular mask with a 3.0" radius.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`. 

a full description of over sampling and how to use it is given in `autogalaxy_workspace/*/guides/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sampling=ag.OverSamplingDataset(uniform=ag.OverSamplingUniform(sub_size=1))
)

"""
__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

These are given by `masked_dataset.grid`, which we can plot and see is a uniform grid of (y,x) Cartesian coordinates
which have had the 3.0" circular mask applied.

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to evaluate a light profile the intensity of the profile at the centre of each image-pixel is computed, making
it straight forward to compute the light profile's image to the image data.
"""
grid_plotter = aplt.Grid2DPlotter(grid=masked_dataset.grid)
grid_plotter.figure_2d()

print(
    f"(y,x) coordinates of first ten unmasked image-pixels {masked_dataset.grid[0:9]}"
)

"""
To perform light profile calculations we convert this 2D (y,x) grid of coordinates to elliptical coordinates:

 $\eta = \sqrt{(x - x_c)^2 + (y - y_c)^2/q^2}$

Where:

 - $y$ and $x$ are the (y,x) arc-second coordinates of each unmasked image-pixel, given by `masked_dataset.grid`.
 - $y_c$ and $x_c$ are the (y,x) arc-second `centre` of the light or mass profile.
 - $q$ is the axis-ratio of the elliptical light or mass profile (`axis_ratio=1.0` for spherical profiles).
 - The elliptical coordinates are rotated by a position angle, defined counter-clockwise from the positive 
 x-axis.

**PyAutoGalaxy** does not use $q$ and $\phi$ to parameterize a light profile but expresses these 
as "elliptical components", or `ell_comps` for short:

$\epsilon_{1} =\frac{1-q}{1+q} \sin 2\phi, \,\,$
$\epsilon_{2} =\frac{1-q}{1+q} \cos 2\phi.$
"""
profile = ag.EllProfile(centre=(0.1, 0.2), ell_comps=(0.1, 0.2))

"""
First we transform `masked_dataset.grid ` to the centre of profile and rotate it using its angle.
"""
transformed_grid = profile.transformed_to_reference_frame_grid_from(
    grid=masked_dataset.grid
)

grid_plotter = aplt.Grid2DPlotter(grid=transformed_grid)
grid_plotter.figure_2d()
print(
    f"transformed coordinates of first ten unmasked image-pixels {transformed_grid[0:9]}"
)

"""
Using these transformed (y',x') values we compute the elliptical coordinates $\eta = \sqrt{(x')^2 + (y')^2/q^2}$
"""
elliptical_radii = profile.elliptical_radii_grid_from(grid=transformed_grid)

print(
    f"elliptical coordinates of first ten unmasked image-pixels {elliptical_radii[0:9]}"
)

"""
__Likelihood Setup: Light Profiles (Setup)__

To perform a likelihood evaluation we now compose our galaxy model.

We first define the light profiles which represents the galaxy's light, in this case its bulge and disk, which will be 
used to fit the galaxy light.

A light profile is defined by its intensity $I (\eta_{\rm l}) $, for example the Sersic profile:

$I_{\rm  Ser} (\eta_{\rm l}) = I \exp \bigg\{ -k \bigg[ \bigg( \frac{\eta}{R} \bigg)^{\frac{1}{n}} - 1 \bigg] \bigg\}$

Where:

 - $\eta$ are the elliptical coordinates (see above).
 - $I$ is the `intensity`, which controls the overall brightness of the Sersic profile.
 - $n$ is the ``sersic_index``, which via $k$ controls the steepness of the inner profile.
 - $R$ is the `effective_radius`, which defines the arc-second radius of a circle containing half the light.

In this example, we assume our galaxy is composed of two light profiles, an elliptical Sersic and Exponential (a Sersic
where `sersic_index=4`) which represent the bulge and disk of the galaxy. 
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=0.5,
    effective_radius=1.6,
)

"""
Using the masked 2D grid defined above, we can calculate and plot images of each light profile component.

(The transformation to elliptical coordinates above are built into the `image_2d_from` function and performed 
implicitly).
"""
image_2d_bulge = bulge.image_2d_from(grid=masked_dataset.grid)

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=masked_dataset.grid)
bulge_plotter.figures_2d(image=True)

image_2d_disk = disk.image_2d_from(grid=masked_dataset.grid)

disk_plotter = aplt.LightProfilePlotter(light_profile=disk, grid=masked_dataset.grid)
disk_plotter.figures_2d(image=True)

"""
__Likelihood Setup: Galaxy__

We now combine the light profiles into a single `Galaxy` object.

When computing quantities for the light profiles from this object, it computes each individual quantity and 
adds them together. 

For example, for the `bulge` and `disk`, when it computes their 2D images it computes each individually and then adds
them together.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

"""
__Likelihood Step 1: Galaxy Image__

Compute a 2D image of the galaxy's light as the sum of its individual light profiles (the `Sersic` 
bulge and `Exponential` disk). 

This computes the `image` of each light profile and adds them together. 
"""
galaxy_image_2d = galaxy.image_2d_from(grid=masked_dataset.grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=masked_dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
To convolve the galaxy's 2D image with the imaging data's PSF, we need its `blurring_image`. 

This represents all flux values not within the mask, which are close enough to it that their flux blurs into the mask 
after PSF convolution.

To compute this, a `blurring_mask` and `blurring_grid` are used, corresponding to these pixels near the edge of the 
actual mask whose light blurs into the image:
"""
galaxy_blurring_image_2d = galaxy.image_2d_from(grid=masked_dataset.blurring_grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=masked_dataset.blurring_grid)
galaxy_plotter.figures_2d(image=True)

"""
__Likelihood Step 2: Convolution__

Convolve the 2D image of the galaxy above with the PSF in real-space (as opposed to via an FFT) using a `Convolver`.
"""
convolved_image_2d = masked_dataset.convolver.convolve_image(
    image=galaxy_image_2d, blurring_image=galaxy_blurring_image_2d
)

array_2d_plotter = aplt.Array2DPlotter(array=convolved_image_2d)
array_2d_plotter.figure_2d()

"""
__Likelihood Step 3: Likelihood Function__

We now quantify the goodness-of-fit of our galaxy model.

We compute the `log_likelihood` of the fit, which is the value returned by the **PyAutoGalaxy** `log_likelihood_function`.

The likelihood function for parametric galaxy modeling consists of two terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

We now explain what each of these terms mean.

__Likelihood Step 4: Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `convolved_image_2d`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_image = convolved_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = ag.Array2D(values=chi_squared_map, mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=chi_squared_map)
array_2d_plotter.figure_2d()

"""
__Likelihood Step 5: Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term in the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the galaxy modeling process and has no impact on the 
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Likelihood Step 6: Calculate The Log Likelihood!__

We can now, finally, compute the `log_likelihood` of the galaxy model, by combining the two terms computed above using
the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__Fit__

This 6 step process to perform a likelihood function evaluation is what is performed in the `FitImaging` object, which
those of you familiar will have seen before.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=masked_dataset, galaxies=galaxies)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)


"""
__Galaxy Modeling__

To fit a galaxy model to data, **PyAutoGalaxy** samples the likelihood function illustrated in this tutorial using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
but **PyAutoGalaxy** supports multiple MCMC and optimization algorithms. 

__Wrap Up__

We have presented a visual step-by-step guide to the **PyAutoGalaxy** parametric likelihood function, which uses 
analytic light profiles to fit the galaxy light.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in the `guides` package:

 - `over_sampling`: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and used to evaluate the light profile more accurately.
"""
