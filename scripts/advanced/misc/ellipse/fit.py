"""
Fits
====

This guide shows how to fit data using ellipse fitting and the `FitEllipse` object, including visualizing and
interpreting its results.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutoriag.

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
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Loading Data__

We we begin by loading the galaxy dataset `simple` from .fits files, which is the dataset we will use to demonstrate 
ellipse fitting.

This uses the `Imaging` object used in other examples.

Ellipse fitting does not use the Point Spread Function (PSF) of the dataset, so we do not need to load it.
"""
dataset_name = "ellipse"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We can use the `ImagingPlotter` to plot the image and noise-map of the dataset.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True, noise_map=True)

"""
The `ImagingPlotter` also contains a subplot which plots all these properties simultaneously.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We now mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

We use a `Mask2D` object, which for this example is a 3.0" circular mask.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

"""
We now combine the imaging dataset with the mask.
"""
dataset = dataset.apply_mask(mask=mask)

"""
We now plot the image with the mask applied, where the image automatically zooms around the mask to make the galaxy
appear bigger.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.set_title("Image Data With Mask Applied")
dataset_plotter.figures_2d(data=True)

"""
The mask is also used to compute a `Grid2D`, where the (y,x) arc-second coordinates are only computed in unmasked
pixels within the masks' circle.

As shown in the previous overview example, this grid will be used to perform galaxying calculations when fitting the
data below.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.grid)
grid_plotter.set_title("Grid2D of Masked Dataset")
grid_plotter.figure_2d()

"""
__Ellipse Interpolation__

Ellipse fitting performs interpolation calculations which map each data and noise-map value of the dataset
to coordinates on each ellipse we fit to the data.

Interpolation is performed using the `DatasetInterp` object, which is created by simply inputting the dataset.
The object stores in memory the interpolation weights and mappings, ensuring they are performed efficiently.

This object is not passed to the `FitEllipse` object below, but is instead created inside of it to perform the
interpolation. It is included in this example simply so that you are aware that this interpolation is performed.
"""
interp = ag.DatasetInterp(dataset=dataset)

"""
To perform the interpolation we create an `Ellipse` object. 
"""
ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

"""
We can use the `DatasetEllipsePlotter` to plot the ellipse over the dataset.
"""
# dataset_plotter = aplt.DatasetEllipsePlotter(dataset=dataset, ellipse=ellipse)

"""
The ellipse has an attribute `points_from_major_axis` which is a subset of (y,x) coordinates on the ellipse that are
equally spaced along the major-axis. 

The number of points is automatically computed based on the resolution of the data and the size of the ellipse's 
major-axis. 

This value is chosen to ensure that the number of points computed matches the number of pixels in the data
which the ellipse interpolates over. If the ellipse is bigger, the number of points increases in order to
ensure that the ellipse uses more of the data's pixels.

To determine the number of pixels the ellipse's circular radius in units of pixels is required. This is
why `pixel_scale` is an input parameter of this function and other functions in this class.
"""
points_from_major_axis = ellipse.points_from_major_axis_from(
    pixel_scale=dataset.pixel_scales[0]
)

print("Points on Major Axis of Ellipse:")
print(points_from_major_axis)

"""
These are the points which are passed into the `DatasetInterp` object to perform the interpolation.

The output of the code below is therefore the data values of the dataset interpolated to these (y,x) coordinates on
the ellipse.
"""
data_interp = interp.data_interp(points_from_major_axis)

print("Data Values Interpolated to Ellipse:")
print(data_interp)

"""
The same interpolation is performed on the noise-map of the dataset, for example to compute the chi-squared map.
"""
noise_map_interp = interp.noise_map_interp(points_from_major_axis)

print("Noise Values Interpolated to Ellipse:")
print(noise_map_interp)

"""
__Ellipse Fitting__

Ellipse fitting behaves differently to light-profile fitting. In light-profile fitting, a model-image of
the data is created and subtracted from the data pixel-by-pixel to create a residual-map, which is plotted in 2D
in order to show where the model fit the data poorly.

For ellipse fitting, it may be unclear what the `model_data` is, as a model image of the data is not created. 

However, the `model_data` has actually been computed in the interpolation above. The `model_data` is simply the 
data values interpolated to the ellipse's coordinates.
"""
model_data = data_interp

print("Model Data Values:")
print(model_data)

"""
If this is the model data, what is the residual map?

The residual map is each value in the model data minus the mean of the model data. This is because the goodness-of-fit
of an ellipse is quantified by how well the data values trace round the ellipse. A good fit means that all values
on the ellipse are close to the mean of the data and a bad fit means they are not.

The goal of the ellipse fitting is therefore to find the ellipses that trace round the data with values that are
as close to one another as possible.
"""
residual_map = data_interp - np.mean(data_interp)

print("Residuals:")
print(residual_map)

"""
The `normalized_residual_map` and `chi_squared_map` follow the same definition as in light-profile fitting, where:

- Normalized Residuals = (Residual Map) / Noise Map
- Chi-Squared = ((Residuals) / (Noise)) ** 2.0

Where the noise-map is the noise values interpolated to the ellipse.
"""
normalized_residual_map = residual_map / noise_map_interp

print("Normalized Residuals:")
print(normalized_residual_map)

chi_squared_map = (residual_map / noise_map_interp) ** 2.0

print("Chi-Squareds:")
print(chi_squared_map)

"""
Finally, the log likelihood of the fit is computed as:

 - log likelihood = -2.0 * (chi-squared)
 
Note that, unlike light profile fitting, the log likelihood does not include the noise normalization term. This is
because the noise normalization term varies numerically when the data is interpolated to the ellipse, making it
unstable to include in the log likelihood.
"""
log_likelihood = -2.0 * np.sum(chi_squared_map)

print("Log Likelihood:")
print(log_likelihood)

"""
__FitEllipse__

We now use a `FitEllipse` object to fit the ellipse to the dataset, which performs all the calculations we have
discussed above and contains all the quantities we have inspected as attributes.
"""
fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse)

print("Data Values Interpolated to Ellipse:")
print(fit.data_interp)
print("Noise Values Interpolated to Ellipse:")
print(fit.noise_map_interp)
print("Model Data Values:")
print(fit.model_data)
print("Residuals:")
print(fit.residual_map)
print("Normalized Residuals:")
print(fit.normalized_residual_map)
print("Chi-Squareds:")
print(fit.chi_squared_map)
print("Log Likelihood:")
print(fit.log_likelihood)

"""
The `FitEllipse` object can be input into a `FitEllipsePlotter` to plot the results of the fit in 2D on the 
interpolated ellipse coordinates.

The plot below shows in white the ellipse fitted to the data and in black the contour of values in the data that
match the mean of the data over the ellipse. 

A good fit indicates that the white ellipse traces round the black contour well, which is close for the example
below but not perfect.
"""
fit_plotter = aplt.FitEllipsePlotter(
    fit_list=[fit], mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
__Multiple Ellipses__

It is rare to use only one ellipse to fit a galaxy, as the goal of ellipse fitting is to find the collection
of ellipses that best trace round the data.

For example, one model might consist ellipses, which all have the same `centre` and `ell_comps` but have different
`major_axis` values, meaning they grow in size.

We can therefore create multiple ellipses and fit them to the data, for example by creating a list of `FitEllipse`
objects.
"""
major_axis_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

ellipse_list = [
    ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.5), major_axis=major_axis)
    for major_axis in major_axis_list
]

fit_list = [ag.FitEllipse(dataset=dataset, ellipse=ellipse) for ellipse in ellipse_list]

print("Log Likelihoods of Multiple Ellipses:")
print([fit.log_likelihood for fit in fit_list])

print("Overall Log Likelihood:")
print(sum([fit.log_likelihood for fit in fit_list]))

fit_plotter = aplt.FitEllipsePlotter(
    fit_list=fit_list, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
A subplot can be plotted which contains all of the above quantities.
"""
# fit_plotter.subplot_fit()
#
# """
# __Bad Fit__
#
# A bad ellipse fit will occur when the ellipse model does not trace the data well, for example because the input
# angle does not align with the galaxy's elliptical shape.
#
# We can produce such a fit by inputting an ellipse with an angle that is not aligned with the galaxy's elliptical shape.
# """
# ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)
#
# """
# A new fit using this plane shows residuals, normalized residuals and chi-squared which are non-zero.
# """
# fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse)
#
# # fit_plotter = aplt.FitEllipsePlotter(fit=fit)
# # fit_plotter.subplot_fit()
#
# """
# We also note that its likelihood decreases.
# """
# print(fit.log_likelihood)
#
# """
# __Fit Quantities__
#
# The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
#
# There is a `model_image`, which is the image-plane image of the tracer we inspected in the previous tutorial
# blurred with the imaging data's PSF.
#
# This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the
# goodness-of-fit.
#
# If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
# """
# print(fit.model_data.slim)
#
# # The native property provides quantities in 2D NumPy Arrays.
# # print(fit.model_data.native)
#
# """
# There are numerous ndarrays showing the goodness of fit:
#
#  - `residual_map`: Residuals = (Data - Model_Data).
#  - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
#  - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
# """
# print(fit.residual_map.slim)
# print(fit.normalized_residual_map.slim)
# print(fit.chi_squared_map.slim)
#
# """
# __Figures of Merit__
#
# There are single valued floats which quantify the goodness of fit:
#
#  - `chi_squared`: The sum of the `chi_squared_map`.
#
#  - `noise_normalization`: The normalizing noise term in the likelihood function
#     where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).
#
#  - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
# """
# print(fit.chi_squared)
# print(fit.noise_normalization)
# print(fit.log_likelihood)

"""
Fin.
"""
