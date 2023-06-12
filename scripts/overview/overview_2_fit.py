"""
Overview: Fit
-------------

**PyAutoGalaxy** uses `Plane` objects to represent multi-galaxy systems. 

We now use these objects to fit `Imaging` data of a galaxy.

The `autogalaxy_workspace` comes distributed with simulated images of galaxies (an example of how these simulations
are made can be found in the `simulate.py` example, with all simulator scripts located in `autogalaxy_workspac/simulators`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Loading Data__

We we begin by loading the galaxy dataset `simple__sersic` from .fits files, which is the dataset we will use to 
demonstrate fitting.
"""
dataset_name = "simple__sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
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

The Grid2DIterate object. represents a grid of (y,x) coordinates like an ordinary Grid2D, but when the light-profile's 
image is evaluated for the fit the light profile intensity is oteratively increased (in steps of 2, 4, 8, 16, 24) 
until a fractional accuracy of 99.99% is met.

This ensures that the divergent and bright central regions of the galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
dataset = dataset.apply_settings(
    ag.SettingsImaging(
        grid_class=ag.Grid2DIterate,
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16, 24],
    )
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

Following the previous overview, we can make a plane from a collection of `LightProfile` and `Galaxy`
objects.

The combination of `LightProfile`'s below is the same as those used to generate the simulated 
dataset we loaded above.

It therefore produces a plane whose image looks exactly like the dataset. As discussed in the previous overview, this
plane can be extended to include additional `LightProfile`'s`s and `Galaxy``s, for example if you wanted to fit data
with multiple galaxies.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

plane = ag.Plane(galaxies=[galaxy])

plane_plotter = aplt.PlanePlotter(plane=plane, grid=dataset.grid)
plane_plotter.figures_2d(image=True)


"""
We now use the `FitImaging` object to fit this plane to the dataset. 

The fit performs the necessary tasks to create the `model_image` we fit the data with, such as blurring the plane`s 
image with the `Imaging` Point Spread Function (PSF). We can see this by comparing the plane`s image (which isn't PSF 
convolved) and the fit`s model image (which is).

[For those not familiar with Astronomy data, the PSF describes how the observed emission of the galaxy is blurred by
the telescope optics when it is observed. It mimicks this blurring effect via a 2D convolution operation].
"""
fit = ag.FitImaging(dataset=dataset, plane=plane)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d(model_image=True)

"""
The fit creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset`s `image`.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

we'll plot all 3 of these, alongside a subplot containing them all, which also shows the data,
model image and individual galaxies in the fit.

For a good model where the model image and plane are representative of the galaxy system the
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

We can produce such an image by creating a plane with a different galaxy. In the example below, we 
change the centre of the galaxy from (0.0, 0.0) to (0.05, 0.05), which leads to residuals appearing
in the centre of the fit.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.05, 0.05),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

plane = ag.Plane(galaxies=[galaxy])

fit_bad = ag.FitImaging(dataset=dataset, plane=plane)

"""
A new fit using this plane shows residuals, normalized residuals and chi-squared which are non-zero. 
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
__Wrap Up__

A more detailed description of **PyAutoGalaxy**'s fitting methods are given in chapter 1 of the **HowToGalaxy** 
tutorials, which I strongly advise new users check out!
"""
