"""
Plots: MultiSubPlots
====================

This example illustrates how to plot figures from different plotters on the same subplot, using the example of
combining an `ImagingPlotter` and `FitImagingPlotter`.
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
First, lets load example imaging of of a galaxy as an `Imaging` object.
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
We now mask the data and fit it with a `Plane` to create a `FitImaging` object.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

galaxy = ag.Galaxy(
    redshift=1.0,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

"""
We now pass the imaging to an `ImagingPlotter` and the fit to an `FitImagingPlotter`.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
fit_plotter = aplt.FitImagingPlotter(fit=fit)

"""
We next pair the `MatPlot2D` objects of the two plotters, which ensures the figures plot on the same subplot.
"""
dataset_plotter.mat_plot = fit_plotter.mat_plot_2d

"""
We next open the subplot figure, specifying: 

 - How many subplot figures will be on our image.
 - The shape of the subplot.
 - The figure size of the subplot. 
"""
dataset_plotter.open_subplot_figure(
    number_subplots=5, subplot_shape=(1, 5), subplot_figsize=(18, 3)
)

"""
We now call the `figures_2d` method of all the plots we want to be included on our subplot. These figures will appear
sequencially in the subplot in the order we call them.
"""
dataset_plotter.figures_2d(data=True, signal_to_noise_map=True)
fit_plotter.figures_2d(
    model_image=True, normalized_residual_map=True, chi_squared_map=True
)

"""
This outputs the figure, which in this example goes to your display as we did not specify a file format.
"""
dataset_plotter.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot")

"""
Close the subplot figure, in case we were to make another subplot.
"""
dataset_plotter.close_subplot_figure()
