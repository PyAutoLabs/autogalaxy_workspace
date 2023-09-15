"""
Modeling: Imaging & Interferometer
==================================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.

__Benefits__

A number of benefits are apparent if we combine the analysis of both datasets at both wavelengths:

 - The galaxy appears completely different in the g-band and at sub-millimeter wavelengths, providing a lot
 more information with which to constrain the galaxy structure.
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
import numpy as np

"""
__Interferometer Masking__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0, sub_size=1
)

"""
__Interferometer Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit 
with the galaxy model.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "simple"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

interferometer = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

interferometer_plotter = aplt.InterferometerPlotter(dataset=interferometer)
interferometer_plotter.subplot_dataset()
interferometer_plotter.subplot_dirty_images()

"""
We now create the `Interferometer` object which is used to fit the galaxy model.
"""
settings_interferometer = ag.SettingsInterferometer(
    transformer_class=ag.TransformerNUFFT
)

"""
__Imaging Dataset__

Load and plot the galaxy dataset `simple` via .fits files, which we will fit with the galaxy model.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

imaging = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "g_data.fits"),
    psf_path=path.join(dataset_path, "g_psf.fits"),
    noise_map_path=path.join(dataset_path, "g_noise_map.fits"),
    pixel_scales=0.08,
)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Imaging Masking__

The model-fit requires a `Mask2D` defining the regions of the image we fit the galaxy model to the data, which we define
and use to set up the `Imaging` object that the galaxy model fits.
"""
mask = ag.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Model__

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this 
example our galaxy model is:

 - An `Sersic` `LightProfile` for the galaxy's bulge and disk, which are complete different for each 
 waveband. [28 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=28.
"""
bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Analysis__

We create analysis objects for both datasets.
"""
analysis_imaging = ag.AnalysisImaging(dataset=imaging)
analysis_interferometer = ag.AnalysisInterferometer(dataset=interferometer)

"""
By adding the two analysis objects together, we create an overall `CombinedAnalysis` which we can use to fit the 
multi-wavelength imaging data, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis objects (e.g. the fit to each separate waveband).

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis and therefore each dataset.
 
 - Next, we will use this combined analysis to parameterize a model where certain galaxy parameters vary across
 the dataset.
"""
analysis = analysis_imaging + analysis_interferometer

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
In other scripts in the `multi` package, we made the `intensity` a free parameter for each dataset, motivated by
the notion that a galaxy will not change its appearance significantly across wavelength.

Imaging and interferometer datasets observe completely different properties of the, such that the galaxy appears 
completely different in the imaging data (e.g. optical emission) and sub-millimeter  wavelengths, meaning a completely 
different model should be used for each dataset.
"""
analysis = analysis.with_free_parameters(model.galaxies.galaxy)

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="imaging_and_interferometer",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The galaxy model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Plane` and `FitInterferometer` objects.
 - Information on the posterior as estimated by the `Nautilus` non-linear search.
"""
print(result_list[0].max_log_likelihood_instance)

plane_plotter = aplt.PlanePlotter(
    plane=result_list[0].max_log_likelihood_plane,
    grid=real_space_mask.derive_grid.unmasked_sub_1,
)
plane_plotter.subplot_plane()

fit_plotter = aplt.FitImagingPlotter(fit=result_list[0].max_log_likelihood_fit)
fit_plotter.subplot_fit()

fit_plotter = aplt.FitInterferometerPlotter(fit=result_list[1].max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

search_plotter = aplt.DynestyPlotter(samples=result_list.samples)
search_plotter.cornerplot()

"""
Checkout `autogalaxy_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**.
"""
