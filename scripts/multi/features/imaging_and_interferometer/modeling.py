"""
Modeling: Imaging & Interferometer
==================================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

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

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt
import numpy as np

"""
__Interferometer Masking__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

"""
__Interferometer Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit 
with the galaxy model.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "simple"
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

interferometer = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

interferometer_plotter = aplt.InterferometerPlotter(dataset=interferometer)
interferometer_plotter.subplot_dataset()
interferometer_plotter.subplot_dirty_images()


"""
__Imaging Dataset__

Load and plot the galaxy dataset `simple` via .fits files, which we will fit with the galaxy model.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

imaging = ag.Imaging.from_fits(
    data_path=Path(dataset_path, "g_data.fits"),
    psf_path=Path(dataset_path, "g_psf.fits"),
    noise_map_path=Path(dataset_path, "g_noise_map.fits"),
    pixel_scales=0.08,
)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Imaging Masking__

The model-fit requires a 2D mask defining the regions of the image we fit the galaxy model to the data, which we define
and use to set up the `Imaging` object that the galaxy model fits.
"""
mask = ag.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()


"""
__Analysis__

We create analysis objects for both datasets.
"""
analysis_imaging = ag.AnalysisImaging(dataset=imaging, use_jax=True)
analysis_interferometer = ag.AnalysisInterferometer(
    dataset=interferometer, use_jax=True
)

"""
We now combine them using the factor analysis class, which allows us to fit the two datasets simultaneously.

Imaging and interferometer datasets observe completely different properties of the, such that the galaxy appears 
completely different in the imaging data (e.g. optical emission) and sub-millimeter wavelengths, meaning a completely 
different model should be used for each dataset.

For this reason, we move all model composition to the `AnalysisFactor` class, which allows us to fit the two datasets
simultaneously but with different models.

There is actually no benefit to fitting both simultaneously when the model for each fit is completely different, 
so this is simply an illustration of how to combine two different datasets. However, if you do this combination
of datasets you should not do them simultaneously unless you update the model to link them together.
"""
analysis_factor_list = []

for analysis in [analysis_imaging, analysis_interferometer]:

    bulge = af.Model(ag.lp_linear.Sersic)
    disk = af.Model(ag.lp_linear.Exponential)

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model_analysis = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The `info` of the model shows us there are two models, one for the imaging dataset and one for the interferometer
dataset. 
"""
print(factor_graph.global_prior_model.info)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("multi") / "features",
    name="imaging_and_interferometer",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The search returns a result object, which includes: 

 - The galaxy model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Galaxies` and `FitInterferometer` objects.
 - Information on the posterior as estimated by the `Nautilus` non-linear search.
"""
print(result_list[0].max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result_list[0].max_log_likelihood_galaxies,
    grid=real_space_mask.derive_grid.unmasked,
)
galaxies_plotter.subplot_galaxies()

fit_plotter = aplt.FitImagingPlotter(fit=result_list[0].max_log_likelihood_fit)
fit_plotter.subplot_fit()

fit_plotter = aplt.FitInterferometerPlotter(fit=result_list[1].max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/results` for a full description of analysing results in **PyAutoGalaxy**.
"""
