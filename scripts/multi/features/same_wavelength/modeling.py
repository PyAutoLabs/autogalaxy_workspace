"""
Modeling: Same Wavelength
=========================

This script fits a multiple `Imaging` datasets observed at the same wavelength of a galaxy with a
model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

This script demonstrates how PyAutoGalaxy's multi-dataset modeling tools can also simultaneously analyse datasets
observed at the same wavelength.

An example use case might be analysing undithered HST images before they are combined via the multidrizzing process,
to remove correlated noise in the data.

It is common for dithered datasets to be observed with small shifts (e.g. half a pixel), which needs to be
modelled in the analysis. The example `multi/modeling/features/dataset_offset.py` demonstrates how to do this.

This is an advanced script and assumes previous knowledge of the core **PyAutoGalaxy** API for galaxy modeling. Thus,
certain parts of code are not documented to ensure the script is concise.
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

"""
__Pixel Scales__

If observed at the same wavelength, it is likely the datasets have the same pixel-scale.

Nevertheless, we specify this as a list as there could be an exception.
"""
pixel_scales_list = [0.1, 0.1]

"""
__Dataset__

Load and plot each multi-wavelength galaxy dataset, using a list of their waveband colors.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "same_wavelength"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    ag.Imaging.from_fits(
        data_path=Path(dataset_path, f"image_{i}.fits"),
        psf_path=Path(dataset_path, f"psf_{i}.fits"),
        noise_map_path=Path(dataset_path, f"noise_map_{i}.fits"),
        pixel_scales=pixel_scales,
    )
    for i, pixel_scales in enumerate(pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the galaxy model to the data, which we define
and use to set up the `Imaging` object that the galaxy model fits.

For multi-wavelength galaxy modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_list = [
    ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]


dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Analysis__

We create an `Analysis` object for every dataset.
"""
analysis_list = [
    ag.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
]

"""
__Model__

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a galaxy model where:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 

 - The galaxy's disk is a linear parametric `Exponential`disk [5 parameters]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
We now combine them using the factor analysis class, which allows us to fit the two datasets simultaneously.

Unlike other examples, no customization to the model is applied that, for example, adds more free parameters,
given that the datasets do not vary over wavelength.
"""
analysis_factor_list = []

for analysis in analysis_list:

    model_analysis = model.copy()

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The `info` of the model shows us there are two models each with linear light profiles.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=Path("multi") / "features",
    name="same_wavelength",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a factor graph.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's galaxies shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    galaxies_plotter = aplt.GalaxiesPlotter(
        galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
    )
    galaxies_plotter.subplot_galaxies()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
"""
plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/results` for a full description of analysing results in **PyAutoGalaxy**.
"""
