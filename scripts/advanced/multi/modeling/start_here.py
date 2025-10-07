"""
Modeling: Multi Modeling
========================

This script fits multiple multi-wavelength `Imaging` datasets of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

Two images are fitted, corresponding to a greener ('g' band) and redder image (`r` band).

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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Load and plot each multi-wavelength galaxy dataset, using a list of their waveband colors.

Note how the disk appears brighter in the g-band image, whereas the bulge is clearer in the r-band image.
Multi-wavelength image can therefore better decompose the structure of galaxies.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    ag.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{color}_data.fits",
        psf_path=Path(dataset_path) / f"{color}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{color}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the galaxy model to the data, which we define
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
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

For a new user, the details of over-sampling are not important, therefore just be aware that below we make it so that 
all calculations use an adaptive over sampling scheme which ensures high accuracy and precision.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
for dataset in dataset_list:
    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a galaxy model where:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 

 - The galaxy's disk is a linear parametric `Exponential` disk [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.

__Model Extension__

Galaxies change appearance across wavelength, for example their size.

Models applied to combined analyses can be extended to include free parameters specific to each dataset. In this example,
we want the galaxy's effective radii to vary across the g and r-band datasets, which will be illustrated below.

__Linear Light Profiles__

As an advanced user you should be familiar wiht linear light profiles, see elsewhere in the workspace for informaiton
if not.

For multi wavelength dataset modeling, the `lp_linear` API is extremely powerful as the `intensity` varies across
the datasets, meaning that making it linear reduces the dimensionality of parameter space significantly.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Analysis List__

Set up two instances of the `Analysis` class object, one for each dataset.
"""
analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
__Analysis Factor__

Each analysis object is wrapped in an `AnalysisFactor`, which pairs it with the model and prepares it for use in a 
factor graph. This step allows us to flexibly define how each dataset relates to the model.

The term "Factor" comes from factor graphs, a type of probabilistic graphical model. In this context, each factor 
represents the connection between one dataset and the shared model.

The API for extending the model across datasets is shown below, by overwriting the `effective_radius`
variables of the model passed to each `AnalysisFactor` object with new priors, making each dataset have its own
`effective_radius` free parameter.

NOTE: Other aspects of galaxies may vary across wavelength, none of which are included in this example. The API below 
can easily be extended to include these additional parameters, and the `features` package explains other tools for 
extending the model across datasets.
"""
analysis_factor_list = []

for analysis in analysis_list:

    model_analysis = model.copy()
    model_analysis.galaxies.galaxy.bulge.effective_radius = af.UniformPrior(
        lower_limit=0.0, upper_limit=10.0
    )

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

All `AnalysisFactor` objects are combined into a `FactorGraphModel`, which represents a global model fit to 
multiple datasets using a graphical model structure.

The key outcomes of this setup are:

 - The individual log likelihoods from each `Analysis` object are summed to form the total log likelihood 
   evaluated during the model-fitting process.
   
 - Results from all datasets are output to a unified directory, with subdirectories for visualizations 
   from each analysis object, as defined by their `visualize` methods.

This is a basic use of **PyAutoFit**'s graphical modeling capabilities, which support advanced hierarchical 
and probabilistic modeling for large, multi-dataset analyses.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
To inspect this new model, with extra parameters for each dataset created, we 
print `factor_graph.global_prior_model.info`.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

print(factor_graph.global_prior_model.info)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("multi") / "modeling",
    name="start_here",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__

To fit multiple datasets, we pass the `FactorGraphModel` to a non-linear search.

Unlike single-dataset fitting, we now pass the `factor_graph.global_prior_model` as the model and 
the `factor_graph` itself as the analysis object.

This structure enables simultaneous fitting of multiple datasets in a consistent and scalable way.
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.

For example, close inspection of the `max_log_likelihood_instance` of the two results shows that all parameters,
except the `effective_radius` of the source galaxy's `bulge`, are identical.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's galaxies shows that the galaxy appears different, owning to its different intensities.
"""
for result in result_list:
    galaxies_plotter = aplt.GalaxiesPlotter(
        galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
    )
    galaxies_plotter.subplot_galaxies()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=16). 

Therefore, the samples is identical in every result object.
"""
plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_cornerpy()

"""
__Wrap Up__

This simple example introduces the API for fitting multiple datasets with a shared model.

It should already be quite intuitive how this API can be adapted to fit more complex models, or fit different
datasets with different models. For example, an `AnalysisImaging` and `AnalysisInterferometer` can be combined, into
a single factor graph model, to simultaneously fit a imaging and interferometric data.

The `advanced/multi/modeling` package has more examples of how to fit multiple datasets with different models,
including relational models that vary parameters across datasets as a function of wavelength.
"""
