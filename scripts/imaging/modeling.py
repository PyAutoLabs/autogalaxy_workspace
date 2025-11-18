"""
Modeling: Start Here
====================

This script is the starting point for modeling of CCD imaging data (E.g. Hubble Space Telescope, Euclid) with
**PyAutoGalaxy** and it provides an overview of the modeling API.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing lens
modeling in different ways and customizing the analysis.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

__Plotters__

To produce images of the data `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of galaxies.

The `PLotter` API is described in the script `autogalaxy_workspace/*/guides/plot`.

__Simulation__

This script fits a simulated `Imaging` dataset of a galaxy, which is produced in the
script `autogalaxy_workspace/*/simulators/imaging/start_here.py`

__Data Preparation__

The `Imaging` dataset fitted in this example confirms to a number of standard that make it suitable to be fitted in
**PyAutoGalaxy**.

If you are intending to fit your own data, you will need to ensure it conforms to these standards, which are
described in the script `autogalaxy_workspace/*/data_preparation/imaging/start_here.ipynb`.
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
__Dataset__

Load the strong dataset `simple` via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
Use an `ImagingPlotter` the plot the data, including: 

 - `data`: The image of the strong lens.
 - `noise_map`: The noise-map of the image, which quantifies the noise in every pixel as their RMS values.
 - `psf`: The point spread function of the image, which describes the blurring of the image by the telescope optics.
 - `signal_to_noise_map`: Quantifies the signal-to-noise in every pixel.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data. 

We create a 3.0 arcsecond circular mask and apply it to the `Imaging` object that the model fits.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
If we plot the masked data, the mask removes the exterior regions of the image where there is no emission from the 
galaxy.

The mask used to fit the data can be customized, as described in 
the script `autogalaxy_workspace/*/modeling/imaging/customize/custom_mask.py`
"""
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
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
The imaging subplot updates the bottom two panels to reflect the update to over sampling, which now uses a higher
values in the centre.

Whilst you may not yet understand the details of over-sampling, you can at least track it visually in the plots
and later learnt more about it in the `over_sampling.ipynb` guide.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

In this example we compose a model where:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 

 - The galaxy's disk is a linear parametric `Exponential` disk, whose centre is aligned with the bulge [3 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

__Linear Light Profiles__

The model below uses a `linear light profile` for the bulge and disk, via the API `lp_linear`. This is a specific type 
of light profile that solves for the `intensity` of each profile that best fits the data via a linear inversion. 
This means it is not a free parameter, reducing the dimensionality of non-linear parameter space. 

Linear light profiles significantly improve the speed, accuracy and reliability of modeling and they are used
by default in every modeling example. A full description of linear light profiles is provided in the
`autogalaxy_workspace/*/modeling/imaging/features/linear_light_profiles.py` example.

A standard light profile can be used if you change the `lp_linear` to `lp`, but it is not recommended.

__Model Composition__

The API below for composing a model uses the `Model` and `Collection` objects, which are imported from 
**PyAutoGalaxy**'s parent project **PyAutoFit** 

The API is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the galaxy.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html

__Coordinates__

The model fitting default settings assume that the galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the galaxy is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autogalaxy_workspace/*/preprocess`). 
 - Manually override the model priors (`autogalaxy_workspace/*/modeling/imaging/customize/priors.py`).
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/generag.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. 

All examples in the autogalaxy workspace use the nested sampling algorithm 
Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/), which extensive testing has revealed gives the most 
accurate and efficient modeling results.

Nautilus has one main setting that trades-off accuracy and computational run-time, the number of `live_points`. 
A higher number of live points gives a more accurate result, but increases the run-time. A lower value may give 
less reliable modeling (e.g. the fit may infer a local maxima), but is faster. 

The suitable value depends on the model complexity whereby models with more parameters require more live points. 
The default value of 200 is sufficient for the vast majority of common galaxy models. Lower values often given reliable
results though, and speed up the run-times. In this example, given the model is quite simple (N=21 parameters), we 
reduce the number of live points to 100 to speed up the run-time.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Iterations Per Update__

Every N iterations, the non-linear search outputs the current results to the folder `autogalaxy_workspace/output`,
which includes producing visualization. 

Depending on how long it takes for the model to be fitted to the data (see discussion about run times below), 
this can take up a large fraction of the run-time of the non-linear search.

For this fit, the fit is very fast, thus we set a high value of `iterations_per_quick_update=10000` to ensure these updates
so not slow down the overall speed of the model-fit. 
# 
**If the iteration per update is too low, the model-fit may be significantly slowed down by the time it takes to
output results and visualization frequently to hard-disk. If your fit is consistent displaying a log saying that it
is outputting results, try increasing this value to ensure the model-fit runs efficiently.**
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "modeling",
    name="start_here",
    unique_tag=dataset_name,
    n_live=200,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

We next create an `AnalysisImaging` object, which can be given many inputs customizing how the model is fitted to the 
data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autogalaxy_workspace/*/imaging/log_likelihood_function`

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Times__

Modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the model to be fitted to 
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex
   models require more iterations to converge to a solution.

For this analysis, the log likelihood evaluation time is ~0.03 seconds, which is extremely fast for modeling. More advanced
modeling features (e.g. shapelets, multi Gaussian expansions, pixelizations) have slower log likelihood evaluation
times (1-3 seconds), and you should be wary of this when using these features.PointDataset

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

Estimating this quantity is more tricky, as it varies depending on the model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted.

For this example, we conservatively estimate that the non-linear search will perform ~10000 iterations per free 
parameter in the model. This is an upper limit, with models typically converging in far fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. Parallelization with Nautilus scales well, it speeds up the model-fit by the 
`number_of_cores` for N < 8 CPUs and roughly `0.5*number_of_cores` for N > 8 CPUs. This scaling continues 
for N> 50 CPUs, meaning that with super computing facilities you can always achieve fast run times!

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Output Folder__

Now this is running you should checkout the `autogalaxy_workspace/output` folder. This is where the results of the 
search are written to hard-disk (in the `start_here` folder), where all outputs are human readable (e.g. as .json,
.csv or text files).

As the fit progresses, results are written to the `output` folder on the fly using the highest likelihood model found
by the non-linear search so far. This means you can inspect the results of the model-fit as it runs, without having to
wait for the non-linear search to terminate.
 
The `output` folder includes:

 - `model.info`: Summarizes the model, its parameters and their priors discussed in the next tutorial.
 
 - `model.results`: Summarizes the highest likelihood model inferred so far including errors.
 
 - `images`: Visualization of the highest likelihood model-fit to the dataset, (e.g. a fit subplot showing the 
 galaxies, model data and residuals).
 
 - `files`: A folder containing .fits files of the dataset, the model as a human-readable .json file, 
 a `.csv` table of every non-linear search sample and other files containing information about the model-fit.
 
 - search.summary: A file providing summary statistics on the performance of the non-linear search.
 
 - `search_internal`: Internal files of the non-linear search (in this case Nautilus) used for resuming the fit and
  visualizing the search.

__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Galaxies` and `FitImaging` objects.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
The result contains the full posterior information of our non-linear search, including all parameter samples, 
log likelihood values and tools to compute the errors on the model. 

There are built in visualization tools for plotting this.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()


"""
This script gives a concise overview of the PyAutoGalaxy modeling API, fitting one the simplest models possible.
So, what next? 

__Features__

The examples in the `autogalaxy_workspace/*/modeling/imaging/features` package illustrate other modeling features. 

We recommend you checkout the following features, because the make modeling in general more reliable and 
efficient (you will therefore benefit from using these features irrespective of the quality of your data and 
scientific topic of study).

We recommend you now checkout the following feature:

- ``linear_light_profiles``: The model light profiles use linear algebra to solve for their intensity, reducing model complexity.

All other features may be useful to specific users with specific datasets and scientific goals, but are not useful
for general modeling.

The folders `autogalaxy_workspace/*/modeling/imaging/searches` and `autogalaxy_workspace/*/modeling/imaging/customize`
provide guides on how to customize many other aspects of the model-fit. Check them out to see if anything
sounds useful, but for most users you can get by without using these forms of customization!
  
__Data Preparation__

If you are looking to fit your own CCD imaging data of a galaxy, checkout  
the `autogalaxy_workspace/*/data_preparation/imaging/start_here.ipynb` script for an overview of how data should be 
prepared before being modeled.

__HowToGalaxy__

This `start_here.py` script, and the features examples above, do not explain many details of how modeling is 
performed, for example:

 - How does PyAutoGalaxy perform calculations which determine how a gaalxy emits light and therefore fit a model?
 - How is the model fitted to data? What quantifies the goodness of fit (e.g. how is a log likelihood computed?).
 - How does Nautilus find the highest likelihood models? What exactly is a "non-linear search"?

You do not need to be able to answer these questions in order to fit models with PyAutoGalaxy and do science.
However, having a deeper understanding of how it all works is both interesting and will benefit you as a scientist

This deeper insight is offered by the **HowToGalaxy** Jupyter notebook lectures, found 
at `autogalaxy_workspace/*/howtogalaxy`. 

I recommend that you check them out if you are interested in more details!
"""
