"""
Modeling: Light Parametric
==========================

This script fits `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.
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
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple__sersic` from .fits files, which we will fit 
with the model.

This includes the method used to Fourier transform the real-space image of the galaxy to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.
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

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used, 
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk, the centres of 
 which are aligned [10 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

__Linear Light Profiles__

The model below uses a `linear light profile` for the bulge and disk, via the API `lp_linear`. This is a specific type 
of light profile that solves for the `intensity` of each profile that best fits the data via a linear inversion. 
This means it is not a free parameter, reducing the dimensionality of non-linear parameter space. 

Linear light profiles significantly improve the speed, accuracy and reliability of modeling and they are used
by default in every modeling example. A full description of linear light profiles is provided in the
`autogalaxy_workspace/*/modeling/imaging/features/linear_light_profiles.py` example.

A standard light profile can be used if you change the `lp_linear` to `lp`, but it is not recommended.

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

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/).

The folders: 

 - `autogalaxy_workspace/*/modeling/imaging/searches`.
 - `autogalaxy_workspace/*/modeling/imaging/customize`
  
Give overviews of the  non-linear searches **PyAutoGalaxy** supports and more details on how to customize the
model-fit, including the priors on the model. 

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autogalaxy_workspace/output/imaging/simple__sersic/mass[sie]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer", "modeling"),
    name="start_here",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `Interferometer`dataset.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=True)

"""
__VRAM Use__

When running AutoLens with JAX on a GPU, the analysis must fit within the GPU’s available VRAM. If insufficient 
VRAM is available, the analysis will fail with an out-of-memory error, typically during JIT compilation or the 
first likelihood call.

Two factors dictate the VRAM usage of an analysis:

- The number of arrays and other data structures JAX must store in VRAM to fit the model
  to the data in the likelihood function. This is dictated by the model complexity and dataset size.

- The `batch_size` sets how many likelihood evaluations are performed simultaneously.
  Increasing the batch size increases VRAM usage but can reduce overall run time,
  while decreasing it lowers VRAM usage at the cost of slower execution.

Before running an analysis, users should check that the estimated VRAM usage for the
chosen batch size is comfortably below their GPU’s total VRAM.

The method below prints the VRAM usage estimate for the analysis and model with the specified batch size,
it takes about 20-30 seconds to run so you may want to comment it out once you are familiar with your GPU's VRAM limits.

For a MGE model with the low visibility dataset fitted in this example VRAM use is relatively low (~0.3GB) For other 
models (e.g. pixelized sources) and datasets with more visibilities it can be much higher (> 1GB going beyond 10GB).
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Times__

Modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to 
   the dataset such that a log likelihood is returned.
 
 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.
   
For this analysis, the log likelihood evaluation time is ~0.01 seconds on CPU, < 0.001 seconds on GPU, which is 
extremely fast for lens modeling. 

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. For this model, this is typically around
? iterations, meaning that this script takes ? on CPU and ? on GPU.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs the 
Nautilus non-linear search in order to find which models fit the data with the highest likelihood.

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
    galaxies=result.max_log_likelihood_galaxies,
    grid=real_space_mask.derive_grid.unmasked,
)
galaxies_plotter.subplot()
fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
The result contains the full posterior information of our non-linear search, including all parameter samples, 
log likelihood values and tools to compute the errors on the lens model. 

There are built in visualization tools for plotting this.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/interferometer/modeling/results.py` for a full description of the result object.
"""
