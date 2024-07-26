"""
Modeling
========

This guide shows how to perform ellipse fitting modeling on data using a non-linear search, including visualizing and
interpreting its results.

__Fit__

The non-linear search in this example calls a `log_likelihood_function` using the `Analysis` class many times, in
order to determine ellipse parameters and therefore overall distribution of ellipses that best-fit the data.

The `log_likelihood_function` and how the ellipses are used to fit the data are described in the `fit.py` script,
which you should read first in order to better understand how ellipse fitting works.

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
import autofit as af
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

We use a `Mask2D` object, which for this example is 4.0" circular mask.

For ellipse fitting, the mask radius defines the region of the image that the ellipses are fitted over. We therefore
define the `mask_radius` as a variable which is used below to define the sizes of the ellipses in the model fitting.
"""
mask_radius = 4.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
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
__Model Composition__

The API below for composing a model uses the `Model` and `Collection` objects, which are imported from the 
parent project **PyAutoFit** 

The API is fairly self explanatory and is straight forward to extend, for example adding more ellipses
to the galaxy.

The model is composed of 10 ellipses as follows:

1) The ellipses have fixed sizes that are input manually, which incrementally grow in size in order to cover
the entire galaxy.

2) All 10 ellipses have the same centre and elliptical components, meaning that the model have N=4 free parameters.

__Model Cookbook__

A full description of model composition, including model customization, is provided by the model cookbook: 

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html

__Coordinates__

The model fitting default settings assume that the galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the galaxy is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autogalaxy_workspace/*/preprocess`). 
 - Manually override the model priors (`autogalaxy_workspace/*/imaging/modeling/customize/priors.py`).
"""
number_of_ellipses = 10

major_axis_list = np.linspace(0.3, mask_radius, number_of_ellipses)

total_ellipses = len(major_axis_list)

ellipse_list = af.Collection(
    af.Model(ag.Ellipse) for _ in range(total_ellipses)
)


centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

for i, ellipse in enumerate(ellipse_list):

    ellipse.centre.centre_0 = centre_0  # All Gaussians have same y centre.
    ellipse.centre.centre_1 = centre_1  # All Gaussians have same x centre.

    ellipse.ell_comps.ell_comps_0 = ell_comps_0 # All Gaussians have same elliptical components.
    ellipse.ell_comps.ell_comps_1 = ell_comps_1  # All Gaussians have same elliptical components.

    ellipse.major_axis = major_axis_list[i]

model = af.Collection(ellipses=ellipse_list)

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

This example uses the nested sampling algorithm  Dynesty (https://dynesty.readthedocs.io/en/stable/), which extensive 
testing has revealed gives the most accurate and efficient modeling results for ellipse fitting.

Dynesty has one main setting that trades-off accuracy and computational run-time, the number of `live_points`. 
A higher number of live points gives a more accurate result, but increases the run-time. A lower value may give 
less reliable modeling (e.g. the fit may infer a local maxima), but is faster. 

The suitable value depends on the model complexity whereby models with more parameters require more live points. 
The default value of 200 is sufficient for the vast majority of ellipse fitting problems. Lower values often given 
reliable results though, and speed up the run-times. 

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.

An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.

__Parallel Script__

Depending on the operating system (e.g. Linux, Mac, Windows), Python version, if you are running a Jupyter notebook 
and other factors, this script may not run a successful parallel fit (e.g. running the script 
with `number_of_cores` > 1 will produce an error). It is also common for Jupyter notebooks to not run in parallel 
correctly, requiring a Python script to be run, often from a command line terminal.

To fix these issues, the Python script needs to be adapted to use an `if __name__ == "__main__":` API, as this allows
the Python `multiprocessing` module to allocate threads and jobs correctly. An adaptation of this example script 
is provided at `autolens_workspace/scripts/imaging/modeling/customize/parallel.py`, which will hopefully run 
successfully in parallel on your computer!

Therefore if paralellization for this script doesn't work, check out the `parallel.py` example. You will need to update
all scripts you run to use the this format and API. 

__Iterations Per Update__

Every N iterations, the non-linear search outputs the current results to the folder `autogalaxy_workspace/output`,
which includes producing visualization. 

Depending on how long it takes for the model to be fitted to the data (see discussion about run times below), 
this can take up a large fraction of the run-time of the non-linear search.

For this fit, the fit is very fast, thus we set a high value of `iterations_per_update=10000` to ensure these updates
so not slow down the overall speed of the model-fit. 
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="ellipse_fitting",
    unique_tag=dataset_name,
    sample="rwalk",
    n_live=200,
    number_of_cores=4,
    iterations_per_update=10000,
)

"""
__Analysis__

We next create an `AnalysisEllipse` object, which can be given many inputs customizing how the model is fitted to the 
data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autogalaxy_workspace/*/imaging/log_likelihood_function`
"""
analysis = ag.AnalysisEllipse(dataset=dataset)

"""
__Run Times__

Modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the model to be fitted to 
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex
   models require more iterations to converge to a solution.

The log likelihood evaluation time can be estimated before a fit using the `profile_log_likelihood_function` method,
which returns two dictionaries containing the run-times and information about the fit.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

"""
The overall log likelihood evaluation time is given by the `fit_time` key.

For this example, it is ~0.04 seconds, which is extremely fast for modeling. For higher resolution datasets ellipse
fitting can slow down to a likelihood evaluation time of order 0.5 - 1.0 second, which is still reasonably fast.
"""
print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
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
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
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
 - The corresponding maximum log likelihood `Ellipse` and `FitEllipse` objects.
"""
instance = result.max_log_likelihood_instance

print("Max Log Likelihood Model:")
print(instance)

print(f"First Ellipse Centre: {instance.ellipses[0].centre}")
print(f"First Ellipse Elliptical Components: {instance.ellipses[0].ell_comps}")
print(f"First Ellipse Major Axis: {instance.ellipses[0].major_axis}")
print(f"First Ellipse Axis Ratio: {instance.ellipses[0].axis_ratio}")
print(f"First Ellipse Angle: {instance.ellipses[0].angle}")

for i, ellipse in enumerate(result.max_log_likelihood_instance.ellipses):

    print(f"Ellipse {i} Minor Axis: {ellipse.minor_axis}")

"""
The maximum log likelihood fit is also available via the result, which can visualize the fit.
"""
fit_plotter = aplt.FitEllipsePlotter(fit_list=result.max_log_likelihood_fit_list, mat_plot_2d=aplt.MatPlot2D(use_log10=True))
fit_plotter.figures_2d(data=True)

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
This script gives a concise overview of the ellipse fitting modeling API, fitting one the simplest models possible.
So, what next? 

__Data Preparation__

If you are looking to fit your own CCD imaging data of a galaxy, checkout  
the `autogalaxy_workspace/*/imaging/data_preparation/start_here.ipynb` script for an overview of how data should be 
prepared before being modeled.

__HowToGalaxy__

This example script above explains ellipse fitting, but there are many other ways to model a galaxy, using
light profiles which represent its surface brightness. 

This is explained in the **HowToGalaxy** Jupyter notebook lectures, found at `autogalaxy_workspace/*/howtogalaxy`. 

I recommend that you check them out if you are interested in more details!
"""
