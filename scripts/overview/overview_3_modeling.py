"""
Overview: Modeling
------------------

Modeling is the process of taking data of a galaxy (e.g. imaging data from the Hubble Space Telescope or interferometer
data from ALMA) and fitting it with a model, to determine the `LightProfile`'s that best represent the observed galaxy.

Modeling uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import **PyAutoFit** separately to **PyAutoGalaxy**
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

import autofit as af

"""
__Dataset__

In this example, we fit simulated imaging of a galaxy. 

First, lets load this imaging dataset and plot it.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

We next mask the dataset, to remove the exterior regions of the image that do not contain emission from the galaxy.

Note how when we plot the `Imaging` below, the figure now zooms into the masked region.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot()

"""
__Model__

We compose the model that we fit to the data using PyAutoFit `Model` objects. 

These behave analogously to `Galaxy` objects but their  `LightProfile` parameters are not specified and are instead 
determined by a fitting procedure.

In this example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's disk is a parametric `Exponential` disk [6 parameters].
 
Note how we can easily extend the model below to include extra light profiles in the galaxy.
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential)

"""
The `info` attribute of the galaxy `Model` component shows the model in a readable format.
"""
print(galaxy.info)

"""
We put the model galaxy above into a `Collection`, which is the model we will fit. Note how we could easily 
extend this object to compose complex models containing many galaxies.

The reason we create separate `Collection`'s for the `galaxies` and `model` is so that the `model`
can be extended to include other components than just galaxies.
"""
galaxies = af.Collection(galaxy=galaxy)
model = af.Collection(galaxies=galaxies)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Non-linear Search__

We now choose the non-linear search, which is the fitting method used to determine the set of `LightProfile` (e.g.
bulge and disk) parameters that best-fit our data.

In this example we use `nautilus` (https://github.com/joshspeagle/nautilus), a nested sampling algorithm that is
very effective at modeling.

**PyAutoGalaxy** supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.

The `path_prefix` and `name` determine the output folders the results are written too on hard-disk.
"""
search = af.Nautilus(path_prefix="overview", name="modeling")

"""
__Analysis__

We next create an `AnalysisImaging` object, which contains the `log likelihood function` that the non-linear search 
calls to fit the model to the data.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Run Times__

modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the model to be fitted to 
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.

The log likelihood evaluation time can be estimated before a fit using the `profile_log_likelihood_function` method,
which returns two dictionaries containing the run-times and information about the fit.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

"""
The overall log likelihood evaluation time is given by the `fit_time` key.

For this example, it is ~0.01 seconds, which is extremely fast for modeling. More advanced lens
modeling features (e.g. shapelets, multi Gaussian expansions, pixelizations) have slower log likelihood evaluation
times (1-3 seconds), and you should be wary of this when using these features.

Feel free to go ahead a print the full `run_time_dict` and `info_dict` to see the other information they contain. The
former has a break-down of the run-time of every individual function call in the log likelihood function, whereas the 
latter stores information about the data which drives the run-time (e.g. number of image-pixels in the mask, the
shape of the PSF, etc.).
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

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
nautilus samples, model parameters, visualization) to hard-disk.

Once running you should checkout the `autogalaxy_workspace/output` folder, which is where the results of the search are 
written to hard-disk (in the `overview_modeling` folder) on-the-fly. This includes model parameter estimates with 
errors non-linear samples and the visualization of the best-fit model inferred by the search so far. 
"""
result = search.fit(model=model, analysis=analysis)

"""
__Results__

Whilst navigating the output folder, you may of noted the results were contained in a folder that appears as a random
collection of characters. 

This is the model-fit's unique identifier, which is generated based on the model, search and dataset used by the fit. 
Fitting an identical model, search and dataset will generate the same identifier, meaning that rerunning the script 
will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset, a new 
unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.

The fit above returns a `Result` object, which includes lots of information on the model. 

The `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
Below, 
we print the maximum log likelihood bulge and disk models inferred.
"""
print(result.max_log_likelihood_instance.galaxies.galaxy.bulge)
print(result.max_log_likelihood_instance.galaxies.galaxy.disk)

"""
In fact, the result contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the model. **PyAutoGalaxy** includes
visualization tools for plotting this.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
search_plotter = aplt.NautilusPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
The result also contains the maximum log likelihood `Galaxies` and `FitImaging` objects which can easily be plotted.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=dataset.grid
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
A full guide of result objects is contained in the `autogalaxy_workspace/*/imaging/results` package.

__Model Customization__

The `Model` can be fully customized, making it simple to parameterize and fit many different models
using any combination of light profiles and galaxies:
"""
galaxy_model = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp.DevVaucouleurs,
    disk=ag.lp.Sersic,
    bar=ag.lp.Gaussian,
    clump_0=ag.lp.ElsonFreeFall,
    clump_1=ag.lp.ElsonFreeFall,
)

"""
This aligns the bulge and disk centres in the galaxy model, reducing the
number of free parameter fitted for by Nautilus by 2.
"""
galaxy_model.bulge.centre = galaxy_model.disk.centre

"""
This fixes the galaxy bulge light profile's effective radius to a value of
0.8 arc-seconds, removing another free parameter.
"""
galaxy_model.bulge.effective_radius = 0.8

"""
This forces the light profile disk's effective radius to be above 3.0.
"""
galaxy_model.bulge.add_assertion(galaxy_model.disk.effective_radius > 3.0)

"""
The `info` attribute shows the customized model.
"""
print(galaxy_model.info)

"""
__Linear Light Profiles__

**PyAutoGalaxy** supports 'linear light profiles', where the `intensity` parameters of all parametric components are 
solved via linear algebra every time the model is fitted using a process called an inversion. This inversion always 
computes `intensity` values that give the best fit to the data (e.g. they maximize the likelihood) given the other 
parameter values of the light profile.

The `intensity` parameter of each light profile is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in the example below by 3) and removing 
the degeneracies that occur between the `intnensity` and other light profile
parameters (e.g. `effective_radius`, `sersic_index`).

For complex models, linear light profiles are a powerful way to simplify the parameter space to ensure the best-fit
model is inferred.
"""
sersic_linear = ag.lp_linear.Sersic()

galaxy_model_linear = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.DevVaucouleurs,
    disk=ag.lp_linear.Sersic,
    bar=ag.lp_linear.Gaussian,
)

print(galaxy_model_linear.info)

"""
__Basis Functions__

A natural extension of linear light profiles are basis functions, which group many linear light profiles together in
order to capture complex and irregular structures in a galaxy's emission. 

Using a clever model parameterization a basis can be composed which corresponds to just N = 5-10 parameters, making
model-fitting efficient and robust.

Below, we compose a basis of 10 Gaussians which all share the same `centre` and `ell_comps`. Their `sigma`
values are set via the relation `y = a + (log10(i+1) + b)`, where `i` is the  Gaussian index and `a` and `b` are free 
parameters.

Because `a` and `b` are free parameters (as opposed to `sigma` which can assume many values), we are able to 
compose and fit `Basis` objects which can capture very complex light distributions with just N = 5-10 non-linear 
parameters!
"""
bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussians_bulge = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussians_bulge):
    gaussian.centre = gaussians_bulge[0].centre
    gaussian.ell_comps = gaussians_bulge[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))

bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_bulge,
)

print(bulge.info)

"""
**PyAutoGalaxy** can also apply Bayesian regularization to Basis functions, which smooths the linear light profiles
(e.g. the Gaussians) in order to prevent over-fitting noise.
"""
bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_bulge,
    regularization=ag.reg.Constant,
)

"""
__Wrap Up__

A more detailed description of modeling's is given in chapter 2 of the **HowToGalaxy** 
tutorials, which I strongly advise new users check out!
"""
