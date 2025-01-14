"""
Feature: Start Point
====================

For maximum likelihood estimator (MLE) and Markov Chain Monte Carlo (MCMC) non-linear searches, parameter space
sampling is built around having a "location" in parameter space.

This could simply be the parameters of the current maximum likelihood model in an MLE fit, or the locations of many
walkers in parameter space (e.g. MCMC).

For many model-fitting problems, we may have an expectation of where correct solutions lie in parameter space and
therefore want our non-linear search to start near that location of parameter space. Alternatively, we may want to
sample a specific region of parameter space, to determine what solutions look like there.

The start-point API allows us to do this, by manually specifying the start-point of an MLE fit or the start-point of
the walkers in an MCMC fit. Because nested sampling draws from priors, it cannot use the start-point API.

__Comparison to Priors__

Similar behaviour can be achieved by customizing the priors of a model-fit. We could place `GaussianPrior`'s
centred on the regions of parameter space we want to sample, or we could place tight `UniformPrior`'s on regions
of parameter space we believe the correct answer lies.

The downside of using priors is that our priors have a direct influence on the parameters we infer and the size
of the inferred parameter errors. By using priors to control the location of our model-fit, we therefore risk
inferring a non-representative model.

For users more familiar with statistical inference, adjusting ones priors in the way described above leads to
changes in the posterior, which therefore impacts the model inferred.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects you have seen elsewhere in the workspace.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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

"""
__Dataset + Masking__

Load and plot the galaxy dataset `simple__sersic` via .fits files, which we will fit with the model.
"""
dataset_name = "simple__sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__


We compose our model where in this example:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 
"""
bulge = af.Model(ag.lp_linear.Sersic)

"""
__Start Point Priors__

The start-point API does not conflict with the use of priors, which are still associated with every parameter.

We manually customize the priors of the model used by the non-linear search.

We use broad `UniformPriors`'s so that our priors do not impact our inferred model and errors (which would be
the case with tight `GaussianPrior`'s.
"""
bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=6.0)

"""
We can now compose the overall model using a `Collection`, which takes the model components we defined above.
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
We can inspect the model (with customized priors) via its `.info` attribute.
"""
print(model.info)

"""
__Start Point__

We now define the start point of certain parameters in the model:

 - The galaxy is centred near (0.0, 0.0), so we set a start point there.

 - The size of the galaxy is around 1.0" thus we set the `effective_radius` to start here.
 
 - We know this galaxy is an Early-type, thus we set its `sersic_index` to start at 4.0.

For all parameters where the start-point is not specified (in this case the `ell_comps`, their 
parameter values are drawn randomly from the prior when determining the initial locations of the parameters.
"""
initializer = af.InitializerParamBounds(
    {
        model.galaxies.galaxy.bulge.centre_0: (-0.01, 0.01),
        model.galaxies.galaxy.bulge.centre_1: (-0.01, 0.01),
        model.galaxies.galaxy.bulge.effective_radius: (0.9, 1.1),
        model.galaxies.galaxy.bulge.sersic_index: (3.9, 4.1),
    }
)

"""
A quick look at the model's `info` attribute shows that the starting points above do not change
the priors or model info.
"""
print(model.info)

"""
Information on the initializer can be extracted and printed, which is shown below, where the start points are
clearly visible.
"""
print(initializer.info_from_model(model=model))

"""
__Search + Analysis + Model-Fit__

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
search = af.Emcee(
    path_prefix=path.join("imaging", "customize"),
    name="start_point",
    nwalkers=50,
    nsteps=500,
    initializer=initializer,
)


analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can print the initial `parameter_lists` of the result's `Samples` object to check that the initial 
walker samples were set within the start point ranges above.
"""
samples = result.samples

print(samples.model.parameter_names)

print(samples.parameter_lists[0])
print(samples.parameter_lists[1])
print(samples.parameter_lists[2])

"""
Finish.
"""
