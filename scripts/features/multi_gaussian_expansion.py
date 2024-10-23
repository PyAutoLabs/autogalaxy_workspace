"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the galaxy light into ~15-100 Gaussians, where
the `intensity` of every Gaussian is solved for via a linear algebra using a process called an "inversion"
(see the `light_parametric_linear.py` feature for a full description of this).

This script fits a light model which uses an MGE consisting of 60 Gaussians. It is fitted to simulated data
where the galaxy's light has asymmetric and irregular features, which fitted poorly by symmetric light
profiles like the `Sersic`.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
An MGE fully captures these features and can therefore much better represent the emission of complex galaxies.

The MGE model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this example,
two separate groups of Gaussians are used to represent the `bulge` and `disk` of the lens, which in total correspond
to just N=6 non-linear parameters (a `bulge` and `disk` comprising two linear Sersics has N=10 parameters).

The MGE model parameterization is also composed such that neither the `intensity` parameters or any of the
parameters controlling the size of the Gaussians (their `sigma` values) are non-linear parameters sampled by Nautilus.
This removes the most significant degeneracies in parameter space, making the model much more reliable and efficient
to fit.

Therefore, not only does an MGE fit more complex galaxy morphologies, it does so using fewer non-linear parameters
in a much simpler non-linear parameter space which has far less significant parameter degeneracies!

__Disadvantages__

To fit an MGE model to the data, the light of the ~15-75 or more Gaussian in the MGE must be evaluated and compared
to the data. This is slower than evaluating the light of ~2-3 Sersic profiles, producing slower computational run
times (although the simpler non-linear parameter space will speed up the fit overall).

For many science cases, the MGE can also be a less intuitive model to interpret than a Sersic profile. For example,
it is straight forward to understand how the effective radius of a Sersic profile relates to a galaxy's size,
or the serisc index relates to its concentration. The results of an MGE are less intuitive, and require more
thought to interpret physically.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysicag. For an MGE, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysicag.

**PyAutoGalaxys** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physicag.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's bulge is a super position of `Gaussian`` profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Dataset__

Load and plot the galaxy dataset `light_basis` via .fits files, which we will fit with 
the model.
"""
dataset_name = "asymmetric"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Fit__

We first show how to compose a basis of multiple Gaussians and use them to fit a galaxy's light in data.

This is to illustrate the API for performing an MGE using standard autogalaxy objects like the `Galaxy` 
and `FitImaging` 

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the Gaussians. However, an MGE can do a reasonable job even before we just guess sensible 
parameter values.

__Basis__

We first build a `Basis`, which is built from multiple linear light profiles (in this case, Gaussians). 

Below, we make a `Basis` out of 30 elliptical Gaussian linear light profiles which: 

 - All share the same centre and elliptical components.
 - The `sigma` size of the Gaussians increases in log10 increments.
 
Note that any linear light profile can be used to compose a Basis. This includes shapelets, which are a set of functions
closely related to the Exponential function and are often used to represent the light of disk 
galaxies (see `modeling/features/advanced/shapelets.py`).
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = ag.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = ag.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and use it to fit 
data.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
By plotting the fit, we see that the `Basis` does a reasonable job at capturing the appearance of the galaxy.

There are few residuals, except for perhaps some central regions where the light profile is not perfectly fitted.

Given that there was no non-linear search to determine the optimal values of the Gaussians, this is a pretty good fit!
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Nevertheless, there are still residuals, which we now rectify by fitting the MGE in a non-linear search.

__Model__

We compose our model where in this example:

 - The galaxy's bulge is 60 parametric linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of the Gaussians are all linked together.
 - The `sigma` size of the Gaussians increases in log10 increments.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    # A list of Gaussian model components whose parameters are customized belows.

    gaussian_list = af.Collection(
        af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
        gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
        gaussian.ell_comps = gaussian_list[
            0
        ].ell_comps  # All Gaussians have same elliptical components.
        gaussian.sigma = (
            10 ** log10_sigma_list[i]
        )  # All Gaussian sigmas are fixed to values above.

    bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format, which has a lot more parameters than other examples
as it shows the parameters of every individual Gaussian.

This shows every single Gaussian light profile in the model, which is a lot of parameters! However, the vast
majority of these parameters are fixed to the values we set above, so the model actually has far fewer free
parameters than it looks!
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

Owing to the simplicity of fitting an MGE we an use even fewer live points than other examples, reducing it to
75 live points, speeding up convergence of the non-linear search.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[basis]",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
)

"""
__Run Time__

The likelihood evaluation time for a multi-Gaussian expansion is significantly slower than standard / linear 
light profiles. This is because the image of every Gaussian must be computed and evaluated, and each must be blurred 
with the PSF. In this example, the evaluation time is ~0.5s, compared to ~0.01 seconds for standard light profiles.

Huge gains in the overall run-time however are made thanks to the models significantly reduced complexity and lower
number of free parameters. Furthermore, because there are not free parameters which scale the size of lens galaxy,
this produces significantly faster convergence by Nautilus that any other lens light model. We also use fewer live
points, further speeding up the model-fit.

Overall, it is difficult to state which approach will be faster overall. However, the MGE's ability to fit the data
more accurately and the less complex parameter due to removing parameters that scale the lens galaxy make it the 
superior approach.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this does 
not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms there are many `Gaussian`' in the lens light model and it lists their inferred parameters.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.uniform
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Wrap Up__

A Multi Gaussian Expansion is a powerful tool for modeling the light of galaxies, and offers a compelling method to
fit complex light profiles with a small number of parameters

__Regularization (Advanced / Unused)__

An MGE can be regularized, whereby smoothness is enforced on the `intensity` values of the Gaussians. This was 
implemented to avoid a "positive / negative" ringing effect in the lens light model reconstruction, whereby the 
Gaussians went to a systematic solution which alternated between positive and negative values. 

Regularization was intended to smooth over the `intensity` values of the Gaussians, such that the solution would prefer
a positive-only solution. However, this did not work -- even with high levels of regularization, the Gaussians still
went to negative values. The solution also became far from optimal, often leaving significant residuals in the lens
light model reconstruction.

This problem was solved by switching to a positive-only linear algebra solver, which is the default used 
in **PyAutoLens** and was used for all fits performed above. The regularization feature is currently not used by
any scientific analysis and it is recommended you skip over the example below and do not use it in your own modeling.

However, its implementation is detailed below for completeness, and if you think you have a use for it in your own
modeling then go ahead! Indeed, even with a positive-only solver, it may be that regularization helps prevent overfitting
in certain situations.

__Description__

There is one downside to `Basis` functions, we may compose a model with too much freedom. The `Basis` (e.g. our 20
Gaussians) may overfit noise in the data, or possible the galaxyed source galaxy emission -- neither of which we 
want to happen! 

To circumvent this issue, we have the option of adding regularization to a `Basis`. Regularization penalizes
solutions which are not smooth -- it is essentially a prior that says we expect the component the `Basis` represents
(e.g. a bulge or disk) to be smooth, in that its light changes smoothly as a function of radius.

Below, we compose and fit a model using Basis functions which includes regularization, which adds one addition 
parameter to the fit, the `coefficient`, which controls the degree of smoothing applied.
"""
bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
    regularization=ag.reg.Constant,
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, which has addition priors now associated with regularization.
"""
print(model.info)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[basis_regularized]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
