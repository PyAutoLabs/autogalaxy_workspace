"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the galaxy light into ~15-100 Gaussians, where
the `intensity` of every Gaussian is solved for via a linear algebra using a process called an "inversion"
(see the `light_parametric_linear.py` feature for a full description of this).

This script fits a light model which uses an MGE consisting of 60 Gaussians. It is fitted to simulated data
where the galaxy's light has asymmetric and irregular features, which fitted poorly by symmetric light
profiles like the `Sersic`.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**MGE Source Galaxy:** Discussion of using the MGE for the source galaxy, which is illustrated fully at the end of the example.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Model:** Composing a model using an MGE and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of MGE run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** MGE results, including accessing light profiles with solved for intensity values.
**MGE Source:** Detailed illustration of using MGE source.
**Regularization:** API for applying regularization to MGE, which is not recommend but included for illustration.

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
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load and plot the galaxy dataset `asymetric` via .fits files.
"""
dataset_name = "asymmetric"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
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
__Over Sampling__

Apply adaptive over sampling to ensure the calculation is accurate, you can read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose our model where in this example:

 - The galaxy's bulge is 60 parametric linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of the Gaussians are all linked together.
 - The `sigma` size of the Gaussians increases in log10 increments.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6.
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
    path_prefix=Path("imaging") / "features",
    name="mge",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset,
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

For each linear Gaussian light profile, extra VRAM is used. For around 60 linear Gaussians this  typically requires 
a modest amount of VRAM (e.g. 10–50 MB per batched likelihood). Models that use hundreds of Gaussians, especially in 
combination with a large batch size, may therefore exceed GBs of VRAM and require you to adjust the batch size to fit 
within your GPU's VRAM.

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

Checkout `autogalaxy_workspace/*/guides/results` for a full description of analysing results in **PyAutoGalaxy**.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
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
    path_prefix=Path("imaging") / "features",
    name="light[basis_regularized]",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
