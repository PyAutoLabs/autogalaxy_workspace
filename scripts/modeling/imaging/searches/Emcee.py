"""
Searches: Emcee
===============

Emcee (https://github.com/dfm/emcee) is an ensemble MCMC sampler.

An MCMC algorithm only seeks to map out the posterior of parameter space, unlike a nested sampling algorithm like
Nautilus, which also aims to estimate the Bayesian evidence if the model. Therefore, in principle, an MCMC approach like
Emcee should be faster than Nautilus.

In our experience, `Emcee` is outperformed by `Nautilus` for modeling in all circumstances, whether that be
initializing the model, fitting a pixelized source or models with high dimensionality. Nevertheless, we encourage
you to give it a go yourself, and let us know on the PyAutoGalaxy GitHub if you find an example of a problem where
`Emcee` outperforms Nautilus!

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

In our experience, zeus is okay at initializing a model but not as good at `nautilus. It therefore benefits 
from a 'starting point' which is near the highest likelihood models. We set this starting point up below using
the start point API (see `autogalaxy_workspace/*/modeling/imaging/customize/start_point.ipynb`).

Given this need for a robust starting point, Emcee is only suited to model-fits where we have this information. It may
therefore be useful when performing modeling search chaining (see HowToGalaxy chapter 3). However, even in such
circumstances, we have found that is often outperformed by other searches such as Nautilus and Zeus for both speed
and accuracy.
"""
bulge = af.Model(ag.lp_linear.Sersic)

bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=0.5)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Analysis__ 

We create the Analysis as per using.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Search__

Below we use emcee to fit the model, using the model with start points as described above. See the Emcee docs
for a description of what the input parameters below do.
"""
search = af.Emcee(
    path_prefix=path.join("imaging", "searches"),
    name="Emcee",
    unique_tag=dataset_name,
    nwalkers=30,
    nsteps=500,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    iterations_per_update=5000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `MCMCPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
plotter = aplt.MCMCPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
Finish.
"""
