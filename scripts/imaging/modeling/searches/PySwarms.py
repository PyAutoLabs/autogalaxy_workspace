"""
Searches: PySwarms
==================

PySwarms is a  particle swarm optimization (PSO) algorithm.

Information about PySwarms can be found at the following links:

 - https://github.com/ljvmiranda921/pyswarms
 - https://pyswarms.readthedocs.io/en/latest/index.html
 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html#module-pyswarms.single.global_best

An PSO algorithm only seeks to only find the maximum likelihood model, unlike MCMC or nested sampling algorithms
like Zzeus and nautilus, which aims to map-out parameter space and infer errors on the parameters.Therefore, in 
principle, a PSO like PySwarm should fit a model very fast.

In our experience, the parameter spaces fitted by models are too complex for `PySwarms` to be used without a lot
of user attention and care.  Nevertheless, we encourage you to give it a go yourself, and let us know on the PyAutoGalaxy 
GitHub if you find an example of a problem where `PySwarms` outperforms Nautilus!

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
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

"""
__Model__ 

In our experience, zeus is okay at initializing a model but not as good at `nautilus. It therefore benefits 
from a 'starting point' which is near the highest likelihood models. We set this starting point up below using
the start point API (see `autogalaxy_workspace/*/imaging/modeling/customize/start_point.ipynb`).

Given this need for a robust starting point, PySwarms is only suited to model-fits where we have this information. It may
therefore be useful when performing modeling search chaining (see HowToGalaxy chapter 3). However, even in such
circumstances, we have found that is often unrealible and often infers a local maxima.
"""
bulge = af.Model(ag.lp.Sersic)

bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.intensity = af.LogUniformPrior(lower_limit=0.5, upper_limit=1.5)
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

Below we use `PySwarmsGlobal` to fit the model, using the model where the particles start as described above. 
See the PySwarms docs for a description of what the input parameters below do and what the `Global` search technique is.
"""
search = af.PySwarmsGlobal(
    path_prefix=path.join("imaging", "searches"),
    name="PySwarmsGlobal",
    unique_tag=dataset_name,
    n_particles=30,
    iters=300,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    iterations_per_update=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `MLEPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
plotter = aplt.MLEPlotter(samples=result.samples)
plotter.cost_history()

"""
__Search__

We can also use a `PySwarmsLocal` to fit the model
"""
search = af.PySwarmsLocal(
    path_prefix=path.join("imaging", "searches"),
    name="PySwarmsLocal",
    unique_tag=dataset_name,
    n_particles=30,
    iters=300,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    iterations_per_update=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

plotter = aplt.MLEPlotter(samples=result.samples)
plotter.cost_history()

"""
Finish.
"""
