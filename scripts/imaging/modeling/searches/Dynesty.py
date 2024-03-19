"""
Searches: Emcee
===============

Nautilus (https://github.com/joshspeagle/Nautilus) is a nested sampling algorithm.

A nested sampling algorithm estimates the Bayesian evidence of a model as well as the posterior.

Dynesty used to be the main model-fitting algorithm used by PyAutoLens. However, we now recommend the nested sampling
algorithm `Nautilus` instead, which is faster and more accurate than Dynesty. We include this tutorial for Dynesty
for those who are interested in comparing the two algorithms.

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

"""
__Model__ 

In our experience, zeus is okay at initializing a model but not as good at `nautilus. It therefore benefits 
from a 'starting point' which is near the highest likelihood models. We set this starting point up below using
the start point API (see `autogalaxy_workspace/*/imaging/modeling/customize/start_point.ipynb`).

Given this need for a robust starting point, Emcee is only suited to model-fits where we have this information. It may
therefore be useful when performing modeling search chaining (see HowToGalaxy chapter 3). However, even in such
circumstances, we have found that is often outperformed by other searches such as Nautilus and Zeus for both speed
and accuracy.
"""
bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
__Analysis__ 

We create the Analysis as per using.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Search__

Below we use dynesty to fit the model, using the model with start points as described above. See the Dynesty docs
for a description of what the input parameters below do.

There are two important inputs worth noting:

- `sample="rwalk"`: Makes dynesty use random walk nested sampling, which proved to be effective at modeling.
- `walks-10`: Only 10 random walks are performed per sample, which is efficient for modeling.

"""
search = af.DynestyStatic(
    path_prefix=path.join("searches"),
    name="DynestyStatic",
    nlive=50,
    sample="rwalk",
    walks=10,
    bound="multi",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    iterations_per_update=2500,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `MCMCPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
Finish.
"""
