"""
Plots: MCMCPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a emcee non-linear search using
a `MCMCPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
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
First, lets create a result via emcee by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.

We use a model with an initialized starting point, which is necessary for modeling with emcee.
"""
dataset_name = "simple__sersic"

search = af.Emcee(
    path_prefix=path.join("plot"),
    name="MCMCPlotter",
    unique_tag=dataset_name,
    nwalkers=30,
    nsteps=500,
)

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

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)
"""
We now pass the samples to a `MCMCPlotter` which will allow us to use emcee's in-built plotting libraries to 
make figures.

The emcee readthedocs describes fully all of the methods used below 

 - https://emcee.readthedocs.io/en/stable/user/sampler/
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.MCMCPlotter(samples=result.samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
    bins=20,
    range=None,
    color="k",
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
)


"""
Finish.
"""
