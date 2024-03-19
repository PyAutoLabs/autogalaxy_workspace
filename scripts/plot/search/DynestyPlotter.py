"""
Plots: NestPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a nautilus non-linear search using
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
First, lets create a result via nautilus by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.
"""
dataset_name = "simple__sersic"

search = af.Nautilus(
    path_prefix=path.join("plot"),
    name="NestPlotter",
    unique_tag=dataset_name,
    n_live=100,
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
We now pass the samples to a `NestPlotter` which will allow us to use nautilus's in-built plotting libraries to 
make figures.

The nautilus readthedocs describes fully all of the methods used below 

 - https://nautilus.readthedocs.io/en/latest/quickstart.html
 - https://nautilus.readthedocs.io/en/latest/api.html#module-nautilus.plotting
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.

Nautilus plotters use `_kwargs` dictionaries to pass visualization settings to matplotlib lib. For example, below,
we:

 - Set the fontsize of the x and y labels by passing `label_kwargs={"fontsize": 16}`.
 - Set the fontsize of the title by passing `title_kwargs={"fontsize": "10"}`.
 
There are other `_kwargs` inputs we pass as None, you should check out the Nautilus docs if you need to customize your
figure.
"""
plotter = aplt.NestPlotter(samples=result.samples)

"""
__Notation__

Plot are labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).

__Plots__

The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
    dims=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=0.02,
    quantiles_2d=None,
    hist_kwargs=None,
    hist2d_kwargs=None,
    label_kwargs={"fontsize": 16},
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": "10"},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
)

"""
Finish.
"""
