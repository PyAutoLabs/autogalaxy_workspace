"""
Plots: NestPlotter
=======================

This example illustrates how to plot visualization summarizing the results of a ultranest non-linear search using
a `NestPlotter`.

Installation
------------

Because UltraNest is an optional library, you will likely have to install it manually via the command:

`pip install ultranest`

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
First, lets create a result via ultranest by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.
"""
dataset_name = "simple__sersic"

search = af.UltraNest(path_prefix="plot", name="NestPlotter")

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
We now pass the samples to a `NestPlotter` which will allow us to use ultranest's in-built plotting libraries to 
make figures.

The ultranest readthedocs describes fully all of the methods used below 

 - https://johannesbuchner.github.io/UltraNest/readme.html
 - https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.plot
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
ultranest_plotter = aplt.NestPlotter(samples=result.samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
ultranest_plotter.corner()

"""
Finish.
"""
