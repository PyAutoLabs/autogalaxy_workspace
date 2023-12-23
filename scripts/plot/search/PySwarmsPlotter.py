"""
Plots: PySwarmsPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a pyswarms non-linear search using
a `ZeusPlotter`.

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
First, lets create a result via pyswarms by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.
"""
dataset_name = "simple__sersic"

search = af.PySwarmsGlobal(
    path_prefix=path.join("plot", "PySwarmsPlotter"),
    name="PySwarms",
    n_particles=50,
    iters=10,
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
We now pass the samples to a `PySwarmsPlotter` which will allow us to use pyswarms's in-built plotting libraries to 
make figures.

The pyswarms readthedocs describes fully all of the methods used below 

 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.plotters.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
pyswarms_plotter = aplt.PySwarmsPlotter(samples=result.samples)

"""
The `contour` method shows a 2D projection of the particle trajectories.
"""
pyswarms_plotter.contour(
    canvas=None,
    title="Trajectories",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
)


"""
The `cost history` shows in 1D the evolution of each parameters estimated highest likelihood.
"""
pyswarms_plotter.cost_history(ax=None, title="Cost History", designer=None)

"""
Finish.
"""
