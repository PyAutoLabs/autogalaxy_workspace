"""
Results: Plane
==============

This tutorial inspects an inferred model using the `Plane` object inferred by the non-linear search.
This allows us to visualize and interpret its results.

This tutorial focuses on explaining how to use the inferred plane to compute results as numpy arrays and only
briefly discusses visualization.

Unlike many tutorials, we will fit a dataset and model where there are two galaxies in the dataset. This will
help us illustrate how the analyse results for most complex models, however the API can be easily
generalized for more simple fits.

__Plot Module__

This example uses the **PyAutoGalaxy** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/imaging/results/examples/data_structure.ipynb` example.
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
__Model Fit__

The code below performs a model-fit using dynesty. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

Note that the model that is fitted has two galaxies, as opposed to just one like usual!
"""
dataset_name = "sersic_x2"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

bulge_0 = af.Model(ag.lp.Sersic)
bulge_0.centre = (0.0, -1.0)

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_0)

bulge_1 = af.Model(ag.lp.Sersic)
bulge_1.centre = (0.0, 1.0)

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_1)

model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]__x2",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Plane__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_plane` which we can visualize.
"""
plane = result.max_log_likelihood_plane

plane_plotter = aplt.PlanePlotter(plane=plane, grid=mask.derive_grid.all_false_sub_1)
plane_plotter.subplot_plane()

"""
This tutorial now focuses on explaining the numerical results contained in the plane.

Visualization is explained separately in the `autogalaxy_workspace/*/plot/` package, in particular the 
`plotters/PlanePlotter.py` script.

__Inferred 2D Images__

The maximum log likelihood plane contains a lot of information about the inferred model.

For example, by passing it a 2D grid of (y,x) coordinates we can return a numpy array containing its 2D image. This
includes the bulge and disk images.

Below, we use the grid of the `imaging` to computed the image on, which is the grid used to fit to the data.
"""
image = plane.image_2d_from(grid=dataset.grid)

"""
__Data Structures Slim / Native__

The image above is returned as a 1D numpy array. 

**PyAutoLens** includes dedicated functionality for manipulating this array, for example mapping it to 2D or
performing the calculation on a high resolution sub-grid which is then binned up. 

This uses the data structure API, which is described in the `results/examples/data_structures.py` example. This 
tutorial will avoid using this API, but if you need to manipulate results in more detail you should check it out.
"""
print(image.slim)

"""
__Grid Choices__

We can input a different grid, which is not masked, to evaluate the image everywhere of interest. We can also change
the grid's resolution from that used in the model-fit.

The examples uses a grid with `shape_native=(3,3)`. This is much lower resolution than one would typically use to 
inspect galaxy properties, but is chosen here so that the `print()` statements display in a concise and readable format.
"""
grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1)

image = plane.image_2d_from(grid=grid)

print(image.slim)

"""
__Sub Gridding__

A grid can also have a sub-grid, defined via its `sub_size`, which defines how each pixel on the 2D grid is split 
into sub-pixels of size (`sub_size` x `sub_size`). 

The calculation below shows how to use a sub-grid and bin it up, full details of the API for this calculation
are given in the `results/examples/data_structure.py` example.
"""
grid_sub = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1, sub_size=2)

image = plane.image_2d_from(grid=grid_sub)

print(image.binned)

"""
__Positions Grid__

We may want the image at specific (y,x) coordinates.

We can use an irregular 2D (y,x) grid of coordinates for this. The grid below evaluates the image at:

- y = 1.0, x = 1.0.
- y = 1.0, x = 2.0.
- y = 2.0, x = 2.0.
"""
grid_irregular = ag.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

image = plane.image_2d_from(grid=grid_irregular)

print(image)

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the plane changes for models with similar log likelihoods. Below, we create and plot
the plane of the 100th last accepted model by dynesty.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

plane = ag.Plane(galaxies=instance.galaxies)

plane_plotter = aplt.PlanePlotter(plane=plane, grid=mask.derive_grid.all_false_sub_1)
plane_plotter.subplot_plane()

"""
__Wrap Up__

This tutorial explained how to compute the results of an inferred model from a plane. 

We covered a lot, but also omitted a lot of details, for example:

 - We could only compute the image of the entire `Plane` object. What if I want these quantities for specific 
 galaxies in the plane?

 - How do I estimate errors on these quantities?
 
We will answer all these questions in tutorial 4 of the `results`, after we've covered how to go about inspecting the
results of a fit in more detail.
"""
