"""
Results: Galaxies
=================

In results tutorial 2, we inspected the results of a `Plane` and computed the overall properties of the model's
image and other quantities.

However, we did not compute the individual properties of each galaxy. For example, we did not compute an image of the
left galaxy or compute individual quantities for each light profile.

This tutorial illustrates how to compute these more complicated results. This is a key reason why we have opted
to include two galaxies in the overall model in these tutorials.

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
__Individual galaxy Components__

We are able to create an image of each galaxy as follows, which includes the emission of only one galaxy at a
time.
"""
image = plane.galaxies[0].image_2d_from(grid=dataset.grid)
image = plane.galaxies[1].image_2d_from(grid=dataset.grid)

"""
In order to create images of each light profile (e.g. the `bulge`), we can extract each individual component from 
each galaxy.

The plane's list of galaxies is in order of how we specify them in the `collection` above.
"""
bulge_0 = plane.galaxies[0].bulge
bulge_1 = plane.galaxies[1].bulge

"""
For simplicity, each galaxy did not contain more light profiles than a bulge. But you could easily
extract a `disk` if it were present:

 disk_0 = plane.galaxies[0].disk
 disk_1 = plane.galaxies[1].disk

Finally, we can use the extracted bulge components to make images of the bulge.
"""
bulge_0_image_2d = bulge_0.image_2d_from(grid=dataset.grid)
bulge_1_image_2d = bulge_1.image_2d_from(grid=dataset.grid)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(bulge_0_image_2d.slim[0])
print(bulge_1_image_2d.slim[0])

"""
It is more concise to extract these quantities in one line of Python:
"""
bulge_0_image_2d = plane.galaxies[0].bulge.image_2d_from(grid=dataset.grid)

"""
The `LightProfilePlotter` makes it straight forward to extract and plot an individual light profile component.
"""
bulge_plotter = aplt.LightProfilePlotter(
    light_profile=plane.galaxies[0].bulge, grid=dataset.grid
)
bulge_plotter.figures_2d(image=True)

"""
__Alternative API__

In the first results tutorial, we used `Samples` objects to inspect the results of a model.

We saw how these samples created instances, which include a `galaxies` property that mains the API of the `Model`
creates above (e.g. `galaxies.galaxy.bulge`). 

We can also use this instance to extract individual components of the model.
"""
samples = result.samples

ml_instance = samples.max_log_likelihood()

bulge = ml_instance.galaxies.galaxy_0.bulge

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
print(bulge_image_2d.slim[0])

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=dataset.grid)
bulge_plotter.figures_2d(image=True)

"""
In fact, if we create a `Plane` from an instance (which is how `result.max_log_likelihood_plane` is created) we
can choose whether to access its attributes using each API: 
"""
plane = result.max_log_likelihood_plane
print(plane.galaxies[0].bulge)
# print(plane.galaxies.galaxy.bulge)

"""
We'll use the former API from here on. 

Whilst its a bit less clear and concise, it is more representative of the internal **PyAutoGalaxy** source code and
therefore gives a clearer sense of how the internals work.

__Galaxies__

Above, we extract the `bulge` light profiles of each galaxy. 

We can just as easily extract each `Galaxy` and use it to perform the calculations above. Note that because the 
galaxy`:
"""
galaxy_0 = plane.galaxies[0]

galaxy_0_image_2d = galaxy_0.image_2d_from(grid=dataset.grid)

"""
We can also use the `GalaxyPlotter` to plot the galaxy, for example a subplot of each individual light profile 
image (which because this galxy is only a single bulge, is a single image).
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_0, grid=dataset.grid)
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
__Plane Composition__

Lets quickly summarize what we've learnt by printing every object in the plane:
"""
print(plane)
print(plane)
print(plane)
print(plane.galaxies[0])
print(plane.galaxies[0])
print(plane.galaxies[0].bulge)
print(plane.galaxies[1].bulge)
print()

"""
__One Dimensional Quantities__

We have made two dimensional plots of galaxy images.

We can also compute all these quantities in 1D, for inspection and visualization.
 
For example, from a light profile or galaxy we can compute its `image_1d`, which provides us with its image values
(e.g. luminosity) as a function of radius.
"""
galaxy_0 = plane.galaxies[0]
image_1d = galaxy_0.image_1d_from(grid=dataset.grid)
print(image_1d)

galaxy_1 = plane.galaxies[1]
image_1d = galaxy_1.image_1d_from(grid=dataset.grid)
print(image_1d)

"""
How are these 1D quantities from an input 2D grid? 

From the 2D grid a 1D grid is compute where:
 
 - The 1D grid of (x,) coordinates are centred on the galaxy or light profile and aligned with the major-axis. 
 - The 1D grid extends from this centre to the edge of the 2D grid.
 - The pixel-scale of the 2D grid defines the radial steps between each coordinate.
 
If we input a larger 2D grid, with a smaller pixel scale, the 1D plot adjusts accordingly.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)
image_1d = galaxy_0.image_1d_from(grid=grid)
print(image_1d.shape)
print(image_1d)

grid = ag.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02)
image_1d = galaxy_0.image_1d_from(grid=grid)
print(image_1d.shape)
print(image_1d)

"""
We can alternatively input a `Grid1D` where we define the (x,) coordinates we wish to evaluate the function on.
"""
grid_1d = ag.Grid1D.uniform_from_zero(shape_native=(10000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_0, grid=grid)

galaxy_plotter.figures_1d(image=True)

"""
__Decomposed 1D Plot__

We can make a plot containing every individual light profile of a galaxy in 1D, for example showing a  
decomposition of its `bulge` and `disk`.

Every profile on a decomposed plot is computed using a radial grid centred on its profile centre and aligned with
its major-axis. Therefore 2D offsets between the components are not portray in such a figure.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_0, grid=grid)
galaxy_plotter.figures_1d_decomposed(image=True)

"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light models estimated via a model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of the model-fit. 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `effective_radius` with errors, which are plotted as vertical lines.

Below, we manually input one hundred realisations of the galaxy with light profiles that clearly show 
these errors on the figure.
"""
galaxy_pdf_list = [samples.draw_randomly_via_pdf().galaxies.galaxy_0 for i in range(10)]

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=galaxy_pdf_list, grid=grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(image=True)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(image=True)

"""
__Wrap Up__

We have learnt how to extract individual planes, galaxies and light rofiles from the plane that results from
a model-fit and use these objects to compute specific quantities of each component.
"""
