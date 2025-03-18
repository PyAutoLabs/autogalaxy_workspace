"""
Galaxies
========

This tutorial shows how to use galaxies, including visualizing and extracting their individual light profiles.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/guides/data_structures.ipynb` guide.
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
__Grids__

To describe the luminous emission of galaxies, **PyAutoGalaxy** uses `Grid2D` data structures, which are 
two-dimensional Cartesian grids of (y,x) coordinates. 

Below, we make and plot a uniform Cartesian grid:
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
__Light Profiles__

We will use this `Grid2D`'s coordinates to evaluate the galaxy's morphology. We therefore need analytic 
functions representing a galaxy's light distribution(s). 

For this,  **PyAutoGalaxy** uses `LightProfile` objects, for example the `Sersic` `LightProfile` object which
represents a light distribution:
"""
sersic_light_profile = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.2, 0.1),
    intensity=0.005,
    effective_radius=2.0,
    sersic_index=4.0,
)

"""
By passing this profile a `Grid2D`, we evaluate the light at every (y,x) coordinate on the `Grid2D` and create an 
image of the `LightProfile`.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

"""
The PyAutoGalaxy plot module provides methods for plotting objects and their properties, like 
the `LightProfile`'s image.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.figures_2d(image=True)

"""
__Galaxies__

A `Galaxy` object is a collection of `LightProfile` objects at a given redshift. 

The code below creates a galaxy which is made of two components, a bulge and disk.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=0.5,
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

"""
We can create an image the galaxy by passing it the 2D grid above.
"""
image = galaxy.image_2d_from(grid=grid)

"""
The **PyAutoGalaxy** plot module provides methods for plotting galaxies.

Below, we plot its image, which is the sum of the bulge and disk components.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
__Galaxies__

If our observation contains multiple galaxies, we can create a `Galaxies` object to represent all galaxies.

By passing `Galaxy` objects to a `Galaxies`, **PyAutoGalaxy** groups them to indicate they are at the same redshift.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, -1.0),
        ell_comps=(0.25, 0.1),
        intensity=0.1,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 1.0),
        ell_comps=(0.0, 0.1),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

"""
The image of all galaxies summed can easily be computed from this object.

**PyAutoGalaxy** plot tools allow us to plot this image or a subplot containing images of each individual galaxy.
"""
image = galaxies.image_2d_from(grid=grid)

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)
galaxies_plotter.subplot_galaxy_images()

"""
__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies,
    grid=grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
galaxies_plotter.figures_2d(image=True)

"""
__Extending Objects__

The PyAutoGalaxy API is designed such that all of the objects introduced above are extensible. `Galaxy` objects 
can take many `LightProfile`'s and `Galaxies`'s many `Galaxy`'s. 

To finish, lets create 2 merging galaxies, where the second galaxy has multiple star forming clumps.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lmp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=3.5,
        mass_to_light_ratio=0.6,
    ),
    disk=ag.lmp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=1.0,
        effective_radius=2.0,
        mass_to_light_ratio=0.2,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=1.0,
    bulge=ag.lp.Exponential(
        centre=(0.00, 0.00),
        ell_comps=(0.05, 0.05),
        intensity=1.2,
        effective_radius=0.1,
    ),
    extra_galaxy_0=ag.lp.Sersic(centre=(1.0, 1.0), intensity=0.5, effective_radius=0.2),
    extra_galaxy_1=ag.lp.Sersic(centre=(0.5, 0.8), intensity=0.5, effective_radius=0.2),
    extra_galaxy_2=ag.lp.Sersic(
        centre=(-1.0, -0.7), intensity=0.5, effective_radius=0.2
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

"""
This is what the merging galaxies look like:
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Data Structures Slim / Native__

The images above are returned as a 1D numpy array. 

**PyAutoLens** includes dedicated functionality for manipulating this array, for example mapping it to 2D or
performing the calculation on a high resolution sub-grid which is then binned up. 

This uses the data structure API, which is described in the `guides/data_structures.py` example. This 
tutorial will avoid using this API, but if you need to manipulate results in more detail you should check it out.
"""
print(image.slim)

"""
__Individual Galaxy Components__

We are able to create an image of each galaxy as follows, which includes the emission of only one galaxy at a
time.
"""
image = galaxies[0].image_2d_from(grid=grid)
image = galaxies[1].image_2d_from(grid=grid)

"""
In order to create images of each light profile (e.g. the `bulge`), we can extract each individual component from 
each galaxy.

The list of galaxies is in order of how we specify them in the `collection` above.
"""
bulge_0 = galaxies[0].bulge
bulge_1 = galaxies[1].bulge

"""
You could easily extract the `disk`  of `galaxy_0`:

 disk_0 = galaxies[0].disk

Finally, we can use the extracted bulge components to make images of the bulge.
"""
bulge_0_image_2d = bulge_0.image_2d_from(grid=grid)
bulge_1_image_2d = bulge_1.image_2d_from(grid=grid)

print(bulge_0_image_2d.slim[0])
print(bulge_1_image_2d.slim[0])

"""
It is more concise to extract these quantities in one line of Python:
"""
bulge_0_image_2d = galaxies[0].bulge.image_2d_from(grid=grid)

"""
The `LightProfilePlotter` makes it straight forward to extract and plot an individual light profile component.
"""
bulge_plotter = aplt.LightProfilePlotter(light_profile=galaxies[0].bulge, grid=grid)
bulge_plotter.figures_2d(image=True)

"""
__Galaxies__

We extracted the `bulge` light profiles of each galaxy. 

We can just as easily extract each `Galaxy` and use it to perform the calculations above.
"""
galaxy_0 = galaxies[0]

galaxy_0_image_2d = galaxy_0.image_2d_from(grid=grid)

"""
We can also use the `GalaxyPlotter` to plot the galaxy, for example a subplot of each individual light profile 
image (which because this galxy is only a single bulge, is a single image).
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_0, grid=grid)
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
__Galaxies Composition__

Lets quickly summarise what we've learnt by printing every object:
"""
print(galaxies)
print(galaxies[0])
print(galaxies[0])
print(galaxies[0].bulge)
print(galaxies[1].bulge)
print()

"""
__One Dimensional Quantities__

We have made two dimensional plots of galaxy images.

We can also compute all these quantities in 1D, for inspection and visualization.

For example, from a light profile or galaxy we can compute its `image_1d`, which provides us with its image values
(e.g. luminosity) as a function of radius.
"""
galaxy_0 = galaxies[0]
image_1d = galaxy_0.image_1d_from(grid=grid)
print(image_1d)

galaxy_1 = galaxies[1]
image_1d = galaxy_1.image_1d_from(grid=grid)
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
__Modeling Results__

Modeling uses a non-linear search to fit a model of galaxies to a dataset.

It is illustrated in the `modeling` packages of `autogalaxy_workspace`.

Modeling results have some specific functionality and use cases, which are described in the `results` packages of
`autogalaxy_workspace`,  in particular the `galaxies_fit.py` example script which describes: 

 - `Max Likelihood`: Extract and plot the galaxy models which maximize the likelihood of the fit.
 - `Samples`, Extract the samples of the non-linear search and inspect specific parameter values.
 - `Errors`: Makes plots that quantify the errors on the inferred galaxy properties.
 - `Refitting` Refit specific models from the modeling process to the dataset. 

__Wrap Up__

This tutorial explained how to compute the results of an inferred model from a galaxies. 

We have learnt how to extract individual galaxies and light profiles from the results of 
a model-fit and use these objects to compute specific quantities of each component.
"""
