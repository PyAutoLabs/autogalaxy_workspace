"""
Plots: MapperPlotter
====================

This example illustrates how to plot a `Mapper` using a `MapperPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a galaxy as an `Imaging` object.
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
__Grid__

Now, lets set up a `Grid2D` (using the image of this imaging).
"""
grid = ag.Grid2D.uniform(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

"""
__Galaxies__

The `Mapper` maps pixels from the image-plane of our `Imaging` data to its source plane, via a model.

Lets create a `Plane` which we will use to create the `Mapper`.
"""
pixelization = ag.Pixelization(
    image_mesh=ag.image_mesh.Overlay(shape=(25, 25)),
    mesh=ag.mesh.Delaunay(),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Converting a `Plane` to an `Inversion` performs a number of steps, which are handled by the `GalaxiesToInversion` class. 

This class is where the data and galaxies are combined to fit the data via the inversion.
"""
galaxies_to_inversion = ag.GalaxiesToInversion(
    galaxies=galaxies,
    dataset=dataset,
)

"""
__Mapper__

We can extract a dictionary where every mapper in the plane is a key, paired with values that are each corresponding 
galaxy containing that mapper. 
"""
mapper_galaxy_dict = galaxies_to_inversion.mapper_galaxy_dict

"""
We only need the `Mapper`, which we can extract from this dictionary.
"""
mapper = list(mapper_galaxy_dict)[0]

"""
__Figures__

We now pass the mapper to a `MapperPlotter` and call various `figure_*` methods to plot different attributes.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.figure_2d()

"""
__Subplots__

The `Mapper` can also be plotted with a subplot of its original image.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
The Indexes of `Mapper` plots can be highlighted to show how certain image pixels map to the source plane.
"""
visuals = aplt.Visuals2D(indexes=[0, 1, 2, 3, 4], pix_indexes=[[10, 11], [12, 13, 14]])

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Include__

A `Mapper` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    mapper_image_plane_mesh_grid=True,
    mapper_source_plane_mesh_grid=True,
    mapper_source_plane_data_grid=True,
)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
Finish.
"""
