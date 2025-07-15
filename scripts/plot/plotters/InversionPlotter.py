"""
Plots: InversionPlotter
=======================

This example illustrates how to plot a `Inversion` using a `InversionPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a galaxy as an `Imaging` object.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

"""
__Galaxies__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a model.

Lets create a `Plane` which we will use to create the `Inversion`.
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

inversion = galaxies_to_inversion.inversion

"""
__Figures__

We now pass the inversion to a `InversionPlotter` and call various `figure_*` methods to plot different attributes.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An `Inversion` can have multiple mappers, which reconstruct multiple source galaxies at different redshifts and
planes (e.g. double Einstein ring systems).

To plot an individual source we must therefore specify the mapper index of the source we plot.
"""
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0,
    reconstructed_image=True,
    reconstruction=True,
    reconstruction_noise_map=True,
    regularization_weights=True,
)


"""
__Subplots__

The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
__Mesh Grids__

The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
mapper = inversion.cls_list_from(cls=ag.AbstractMapper)[0]

image_plane_mesh_grid = mapper.image_plane_mesh_grid
visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)
inversion_plotter = aplt.InversionPlotter(inversion=inversion, visuals=visuals_2d)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
Finish.
"""
