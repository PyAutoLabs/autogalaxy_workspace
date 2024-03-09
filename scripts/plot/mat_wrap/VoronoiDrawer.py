"""
Plots: VoronoiDrawer
====================

This example illustrates how to customize the appearance of the Voronoi mesh of a Voronoi mesh using the
`VoronoiDrawer` object.

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

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)
dataset = dataset.apply_mask(mask=mask)

"""
__Plane__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a model.

Lets create a `Plane` which we will use to create the `Inversion`.
"""
pixelization = ag.Pixelization(
    image_mesh=ag.image_mesh.Overlay(shape=(25, 25)),
    mesh=ag.mesh.Voronoi(),
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
    data=dataset.data,
    noise_map=dataset.noise_map,
    w_tilde=dataset.w_tilde,
)

inversion = galaxies_to_inversion.inversion

"""
We can customize the filling of Voronoi cells using the `VoronoiDrawer` object which wraps the 
method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b", linewidth=1.0, linestyle="--")

mat_plot = aplt.MatPlot2D(voronoi_drawer=voronoi_drawer)

"""
We now pass the inversion to a `InversionPlotter` which we will use to illustrate customization with 
the `VoronoiDrawer` object.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot)

inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)
inversion_plotter.subplot_of_mapper(mapper_index=0)
