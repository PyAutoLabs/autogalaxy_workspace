"""
Plots: Plotters Pixelization
============================

This example illustrates the API for plotting using `Plotter` objects for pixelized source reconstructions.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**Fit Imaging:** Plot the fit of a tracer to an imaging dataset for a source reconstruction using a pixelization.
**Inversion:** Plot the inversion object which performs the linear algebra and other calculations which reconstruct the source galaxy.
**Mapper:** Plot the mapper object which maps pixels from the image-plane of the data to its source plane pixelization via a lens model.
**Fit Interferometer:** Plot the fit of a tracer to an interferometer dataset for a source reconstruction using a pixelization.

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(25, 25)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Pixelization__

We can also plot a `FitImaging` which uses a `Pixelization`.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Fit Imaging__

The `FitImaging` object is a base object which represents the fit of a model to an imaging dataset, including the
residuals, chi-squared and model image.

We plot the plane, which being pixelized, is represented by a `Pixelization` object and plotted as a 
delunay mesh of triangles.

The plot below zooms into the brightest pixel of the source-plane, which is useful for visualizing the key regions
of the source that fit the data.
"""
fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
)
fit_plotter.subplot_fit()

"""
__Inversion Plotter__

We can even extract an `InversionPlotter` from the `FitImagingPlotter` and use it to plot all of its usual methods,
which will now include the caustic and border.
"""
inversion_plotter = fit_plotter.inversion_plotter
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
__Inversion__

The fit above has a property called an `inversion`, which contains all of the linear algebra, mesh calculations
and other key quantities used to reconstruct a source galaxy using a pixelization.

This has its own dedicated plotter, the `InversionPlotter`, which can be used to plot the inversion's attributes
and properties in a similar way to the `FitImagingPlotter`.
"""
inversion = fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
Converting a `Galaxies` to an `Inversion` performs a number of steps, which are handled by the `GalaxiesToInversion` 
class. 

This class is where the data and galaxies are combined to fit the data via the inversion.
"""
galaxies_to_inversion = ag.GalaxiesToInversion(
    galaxies=galaxies,
    dataset=dataset,
)

inversion = galaxies_to_inversion.inversion

"""
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
The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
__Mesh Grids__

The image plane mesh grid, showing the centre of every pixel in the image-plane can be computed and plotted.
"""
mapper = fit.inversion.cls_list_from(cls=ag.AbstractMapper)[0]

image_plane_mesh_grid = mapper.image_plane_mesh_grid
visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)
fit_plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals_2d)
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, plane_image=True)

"""
__Mapper__

The `Mapper` is a property of an inversion and maps pixels from the image-plane of the data to its source plane via 
a lens model.

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
visuals = aplt.Visuals2D(indexes=[0, 1, 2, 3, 4])

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Mesh Grids__

The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
image_plane_mesh_grid = mapper.image_plane_mesh_grid

visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals_2d)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__DelaunayDrawer / VoronoiDrawer__

We can customize the filling of Voronoi cells using the `VoronoiDrawer` object which wraps the 
method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
delaunay_drawer = aplt.DelaunayDrawer(edgecolor="b", linewidth=1.0, linestyle="--")
# voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b", linewidth=1.0, linestyle="--")

mat_plot = aplt.MatPlot2D(delaunay_drawer=delaunay_drawer)

"""
We now pass the inversion to a `InversionPlotter` which we will use to illustrate customization with 
the `VoronoiDrawer` object.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot)

try:
    inversion_plotter.figures_2d_of_pixelization(
        pixelization_index=0, reconstruction=True
    )
    inversion_plotter.subplot_of_mapper(mapper_index=0)
except ImportError:
    print(
        "You have not installed the Voronoi natural neighbor interpolation package, see instructions at top of notebook."
    )

"""
Finish.
"""
