"""
Plots: Pixelization
===================

This example illustrates the API for plotting pixelized source reconstructions using the new function-based
plotting API and dedicated functions like `subplot_of_mapper` for mesh-specific visualizations.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

- **Setup:** Set up all objects (e.g. grid, dataset, fit) used to illustrate plotting.
- **Fit Imaging:** Plot the fit of a model to an imaging dataset that uses a pixelization.
- **Reconstruction:** Plot the reconstructed source galaxy mapped back to the image frame.
- **Inversion Plots:** Plot diagnostic subplots of the inversion properties using subplot_of_mapper.
- **Mapper:** Plot the mapper that maps image-plane pixels to the source-plane pixelization.
- **Mesh Grids:** Plot the image and source plane mesh grids.
- **Delaunay:** Customize the filling of Delaunay mesh plots.

__Setup__

To illustrate plotting, we require standard objects like a grid, dataset and fit.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
from autoarray.inversion.plot.mapper_plots import plot_mapper, subplot_image_and_mapper
from autoarray.inversion.plot.inversion_plots import subplot_of_mapper

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/guides/plot/simulator.py"],
        check=True,
    )


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
    mesh=ag.mesh.RectangularAdaptDensity(shape=(25, 25)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Fit Imaging__

The fit of a model using a pixelized source reconstruction is plotted using `aplt.subplot_fit_imaging`.

This subplot shows the data, model image, residuals and source-plane reconstruction on the mesh.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Reconstruction__

The reconstructed source galaxy on the mesh can be plotted using `aplt.plot_array`, passing the reconstruction
mapped back to the native image frame.
"""
inversion = fit.inversion

aplt.plot_array(array=fit.model_data, title="Reconstruction")

"""
__Inversion Plots__

The `subplot_of_mapper` function provides a comprehensive diagnostic subplot of the inversion properties for
a single mapper, including the reconstructed image, source reconstruction, noise map and regularization weights.
"""
subplot_of_mapper(inversion=inversion, mapper_index=0)

"""
__Mapper__

The `Mapper` maps image-plane pixels to the source-plane pixelization.

We extract the mapper from the inversion and plot it using `plot_mapper` and `subplot_image_and_mapper`.
"""
galaxies_to_inversion = ag.GalaxiesToInversion(
    galaxies=galaxies,
    dataset=dataset,
)

inversion = galaxies_to_inversion.inversion

mapper_galaxy_dict = galaxies_to_inversion.mapper_galaxy_dict

mapper = list(mapper_galaxy_dict)[0]

plot_mapper(mapper=mapper)

"""
The `Mapper` can also be plotted with a subplot showing the original image alongside the source-plane reconstruction.
"""
subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
The indexes of `Mapper` plots can be highlighted to show how certain image pixels map to the source plane.
"""
subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
__Mesh Grids__

The image and source plane mesh grids, showing the centre of every source pixel, can be computed and plotted.
"""
image_plane_mesh_grid = mapper.mask.derive_grid.unmasked

subplot_image_and_mapper(
    mapper=mapper, image=dataset.data, mesh_grid=image_plane_mesh_grid
)

"""
__Delaunay__

We can customize the filling of object which wraps the method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
subplot_of_mapper(inversion=inversion, mapper_index=0)

"""
Finish.
"""
