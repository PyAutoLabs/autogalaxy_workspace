"""
Pixelization: Galaxy Reconstruction
===================================

A common pixelization use-case is to reconstruct a galaxy’s surface brightness on a pixelization mesh, and then
export this reconstruction to perform scientific analysis.

It is beneficial to export this reconstruction in a format which is independent of galaxy modeling, so study of
the galaxy can be performed separately.

This script illustrates how galaxy modeling outputs pixelized reconstructions to a .csv file, and how this can be
easily loaded to perform analysis without the need for PyAutoGalaxy.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below is identical to the pixelizaiton `modeling` example, crucially creating a model-fit which
outputs the pixelization reconstruction to a .csv file.
"""
mask_radius = 3.5

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

dataset = dataset.apply_w_tilde(use_jax=True, show_progress=True)

settings_inversion = ag.SettingsInversion(use_positive_only_solver=False)

mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = ag.Preloads(
    mapper_indices=ag.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=ag.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    pixelization=af.Model(
        ag.Pixelization,
        mesh=af.Model(ag.mesh.RectangularUniform, shape=mesh_shape),
        regularization=af.Model(ag.reg.Constant),
    ),
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("interferometer"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU galaxy model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=50000,
)

analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    preloads=preloads,
    settings_inversion=settings_inversion,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

result = search.fit(model=model, analysis=analysis)

"""
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file provides all information on the reconstruction in a format that does not depend on autogalaxy
and can therefore be easily loaded to create images of the galaxy or shared with collaborators who do not have
PyAutoGalaxy installed.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.
"""
import csv

with open(
    search.paths.image_path / "source_plane_reconstruction_0.csv", mode="r"
) as file:
    reader = csv.reader(file)
    header_list = next(reader)  # ['y', 'x', 'reconstruction', 'noise_map']

    reconstruction_dict = {header: [] for header in header_list}

    for row in reader:
        for key, value in zip(header_list, row):
            reconstruction_dict[key].append(float(value))

    # Convert lists to NumPy arrays
    for key in reconstruction_dict:
        reconstruction_dict[key] = np.array(reconstruction_dict[key])

print(reconstruction_dict["y"])
print(reconstruction_dict["x"])
print(reconstruction_dict["reconstruction"])
print(reconstruction_dict["noise_map"])

"""
You can now use standard libraries to performed calculations with the reconstruction on the mesh, again avoiding
the need to use autogalaxy.

For example, we can create a RectangularUniform mesh using the scipy.spatial library, which is a triangulation
of the y and x coordinates of the pixelization mesh. This is useful for visualizing the pixelization
and performing calculations on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

mesh = scipy.spatial.Delaunay(points)

"""
Interpolating the result to a uniform grid is also possible using the scipy.interpolate library, which means the result
can be turned into a uniform 2D image which can be useful to analyse the galaxy with tools which require an uniform grid.

Below, we interpolate the result onto a 201 x 201 grid of pixels with the extent spanning -1.0" to 1.0", which
capture the majority of the reconstruction without being too high resolution.

It should be noted this inteprolation may not be as optimal as the interpolation perforemd above using `MapperValued`,
which uses specifc interpolation methods for a RectangularUniform mesh which are more accurate, but it should be sufficent for
most use-cases.
"""
from scipy.interpolate import griddata

values = reconstruction_dict["reconstruction"]

interpolation_grid = ag.Grid2D.from_extent(
    extent=(-1.0, 1.0, -1.0, 1.0), shape_native=(201, 201)
)

interpolated_array = griddata(points=points, values=values, xi=interpolation_grid)

"""
Finish.
"""
