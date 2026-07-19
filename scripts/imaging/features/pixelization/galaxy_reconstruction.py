"""
Pixelization: Galaxy Reconstruction
===================================

A common pixelization use-case is to reconstruct the irregular component of a galaxy's surface brightness on a
pixelization mesh, and then export that reconstruction to perform scientific analysis.

It is beneficial to export this reconstruction in a format which is independent of modeling, so study of the
galaxy's clumpy light can be performed separately and shared with collaborators who do not have PyAutoGalaxy
installed.

This script illustrates how the bulge + pixelization model fit (see `modeling.py`) outputs the pixelized
reconstruction to a .csv file, and how that file can be easily loaded to perform downstream analysis.

__Contents__

- **Model Fit:** Running a bulge + pixelization model-fit which outputs the reconstruction to a CSV file.
- **Dataset Auto-Simulation:** Automatically simulating the dataset if it does not already exist.
- **Reconstruction CSV:** Loading the galaxy reconstruction from the output CSV file and performing analysis.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below mirrors the pixelization `modeling` example — fitting the `clumpy` dataset with a `Sersic` bulge
plus a pixelization for the irregular clumpy component. Crucially this model-fit outputs the pixelization
reconstruction to a .csv file.
"""
dataset_name = "clumpy"
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
        [sys.executable, "scripts/imaging/features/pixelization/simulator.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=ag.reg.MaternKernel,
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.Sersic,
    pixelization=pixelization,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("features"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

"""
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstructed values and the noise map of these values. The
filename retains the `source_plane_` prefix because it is emitted by the shared library inversion machinery used
by both PyAutoGalaxy and PyAutoLens — the contents are nonetheless the galaxy's pixelized reconstruction here.

This file provides all information on the galaxy reconstruction in a format that does not depend on PyAutoGalaxy
and therefore can be easily loaded to create images of the reconstruction or shared with collaborators who do not
have PyAutoGalaxy installed.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.

NOTE: If the .csv file does not exist, we create a dictionary with the same format but with dummy values so the
rest of the script can be run.
"""
import csv

try:

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

except FileNotFoundError:

    print("`source_plane_reconstruction_0.csv` not found. Using dummy data instead.")

    x = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    reconstruction = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    reconstruction_dict = {
        "x": x,
        "y": y,
        "reconstruction": reconstruction,
        "noise_map": noise_map,
    }

print(reconstruction_dict["y"])
print(reconstruction_dict["x"])
print(reconstruction_dict["reconstruction"])
print(reconstruction_dict["noise_map"])

"""
You can now use standard libraries to perform calculations with the reconstruction on the mesh, without needing
PyAutoGalaxy.

For example, we can create a Delaunay triangulation of the pixelization mesh using the scipy.spatial library, from
the y and x coordinates exported above. This is useful for visualizing the pixelization and performing calculations
on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

mesh = scipy.spatial.Delaunay(points)

"""
Interpolating the result to a uniform grid is also possible using the scipy.interpolate library, which means the
result can be turned into a uniform 2D image which is useful for analysis tools that require a regular grid.

Below, we interpolate the result onto a 201 x 201 grid of pixels with the extent spanning -1.0" to 1.0", which
captures the majority of the galaxy reconstruction without being too high resolution.
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
