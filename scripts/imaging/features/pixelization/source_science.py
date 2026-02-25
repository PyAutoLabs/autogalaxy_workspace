"""
Pixelization: Source Reconstruction
===================================

A common pixelization use-case is to reconstruct the galaxy's surface brightness on a pixelization mesh, and
then export this reconstruction to perform scientific analysis.

It is beneficial to export this reconstruction in a format which is independent of modeling, so study of
the source can be performed separately.

This script illustrates how modeling outputs source reconstructions to a .csv file, and how this can be easily
loaded to perform analysis without the need for PyAutoLens.
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
outputs the pixelization source reconstruction to a .csv file.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

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
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("features"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file is provides all information on the source reconstruction in a format that does not depend autolens
and therefore be easily loaded to create images of the source or shared collaborations who do not have PyAutoLens
installed.

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
the need to use autolens.

For example, we can create a RectangularAdaptDensity mesh using the scipy.spatial library, which is a triangulation
of the y and x coordinates of the pixelization mesh. This is useful for visualizing the pixelization
and performing calculations on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

mesh = scipy.spatiag.Delaunay(points)

"""
Interpolating the result to a uniform grid is also possible using the scipy.interpolate library, which means the result
can be turned into a uniform 2D image which can be useful to analyse the source with tools which require an uniform grid.

Below, we interpolate the result onto a 201 x 201 grid of pixels with the extent spanning -1.0" to 1.0", which
capture the majority of the source reconstruction without being too high resolution.
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
