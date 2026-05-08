"""
Plots: Plotters
===============

This example illustrates the API for plotting using standalone functions, which enable quick visualization of all
key quantities.

The new plotting API replaces class-based plotters (e.g. `Imaging`, `FitImaging`) with standalone
functions that are simpler to call and require less boilerplate code.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotting works and the default
behaviour of plotting visuals.

__Contents__

- **Setup:** Set up all objects (e.g. grid, dataset, fit) used to illustrate plotting.
- **Array2D:** Plot a 2D array using `aplt.plot_array`.
- **Grid2D:** Plot a 2D grid of coordinates using `aplt.plot_grid`.
- **Light Profile:** Plot a light profile image using `aplt.plot_array`.
- **Galaxy:** Plot a galaxy's image using `aplt.plot_array`.
- **Galaxies:** Plot galaxies using `aplt.plot_array` and `aplt.subplot_galaxies`.
- **Imaging:** Plot an imaging dataset using `aplt.subplot_imaging_dataset`.
- **Fit Imaging:** Plot a fit to imaging data using `aplt.subplot_fit_imaging`.
- **Log10:** Plot galaxy images in log10 space for clearer visualization.
- **One Dimensional Plots:** Plot 1D radial profiles using standard matplotlib.
- **Output:** Save figures to disk using `output_path`, `output_filename`, `output_format` arguments.
- **Probability Density Function (PDF) Plots:** Plot 1D light profiles with error regions from model-fit PDFs.

__Setup__

To illustrate plotting, we require standard objects like a grid, dataset and fit.
"""

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import math
import numpy as np

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

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

bulge = ag.lp.Sersic(
    centre=(0.0, -0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=2.0,
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

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

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Array2D__

Any 2D array (e.g. images, noise maps, residuals) can be plotted using `aplt.plot_array`.

The `title` argument sets the title of the plot.
"""
aplt.plot_array(array=dataset.data, title="Data")

"""
__Grid2D__

A 2D grid of (y,x) coordinates can be plotted using `aplt.plot_grid`.
"""
aplt.plot_grid(grid=grid, title="Grid")

"""
__Light Profile__

A light profile's image is computed via `image_2d_from` and then plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=bulge.image_2d_from(grid=grid), title="Image")

"""
__Galaxy__

A galaxy's image is computed via `image_2d_from` and then plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Image")

"""
__Galaxies__

The summed image of all galaxies can be plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
A subplot showing each individual galaxy's image side-by-side can be plotted using `aplt.subplot_galaxies`.
"""
aplt.subplot_galaxies(galaxies=galaxies, grid=grid)

"""
__Imaging__

An imaging dataset (data, noise-map, PSF) can be plotted as individual arrays via `aplt.plot_array`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")

"""
A full subplot of all imaging dataset quantities is plotted using `aplt.subplot_imaging_dataset`.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit Imaging__

The fit of a model to imaging data is plotted using `aplt.subplot_fit_imaging`, which shows the data, model image,
residuals, chi-squared map and other quantities.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
Individual quantities from the fit can also be plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one.

Passing `use_log10=True` plots the array in log10 space, making the galaxy's outskirts more visible.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image", use_log10=True)

"""
__One Dimensional Plots__

1D profiles (e.g. a light profile's intensity as a function of radius) are best plotted using standard matplotlib,
which gives full control over the figure.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy.bulge.centre, angle=galaxy_0.bulge.angle()
)

image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
Using a radial grid of (y,x) coordinates along the x-axis plots the 1D radial profile.
"""
radii = np.arange(10000) * 0.01
grid_radial = ag.Grid2DIrregular(values=[(0.0, r) for r in radii])
image_1d = galaxy_0.image_2d_from(grid=grid_radial)

plt.plot(radii, image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
We can also plot decomposed 1D profiles, displaying each individual light profile separately.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy_0.bulge.centre, angle=galaxy_0.bulge.angle()
)
bulge_image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)

grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy_1.bulge.centre, angle=galaxy_1.bulge.angle()
)
disk_image_1d = galaxy.disk.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], bulge_image_1d, label="Bulge")
plt.plot(grid_2d_projected[:, 1], disk_image_1d, label="Disk")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.legend()
plt.show()
plt.close()

"""
__Output__

All plotting functions accept `output_path`, `output_filename` and `output_format` arguments to save figures to disk.

For example, to save the data image as a `.png` file:
"""
aplt.plot_array(
    array=dataset.data,
    title="Data",
    output_path=dataset_path,
    output_filename="data",
    output_format="png",
)

"""
The same output arguments work for subplot functions:
"""
aplt.subplot_imaging_dataset(
    dataset=dataset,
    output_path=dataset_path,
    output_filename="subplot_dataset",
    output_format="png",
)

aplt.subplot_fit_imaging(
    fit=fit,
    output_path=dataset_path,
    output_format="png",
)

"""
__Probability Density Function (PDF) Plots__

We can make 1D plots that show the errors of the light models estimated via a model-fit.

Here, the `light_profile_pdf_list` is a list of `LightProfile` objects drawn randomly from the PDF of a model-fit.

These are used to estimate the errors at an input `sigma` value of:

 - The 1D light profile, plotted as a shaded region on the figure.
 - The median `half_light_radius` with errors, plotted as vertical lines.

Below, we manually input two light profiles to demonstrate how these errors appear on the figure.
"""
light_profile_pdf_list = [bulge, disk]

sigma = 3.0
low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

image_1d_list = []

for light_profile in light_profile_pdf_list:
    grid_projected = grid.grid_2d_radial_projected_from(
        centre=light_profile.centre, angle=light_profile.angle()
    )

    image_1d_list.append(light_profile.image_2d_from(grid=grid_projected))

min_index = min([image_1d.shape[0] for image_1d in image_1d_list])
image_1d_list = [image_1d[0:min_index] for image_1d in image_1d_list]

(
    median_image_1d,
    errors_image_1d,
) = ag.util.error.profile_1d_median_and_error_region_via_quantile(
    profile_1d_list=image_1d_list, low_limit=low_limit
)

grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=bulge.centre, angle=bulge.angle()
)

plt.plot(
    grid_2d_projected[:min_index, 1], median_image_1d, label="Median Light Profile"
)
plt.fill_between(
    x=grid_2d_projected[:min_index, 1],
    y1=errors_image_1d[0],
    y2=errors_image_1d[1],
    color="lightgray",
    label=f"{sigma} Sigma Error Region",
)

"""
Finish.
"""
