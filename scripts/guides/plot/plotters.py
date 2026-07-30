"""
Plots: Objects
==============

This example illustrates how to plot each key PyAutoGalaxy object — light profiles, galaxies and
collections of galaxies — figure by figure.

Every object follows the same pattern: a quantity is computed via the object's method
(e.g. `image_2d_from()`) and the resulting array or grid is passed to `aplt.plot_array()` or
`aplt.plot_grid()`.

For an introduction to the plotting API itself (customization, output, config defaults, overlays,
subplots) refer to `guides/plot/start_here.py`. For plotting datasets and fits (e.g. `FitImaging`),
refer to the `plot.py` example of each dataset package (e.g. `scripts/imaging/plot.py`).

__Contents__

- **Setup:** Set up the light profiles and galaxies used throughout this example.
- **Light Profile:** A light profile image is computed via `image_2d_from()` and plotted with `aplt.plot_array()`.
- **Galaxy:** A galaxy's image sums the images of its light profiles (e.g. bulge and disk).
- **Galaxies:** The summed image of all galaxies, and a per-galaxy subplot via `aplt.subplot_galaxies()`.
- **Log10:** Plot galaxy images in log10 space for clearer visualization.
- **One Dimensional Plots:** Plot 1D radial profiles using standard matplotlib.
- **Probability Density Function (PDF) Plots:** Plot 1D light profiles with error regions from model-fit PDFs.

__Setup__

To illustrate plotting, we set up a grid, light profiles and galaxies.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import math
import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

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

"""
__Light Profile__

A light profile's image is computed via `image_2d_from` and then plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=bulge.image_2d_from(grid=grid), title="Bulge Image")

"""
__Galaxy__

A galaxy's image is computed via `image_2d_from` and then plotted using `aplt.plot_array`.

This sums the images of all light profiles the galaxy contains (here, its bulge and disk).
"""
aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Galaxy Image")

"""
__Galaxies__

The summed image of all galaxies can be plotted using `aplt.plot_array`.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Galaxies Image")

"""
A subplot showing each individual galaxy's image side-by-side can be plotted using `aplt.subplot_galaxies`.
"""
aplt.subplot_galaxies(galaxies=galaxies, grid=grid)

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
__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
