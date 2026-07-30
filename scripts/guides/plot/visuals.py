"""
Plots: Visuals
==============

This example illustrates the API for adding visuals to plots.

Visuals are passed as direct keyword arguments to `plot_array` (or `plot_grid`), for example `positions=`,
`grid=` and `lines=`.

__Start Here Notebook__

You should refer to the `guides/plot/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

- **Setup:** Set up objects (grid, galaxies, dataset) used to illustrate visual overlays.
- **Light Profile Centres:** Overlay light profile centres on an image using positions.
- **Origin:** Overlay the coordinate origin on an image.
- **Grid:** Overlay a grid of (y,x) coordinates on an image.

__Setup__

To illustrate plotting, we require standard objects like a grid, galaxies and dataset.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

galaxy = ag.Galaxy(
    redshift=1.0,
    bulge_0=ag.lp.SersicSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
    bulge_1=ag.lp.SersicSph(
        centre=(0.4, 0.3), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

dataset_path = Path("dataset") / "imaging" / "simple"

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

data_path = dataset_path / "data.fits"
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
__Light Profile Centres__

The centres of all light profiles in the galaxies can be extracted and overlaid on the image by passing
`positions=` to `plot_array`.

Each entry in `positions` is a numpy array of shape `(N, 2)` containing `(y, x)` coordinates.
"""
light_profile_centres = galaxies.extract_attribute(
    cls=ag.LightProfile, attr_name="centre"
)

image = galaxies.image_2d_from(grid=grid)

positions = [np.array(light_profile_centres)]

aplt.plot_array(
    array=image, positions=positions, title="Image with Light Profile Centres"
)

"""
__Origin__

We can overlay the (y,x) origin on the data to show where the coordinate system is defined from.

By default the origin of (0.0", 0.0") is at the centre of the image.

Origins are passed as `positions=`, where each entry is a numpy array of shape `(N, 2)`.
"""
origin = np.array([[1.0, 1.0]])

aplt.plot_array(array=data, positions=[origin], title="Image with Origin")

"""
__Grid__

We can overlay a grid of (y,x) coordinates over an image by passing `grid=` to `plot_array`.

We'll use a uniform grid at a coarser resolution than our dataset.
"""
coarse_grid = ag.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)

aplt.plot_array(array=data, grid=np.array(coarse_grid), title="Image with Grid")

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
