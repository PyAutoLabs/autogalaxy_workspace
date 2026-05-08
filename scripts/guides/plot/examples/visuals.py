"""
Plots: Visuals
==============

This example illustrates the API for adding visuals to plots.

Visuals are passed as direct keyword arguments to `plot_array` (or `plot_grid`).

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

- **Setup:** Set up objects (grid, galaxies, dataset) used to illustrate visual overlays.
- **Light Profile Centres:** Overlay light profile centres on an image using positions.
- **Mask:** Overlay a mask on an image.
- **Origin:** Overlay the coordinate origin on an image.
- **Grid:** Overlay a grid of (y,x) coordinates on an image.
- **Border:** Overlay the border of a mask on an image.
- **Array Overlay:** Overlay a 2D array on an image.
- **Patch Overlay:** Overlay matplotlib patches (e.g. ellipses) on an image.
- **Vector Field:** Overlay a quiver plot of vectors on an image.

__Setup__

To illustrate plotting, we require standard objects like a grid, galaxies and dataset.
"""

# from autoconf import setup_notebook; setup_notebook()

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
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/guides/plot/simulator.py"],
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
__Mask__

The mask is plotted over an image by passing `mask=` to `plot_array`.

The mask is a `Mask2D` object, which is passed directly.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)

aplt.plot_array(array=data, title="Image with Mask")

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
__Border__

A border is the `Grid2D` of (y,x) coordinates at the centre of every pixel at the border of a mask.

A border is defined as a pixel that is on an exterior edge of a mask (e.g. it does not include the inner pixels of
an annular mask).

Borders are rarely plotted, but are important when it comes to defining the edge of the source-plane for pixelized
source reconstructions, with examples on this topic sometimes plotting the border.

The border is passed via `border=` to `plot_array`.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)

aplt.plot_array(
    array=data,
    title="Image with Border",
)

"""
__Array Overlay__

We can overlay a 2D array over an image by passing `grid=` to `plot_array`.
"""
arr = ag.Array2D.no_mask(
    values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=0.5
)

aplt.plot_array(array=data, title="Image with Array Overlay")

"""
__Patch Overlay__

The matplotlib patch API can be used to plot shapes over an image by passing `patches=` to `plot_array`.

This is used in certain weak lensing plots to visualize galaxy shapes and ellipticities.

In this example, we will use the `Ellipse` patch.
"""
from matplotlib.patches import Ellipse

patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
patch_1 = Ellipse(xy=(-2.0, -3.0), height=1.0, width=2.0, angle=1.0)

aplt.plot_array(array=data, title="Image with Patches")

"""
__Vector Field__

A quiver plot showing vectors (e.g. 2D (y,x) directions at (y,x) coordinates) can be plotted by passing
`vector_yx=` to `plot_array`.

This is often used for weak lensing plots, showing the direction and magnitude of weak lensing inferred at each
galaxy's location.
"""
vectors = ag.VectorYX2DIrregular(
    values=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)

aplt.plot_array(array=data, title="Image with Vector Field")

"""
Finish.
"""
