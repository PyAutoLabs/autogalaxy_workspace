"""
Plots: MultiSubPlots
====================

This example illustrates how to produce multiple subplot figures from different data objects, using the
function-based plotting API.

In the new API, each subplot function (e.g. `subplot_imaging_dataset`, `subplot_fit_imaging`,
`subplot_galaxies`) produces a self-contained subplot. There is no need to manually manage MatPlot objects,
open/close subplot figures, or chain plotters together.

To show multiple subplots for a given dataset and fit, simply call each function in sequence.

__Contents__

- **Imaging Dataset Subplot:** Plotting a subplot of the imaging dataset.
- **Fit Imaging Subplot:** Plotting a subplot of the fit to the imaging dataset.
- **Galaxies Subplot:** Plotting a subplot of the galaxy images.
- **Output:** Saving each subplot to disk as PNG files.
- **Wrap Up:** Summary of the function-based subplot plotting API.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
First, load example imaging of a galaxy and create a `FitImaging` object.
"""
dataset_name = "simple__sersic"
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
        [sys.executable, "scripts/imaging/simulator_sersic.py"],
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

galaxy = ag.Galaxy(
    redshift=1.0,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Imaging Dataset Subplot__

Plot a subplot of the imaging dataset, showing the data, noise map and PSF.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit Imaging Subplot__

Plot a subplot of the fit to the imaging dataset, showing the data, model image, residuals and chi-squared map.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Galaxies Subplot__

Plot a subplot of the galaxies, showing the image of each individual galaxy.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

aplt.subplot_galaxies(galaxies=galaxies, grid=grid)

"""
__Output__

Each subplot function accepts `output_path`, `output_filename` and `output_format` arguments to save the
subplot to disk as a file.

Below we output all three subplots to `.png` files.
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

aplt.subplot_galaxies(
    galaxies=galaxies,
    grid=grid,
    output_path=dataset_path,
    output_format="png",
)

"""
__Wrap Up__

In the new API, each subplot function is self-contained and independent. To produce multiple subplots for a
given analysis, simply call the relevant functions one after another, optionally providing output arguments to
save each subplot to disk.
"""
