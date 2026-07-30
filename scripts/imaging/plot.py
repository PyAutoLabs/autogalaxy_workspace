"""
Plots: Imaging
==============

This example shows how to plot an `Imaging` dataset and a `FitImaging` fit, figure by figure and
via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autogalaxy.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up galaxies and fit the dataset with a `FitImaging` object.
- **Fit Figures:** Plot the fit's model image, residuals and chi-squared maps individually.
- **Galaxy Images:** Plot the model image of each individual galaxy in the fit.
- **Galaxy Subplots:** Plot a multi-panel subplot for each galaxy of the fit.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Outputting to FITS:** Write the dataset to a FITS file instead of an image.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the galaxy dataset `sersic_x2` from .fits files, which is the dataset used to demonstrate
plotting.
"""
dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
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

"""
__Dataset Figures__

The dataset's data, noise-map and PSF are attributes, each plotted individually with
`aplt.plot_array()`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")
aplt.plot_array(array=dataset.psf.kernel, title="PSF")

"""
__Dataset Subplot__

A multi-panel subplot of the dataset is produced with `aplt.subplot_imaging_dataset()`, including
the data, noise-map, PSF and signal-to-noise map.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit__

To plot a fit, we mask the dataset and fit it with galaxies whose light profiles match the true
simulated values, via the `FitImaging` object.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

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
__Fit Figures__

The fit's quantities — model image, residuals, chi-squared and more — are accessed as attributes
and plotted individually with `aplt.plot_array()`.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.noise_map, title="Noise Map")
aplt.plot_array(array=fit.signal_to_noise_map, title="Signal-to-Noise Map")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
__Galaxy Images__

Per-galaxy model images are accessed via `model_images_of_galaxies_list`, for example to inspect
each of the two blended galaxies' contributions to the model image separately.
"""
aplt.plot_array(
    array=fit.model_images_of_galaxies_list[0], title="Galaxy 0 Model Image"
)
aplt.plot_array(
    array=fit.model_images_of_galaxies_list[1], title="Galaxy 1 Model Image"
)

"""
__Galaxy Subplots__

`aplt.subplot_fit_imaging_of_galaxy()` produces a full subplot for a single galaxy of the fit,
showing the data with the other galaxies subtracted alongside that galaxy's model image and
residuals. The galaxy is selected with `galaxy_index`.
"""
aplt.subplot_fit_imaging_of_galaxy(fit=fit, galaxy_index=0)
aplt.subplot_fit_imaging_of_galaxy(fit=fit, galaxy_index=1)

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`, combining the data, model
image, residual-map and chi-squared map in one figure.

Pass `use_log10=True` to plot the image panels on a `log10` colour scale, which makes faint
extended emission visible.
"""
aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging(fit=fit, use_log10=True)

"""
__Outputting to FITS__

Figures are images, but the dataset itself can be written to a FITS file with
`aplt.fits_imaging()`. Passing `file_path` writes a single multi-HDU file with named extensions
(`mask`, `data`, `psf`, `noise_map`); passing `data_path` / `psf_path` / `noise_map_path` instead
writes each component to its own file.
"""
output_path = Path("output") / "plot" / "imaging"
output_path.mkdir(parents=True, exist_ok=True)

aplt.fits_imaging(
    dataset=dataset,
    file_path=output_path / "dataset.fits",
    overwrite=True,
)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), all of
the figures above are output to hard-disk automatically — you do not need to call the plotting
functions yourself to inspect a fit's progress or results.

This is performed by the `Visualizer` attached to the `Analysis` class:
"""
print(ag.AnalysisImaging.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
computes the maximum likelihood fit and outputs its figures to the fit's output folder, under
`image/` (e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`, for
example:

 - `dataset` -> `subplot_dataset`: The multi-panel dataset subplot shown above.
 - `fit` -> `subplot_fit`: The multi-panel fit subplot shown above.
 - `fit` -> `subplot_of_galaxies`: The per-galaxy model image breakdown shown above.

Setting an entry to `true` or `false` in `plots.yaml` therefore switches that figure on or off
for every model-fit, without changing code.

When multiple datasets are fitted simultaneously, the `Visualizer` also outputs figures combining
all datasets in one subplot — see `scripts/multi/plot.py` for details.
"""
