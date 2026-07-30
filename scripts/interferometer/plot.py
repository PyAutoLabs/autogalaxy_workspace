"""
Plots: Interferometer
=====================

This example shows how to plot an `Interferometer` dataset and a `FitInterferometer` fit, figure
by figure and via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autogalaxy.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`. For the full description of the analogous imaging figures and the
`Visualizer` mechanism, see `scripts/imaging/plot.py`.

__Contents__

- **Dataset:** Load the interferometer dataset used throughout this example.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Dirty Images Subplot:** Plot the dirty image, dirty noise-map and dirty signal-to-noise map.
- **Fit:** Set up a galaxy and fit the dataset with a `FitInterferometer` object.
- **Fit Figures:** Plot the fit's dirty-image quantities individually.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Fit Dirty Images Subplot:** Plot the fit's dirty images in one multi-panel subplot.
- **Real Space Subplot:** Plot the fit's real-space image and reconstruction.
- **Outputting to FITS:** Write the dataset to a FITS file instead of an image.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

We define the `real_space_mask` which defines the grid the image is evaluated using, then load the
interferometer dataset `simple` from .fits files, which is the dataset used to demonstrate plotting.

We use `TransformerNUFFT`, the JAX-native Non-Uniform Fast Fourier Transform backed by `nufftax`,
which Fourier transforms the real-space galaxy image to the uv-plane to compare directly to the
observed visibilities.
"""
mask_radius = 3.5

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerNUFFT,
)

"""
__Dataset Subplot__

A multi-panel subplot of the dataset is produced with `aplt.subplot_interferometer_dataset()`,
including the observed visibility data, RMS noise-map and other information. Visibility data is in
uv-space, making it hard to interpret by eye.
"""
aplt.subplot_interferometer_dataset(dataset=dataset)

"""
__Dirty Images Subplot__

The dirty images of the interferometer dataset map the visibilities, noise-map and
signal-to-noise map to real-space images using the dataset's transformer, making them easier to
interpret by eye than the uv-space visibilities.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Fit__

To plot a fit, we fit the dataset with a galaxy whose light profile matches the true simulated
values, via the `FitInterferometer` object.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

"""
__Fit Figures__

The fit's visibilities and residuals are in the uv-plane, making them hard to interpret by eye. The
fit therefore provides `dirty_*` variants of its quantities, which use the transformer to map them
to real-space images, plotted individually with `aplt.plot_array()`.
"""
aplt.plot_array(array=fit.dirty_image, title="Dirty Image")
aplt.plot_array(array=fit.dirty_model_image, title="Dirty Model Image")
aplt.plot_array(array=fit.dirty_residual_map, title="Dirty Residual Map")
aplt.plot_array(
    array=fit.dirty_normalized_residual_map, title="Dirty Normalized Residual Map"
)
aplt.plot_array(array=fit.dirty_chi_squared_map, title="Dirty Chi-Squared Map")

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_interferometer()`, combining the
dirty image, dirty model image, dirty residual-map and dirty chi-squared map in one figure.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
__Fit Dirty Images Subplot__

`aplt.subplot_fit_dirty_images()` collects the fit's dirty images — data, model data, residuals and
chi-squared — into a single subplot, the fit-level counterpart of the dataset subplot above.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Real Space Subplot__

`aplt.subplot_fit_real_space()` plots the fit's real-space quantities: the image of the galaxies
evaluated on the `real_space_mask`, and the reconstruction. These are the quantities being Fourier
transformed to the uv-plane, so they show what the model actually looks like on the sky.
"""
aplt.subplot_fit_real_space(fit=fit)

"""
__Outputting to FITS__

The dataset itself can be written to a FITS file with `aplt.fits_interferometer()`. Passing
`file_path` writes a single multi-HDU file; passing `data_path` / `noise_map_path` /
`uv_wavelengths_path` instead writes each component to its own file.
"""
output_path = Path("output") / "plot" / "interferometer"
output_path.mkdir(parents=True, exist_ok=True)

aplt.fits_interferometer(
    dataset=dataset,
    file_path=output_path / "dataset.fits",
    overwrite=True,
)

"""
__Visualizer__

During a model-fit, all of the figures above are output to hard-disk automatically by the
`Visualizer` attached to the `Analysis` class, exactly as described for imaging in
`scripts/imaging/plot.py`.
"""
print(ag.AnalysisInterferometer.Visualizer)

"""
Which figures are output is controlled by the config file `config/visualize/plots.yaml`, under the
`fit_interferometer` entry (e.g. `subplot_fit_dirty_images`), in addition to the shared `dataset`
and `fit` entries described in `scripts/imaging/plot.py`.
"""
