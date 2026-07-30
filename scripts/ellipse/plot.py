"""
Plots: Ellipse
==============

This example shows how to plot an `Imaging` dataset and a `FitEllipse` fit, figure by figure and
via multi-panel subplots.

Ellipse fitting is a non-parametric measure of galaxy morphology: rather than creating a 2D model
image to subtract from the data, it interpolates the data and noise-map onto the coordinates of an
ellipse and quantifies how closely those interpolated values agree with one another. Quantities are
therefore 1D arrays along the ellipse rather than 2D images, and are inspected via `print()` rather
than `aplt.plot_array()`.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autogalaxy.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data and noise-map individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up an ellipse and fit the dataset with a `FitEllipse` object.
- **Fit Figures:** Inspect the fit's model data, residuals and chi-squared values.
- **Fit Subplot:** Plot the fit's ellipse contours and 1D residuals in one multi-panel subplot.
- **Multiple Ellipses:** Fit and plot several ellipses of increasing size together.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the galaxy dataset `ellipse` from .fits files, which is the dataset used to demonstrate
plotting. Ellipse fitting does not use the Point Spread Function (PSF) of the dataset, so we do not
need to load it.
"""
dataset_name = "ellipse"
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
        [sys.executable, "scripts/ellipse/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Dataset Figures__

The dataset's data and noise-map are attributes, each plotted individually with
`aplt.plot_array()`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")

"""
__Dataset Subplot__

A multi-panel subplot of the dataset is produced with `aplt.subplot_imaging_dataset()`, including
the data, noise-map and signal-to-noise map.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit__

To plot a fit, we mask the dataset and fit it with an `Ellipse` via the `FitEllipse` object.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse)

"""
__Fit Figures__

The fit's quantities are 1D arrays interpolated onto the ellipse's coordinates, accessed as
attributes and inspected with `print()`:

 - `data_interp` / `noise_map_interp`: The dataset's data and noise-map interpolated onto the ellipse.
 - `model_data`: The data values interpolated onto the ellipse (identical to `data_interp`).
 - `residual_map`: The model data minus its mean.
 - `normalized_residual_map`: The `residual_map` divided by `noise_map_interp`.
 - `chi_squared_map`: The `normalized_residual_map` squared.
"""
print("Data Values Interpolated to Ellipse:")
print(fit.data_interp)
print("Noise Values Interpolated to Ellipse:")
print(fit.noise_map_interp)
print("Model Data Values:")
print(fit.model_data)
print("Residuals:")
print(fit.residual_map)
print("Normalized Residuals:")
print(fit.normalized_residual_map)
print("Chi-Squareds:")
print(fit.chi_squared_map)

"""
There are also single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.
 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -2.0 * chi_squared.
"""
print(fit.chi_squared)
print(fit.log_likelihood)

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_ellipse()`, which plots the data with
the fitted ellipse contours overlaid alongside the 1D residuals as a function of position angle.
"""
aplt.subplot_fit_ellipse(fit_list=[fit])

"""
__Multiple Ellipses__

It is rare to use only one ellipse to fit a galaxy, as the goal of ellipse fitting is to find the
collection of ellipses that best trace round the data. We fit a list of ellipses with the same
`centre` and `ell_comps` but growing `major_axis` values, and plot every one of them in a single
multi-panel subplot.
"""
major_axis_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

ellipse_list = [
    ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.5), major_axis=major_axis)
    for major_axis in major_axis_list
]

fit_list = [ag.FitEllipse(dataset=dataset, ellipse=ellipse) for ellipse in ellipse_list]

aplt.subplot_fit_ellipse(fit_list=fit_list)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), all of
the figures above are output to hard-disk automatically — you do not need to call the plotting
functions yourself to inspect a fit's progress or results.

This is performed by the `Visualizer` attached to the `Analysis` class:
"""
print(ag.AnalysisEllipse.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
computes the maximum likelihood fit and outputs its figures to the fit's output folder, under
`image/` (e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`, under the
`fit_ellipse` entry (e.g. `data`, `data_no_ellipse`).

Setting an entry to `true` or `false` in `plots.yaml` therefore switches that figure on or off
for every model-fit, without changing code.
"""
