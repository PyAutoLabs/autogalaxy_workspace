"""
Plots: Multi Galaxy
====================

This example shows how to plot an `Imaging` dataset and a `FitImaging` fit of a multi-galaxy
system, figure by figure and via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autogalaxy.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`. For the full description of these figures and the `Visualizer`
mechanism, see `scripts/imaging/plot.py`.

__Contents__

- **Dataset:** Load the multi-galaxy imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up the two blended galaxies and fit the dataset with a `FitImaging` object.
- **Fit Figures:** Plot the fit's model image, residuals and chi-squared maps individually.
- **Galaxy Images:** Plot the model image of each of the two blended galaxies separately.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the multi-galaxy dataset `simple`: imaging of a simulated close pair of blended galaxies.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
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

To plot a fit, we mask the dataset (enclosing both blended galaxies) and fit it with the two
galaxies whose light profiles match the true simulated values, via the `FitImaging` object. Each
galaxy is modeled with its own light profile, the defining feature of the multi-galaxy regime.
"""
mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

galaxy_centres = [(0.0, -0.75), (0.0, 0.75)]

galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=galaxy_centres[0],
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=30.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=galaxy_centres[1],
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=120.0),
        intensity=0.7,
        effective_radius=1.0,
        sersic_index=1.5,
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

Per-galaxy model images are accessed via `model_images_of_galaxies_list`, letting us inspect each
of the two blended galaxies' contributions to the model image separately — the deliverable of a
blended-pair fit.
"""
aplt.plot_array(
    array=fit.model_images_of_galaxies_list[0], title="Galaxy 0 Model Image"
)
aplt.plot_array(
    array=fit.model_images_of_galaxies_list[1], title="Galaxy 1 Model Image"
)

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`, combining the data, model
image, residual-map and chi-squared map in one figure. `aplt.subplot_fit_imaging_of_galaxy()` shows
each galaxy's decomposed light separately.
"""
aplt.subplot_fit_imaging(fit=fit)

for i in range(len(galaxy_centres)):
    aplt.subplot_fit_imaging_of_galaxy(fit=fit, galaxy_index=i)

"""
__Visualizer__

During a model-fit, all of the figures above are output to hard-disk automatically by the
`Visualizer` attached to the `Analysis` class, exactly as described in `scripts/imaging/plot.py`
(the machinery is identical — only the number of galaxies in the model changes).
"""
print(ag.AnalysisImaging.Visualizer)

"""
Which figures are output is controlled by the config file `config/visualize/plots.yaml`, in
particular `fit` -> `subplot_of_galaxies`, which switches the per-galaxy breakdown shown above on
or off for every model-fit.
"""
