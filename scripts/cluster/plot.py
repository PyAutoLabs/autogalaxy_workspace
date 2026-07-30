"""
Plots: Cluster
==============

This example shows how to plot an `Imaging` dataset and a `FitImaging` fit of a cluster field —
a brightest cluster galaxy (BCG) plus a catalogue-driven member population — figure by figure and
via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autogalaxy.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`. For the full description of these figures and the `Visualizer`
mechanism, see `scripts/imaging/plot.py`.

__Contents__

- **Dataset:** Load the cluster imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up the BCG and catalogue-driven members and fit the dataset with a `FitImaging` object.
- **Fit Figures:** Plot the fit's model image, residuals and chi-squared maps individually.
- **Galaxy Images:** Plot the model image of the BCG, decomposed from the member population.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the cluster dataset `simple` (1 BCG + 10 members).
"""
dataset_name = "simple"
dataset_path = Path("dataset", "cluster", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/cluster/simulator.py"],
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

To plot a fit, we mask the field, load the member catalogue with `ag.galaxy_table_from_csv` and the
BCG centre with `ag.from_json`, and fit the dataset with the BCG plus catalogue-driven members whose
light profiles match the true simulated values, via the `FitImaging` object. The BCG is modeled
individually while every member's intensity follows its catalogue luminosity.
"""
scaling_table = ag.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

member_centres = scaling_table.centres.in_list
member_luminosities = scaling_table.luminosities

bcg_centres = ag.from_json(file_path=dataset_path / "bcg_centres.json")

mask_radius = 11.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

bcg = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=bcg_centres[0],
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.5,
        effective_radius=2.5,
        sersic_index=4.0,
    ),
)

members = []
for centre, luminosity in zip(member_centres, member_luminosities):
    members.append(
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp.SersicSph(
                centre=centre,
                intensity=luminosity,
                effective_radius=0.6,
                sersic_index=3.0,
            ),
        )
    )

galaxies = ag.Galaxies(galaxies=[bcg] + members)

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

The BCG's model image is accessed via `model_images_of_galaxies_list[0]` — the member-subtracted
BCG photometry that motivates cluster light modeling.
"""
aplt.plot_array(array=fit.model_images_of_galaxies_list[0], title="BCG Model Image")

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`, combining the data, model
image, residual-map and chi-squared map in one figure. `aplt.subplot_fit_imaging_of_galaxy()` shows
the BCG's decomposed light separately from the member population.
"""
aplt.subplot_fit_imaging(fit=fit)

aplt.subplot_fit_imaging_of_galaxy(fit=fit, galaxy_index=0)

"""
__Visualizer__

During a model-fit, all of the figures above are output to hard-disk automatically by the
`Visualizer` attached to the `Analysis` class, exactly as described in `scripts/imaging/plot.py`
(the machinery is identical — only the model composition changes).
"""
print(ag.AnalysisImaging.Visualizer)

"""
Which figures are output is controlled by the config file `config/visualize/plots.yaml`, in
particular `fit` -> `subplot_of_galaxies`, which switches the BCG breakdown shown above on or off
for every model-fit.
"""
