"""
Plots: Multi
============

This example shows how to plot multiple datasets — and fits to multiple datasets — together,
with every dataset appearing in one combined subplot.

This uses the same functions the source code's `Visualizer` uses when it outputs figures during a
multi-dataset model-fit:

 - `aplt.subplot_imaging_dataset_list()` — all datasets in one subplot (one row per dataset).
 - `aplt.subplot_fit_imaging_list()` — all fits in one subplot (one row per fit).

The specific example loads a multi-wavelength imaging dataset and plots the g-band and r-band
data and fits together. For an introduction to the plotting API refer to
`guides/plot/start_here.py`; for single-dataset fit plotting refer to `scripts/imaging/plot.py`.

__Contents__

- **Dataset:** Load the multi-wavelength galaxy datasets.
- **Single Dataset Subplots:** Plot the subplot overview of each dataset one-by-one.
- **Combined Dataset Subplot:** Plot all datasets in one subplot with `aplt.subplot_imaging_dataset_list()`.
- **Fits:** Fit each waveband's dataset with galaxies using their true simulated values.
- **Combined Fit Subplot:** Plot all fits in one subplot with `aplt.subplot_fit_imaging_list()`.
- **Multi Fits:** Output a list of figures to a single `.fits` file, where each image goes in each HDU.
- **Visualizer:** How combined figures are output automatically during a multi-dataset model-fit.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the multi-wavelength `simple` datasets.
"""
waveband_list = ["g", "r"]

pixel_scales_list = [0.08, 0.12]

dataset_type = "multi_dataset"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_dataset/simulator.py"],
        check=True,
    )

dataset_list = [
    ag.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{waveband}_data.fits",
        psf_path=Path(dataset_path) / f"{waveband}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{waveband}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for waveband, pixel_scales in zip(waveband_list, pixel_scales_list)
]

"""
__Single Dataset Subplots__

Each dataset's subplot overview can be plotted one-by-one with `aplt.subplot_imaging_dataset()`.
"""
for dataset in dataset_list:
    aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Combined Dataset Subplot__

To compare the datasets it is more useful to see them in a single figure. The
`aplt.subplot_imaging_dataset_list()` function plots every dataset in one subplot, with one row
per dataset showing its data, noise-map and signal-to-noise map.
"""
aplt.subplot_imaging_dataset_list(dataset_list=dataset_list)

"""
__Fits__

To plot fits to every dataset, we mask each dataset and fit it with galaxies using the true
simulated values.

The galaxy's bulge and disk have a different `intensity` at each wavelength (see
`scripts/multi_dataset/simulator.py`), so separate galaxies are composed per waveband.
"""
dataset_list = [
    dataset.apply_mask(
        mask=ag.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=3.0,
        )
    )
    for dataset in dataset_list
]

bulge_intensity_list = [0.2, 0.4]
disk_intensity_list = [0.2, 0.5]

galaxies_list = [
    ag.Galaxies(
        galaxies=[
            ag.Galaxy(
                redshift=0.5,
                bulge=ag.lp.Sersic(
                    centre=(0.0, 0.0),
                    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                    intensity=bulge_intensity,
                    effective_radius=0.6,
                    sersic_index=3.0,
                ),
                disk=ag.lp.Exponential(
                    centre=(0.0, 0.0),
                    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
                    intensity=disk_intensity,
                    effective_radius=1.6,
                ),
            ),
        ]
    )
    for bulge_intensity, disk_intensity in zip(
        bulge_intensity_list, disk_intensity_list
    )
]

fit_list = [
    ag.FitImaging(dataset=dataset, galaxies=galaxies)
    for dataset, galaxies in zip(dataset_list, galaxies_list)
]

"""
__Combined Fit Subplot__

The `aplt.subplot_fit_imaging_list()` function plots every fit in one subplot, with one row per
fit showing its data, model image and residuals.

This is the figure to inspect when checking that a multi-wavelength model fits all datasets well
simultaneously.
"""
aplt.subplot_fit_imaging_list(fit_list=fit_list)

"""
__Multi Fits__

We can also output a list of figures to a single `.fits` file, where each image goes in
each HDU extension.
"""
from autogalaxy import hdu_list_for_output_from

dataset = dataset_list[-1]

image_list = [dataset.data, dataset.noise_map]

hdu_list = hdu_list_for_output_from(
    values_list=[image_list[0].mask.astype("float")] + image_list,
    ext_name_list=["mask"] + ["data", "noise_map"],
    header_dict=dataset.mask.header_dict,
)

hdu_list.writeto("dataset.fits", overwrite=True)

"""
__Visualizer__

During a multi-dataset model-fit (e.g. combining analyses with `af.AnalysisFactor` as in
`scripts/multi_dataset/modeling.py`), the `Visualizer` attached to the `Analysis` class outputs the
combined figures above automatically:

 - Before the fit begins, all datasets are output together via `subplot_imaging_dataset_list`.
 - During and after the fit, the maximum likelihood fit to every dataset is output together via
   `subplot_fit_imaging_list`.

These appear in the fit's output folder under `image/` (e.g. `dataset_combined.png`,
`fit_combined.png`), alongside the per-dataset figures described in `scripts/imaging/plot.py`.

Which figures are output is controlled by `config/visualize/plots.yaml`, e.g. the
`dataset` -> `subplot_dataset` and `fit` -> `subplot_fit` entries.
"""
