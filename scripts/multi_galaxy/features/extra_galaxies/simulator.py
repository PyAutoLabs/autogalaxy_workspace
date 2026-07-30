"""
Simulator: Extra Galaxies (Multi Galaxy)
========================================

A multi-galaxy system — two galaxies whose light blends together and which are co-equal subjects of the fit —
frequently sits in a field with other, fainter galaxies. Those extra galaxies are not part of the blend being
decomposed, but their emission still falls inside the mask and will bias the fit if nothing is done about it.

This script simulates `Imaging` of a blended pair with two such companions, and is used to illustrate the extra
galaxies API in `autogalaxy_workspace/*/multi_galaxy/features/extra_galaxies/modeling`.

The distinction the dataset is built to make concrete is a **tier** distinction:

 - The two galaxies of the pair are **co-equal**. Each gets its own free light model, because neither can be
   called a contaminant of the other — that is the definition of the multi-galaxy regime.
 - The two extra galaxies are **sub-dominant**. They are fainter, further out, and are either scaled out of the
   fit or given a restricted light model with a fixed centre.

Getting a galaxy into the right tier is the modeling decision this feature exists to support.

__Contents__

- **Model:** The system being simulated.
- **Other Scripts:** This dataset is used in the following scripts.
- **Dataset Paths:** The dataset folder the simulated data is output to.
- **Grid / PSF / Simulator:** Standard imaging simulation setup, over-sampled at all four galaxy centres.
- **Galaxies:** The two co-equal blended galaxies.
- **Extra Galaxies:** The two sub-dominant companions.
- **Dataset:** Simulate and write the imaging dataset.
- **Mask Extra Galaxies:** Build and save `mask_extra_galaxies.fits` for the noise-scaling approach.
- **Galaxies json + Centres:** Truth records and the two centre files the modeling script loads.

__Model__

This script simulates `Imaging` of a multi-galaxy system where:

 - Two co-equal galaxies (~1.5" separation) have blended `Sersic` light distributions of comparable brightness.
 - Two extra galaxies further out have fainter `ExponentialSph` light distributions.

__Other Scripts__

This dataset is used in the following script:

 `autogalaxy_workspace/*/multi_galaxy/features/extra_galaxies/modeling.ipynb`

To illustrate how to compose and fit a multi-galaxy model which accounts for the extra galaxies, either by
scaling their emission out of the fit or by including them in the model.

__Start Here Notebook__

If any code in this script is unclear, refer to the `multi_galaxy/simulator.ipynb` notebook.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The dataset is output to `/autogalaxy_workspace/dataset/multi_galaxy/extra_galaxies`.
"""
dataset_type = "multi_galaxy"
dataset_name = "extra_galaxies"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid / PSF / Simulator__

The over sampling scheme is applied at the centre of **all four** galaxies — the two of the pair and the two
extras — so that every galaxy's light is evaluated accurately.
"""
grid = ag.Grid2D.uniform(
    shape_native=(180, 180),
    pixel_scales=0.1,
)

galaxy_centres = [(0.0, -0.75), (0.0, 0.75)]

extra_galaxy_0_centre = (3.5, 2.5)
extra_galaxy_1_centre = (-3.0, -3.0)

extra_galaxy_centres = [extra_galaxy_0_centre, extra_galaxy_1_centre]

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=galaxy_centres + extra_galaxy_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = ag.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Galaxies__

The two co-equal blended galaxies, identical to those simulated by `multi_galaxy/simulator.py`: comparable
brightness (the second is ~70% as bright as the first), overlapping light distributions, different morphologies
(one de Vaucouleurs-like bulge, one disky Sersic). The ~1.5" separation is small enough that each galaxy's light
contaminates the other's centre.
"""
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

"""
__Extra Galaxies__

The two sub-dominant companions, placed at ~4.3" from the field centre — well outside the pair, but comfortably
inside the larger mask the modeling script uses to admit them.

They differ from the pair in exactly the ways that put them in a lower tier:

 - **Fainter**, at `intensity=0.3` against the pair's 1.0 and 0.7.
 - **Spherical and simple**, `ExponentialSph` rather than an elliptical `Sersic` — there is not enough signal in
   a faint companion to constrain a richer morphology, which is why the modeling script gives them a restricted
   light model.
 - **Well separated** from the pair, so their light does not blend into the decomposition being measured. This
   is what makes scaling them out of the fit a legitimate option, where doing the same to either galaxy of the
   pair would destroy the analysis.

Note their redshift is the same as the pair, which is not necessarily the case in real observations. PyAutoGalaxy
models light only, so a different redshift changes nothing about the fit — unlike the lensing case, where extra
galaxies at a different redshift trigger multi-plane ray-tracing.
"""
extra_galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.ExponentialSph(
        centre=extra_galaxy_0_centre, intensity=0.3, effective_radius=0.5
    ),
)

extra_galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.ExponentialSph(
        centre=extra_galaxy_1_centre, intensity=0.3, effective_radius=0.7
    ),
)

galaxies = ag.Galaxies(
    galaxies=[galaxy_0, galaxy_1, extra_galaxy_0, extra_galaxy_1]
)

"""
__Dataset__

Simulate the imaging dataset from all four galaxies and write it to the dataset folder.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Mask Extra Galaxies__

Build and output a `mask_extra_galaxies.fits` covering the two extra galaxy regions, so the modeling script can
load it directly and demonstrate the noise-scaling approach without a separate data-preparation step.

Each circle is sized to ~3x the galaxy's `effective_radius`, which comfortably covers the light extent of the
`ExponentialSph` profiles above. The geometry is derived from the same centres and radii defined for the extra
galaxies in this script, so it stays in sync with any future tweak to those values.

`Mask2D.circular` honours the `PYAUTO_SMALL_DATASETS=1` env var (caps to 15x15 at 0.6"/px), so the mask
automatically shrinks alongside the small-dataset image and never raises an out-of-bounds error.
"""
extra_galaxies_mask = np.zeros(dataset.shape_native, dtype=bool)

for centre, radius in [
    (extra_galaxy_0_centre, 3.0 * 0.5),
    (extra_galaxy_1_centre, 3.0 * 0.7),
]:
    circle = ag.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        centre=centre,
        radius=radius,
        invert=True,  # True inside the circle (i.e. the masked / scaled region)
    )
    extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask_extra_galaxies = ag.Mask2D(
    mask=extra_galaxies_mask,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
)

"""
__Galaxies json + Centres__

Save the true galaxies and both centre files.

Two separate files are written, and the split matters: `galaxy_centres.json` holds the co-equal pair (it
initializes their free centre priors, as in `multi_galaxy/modeling.py`), while `extra_galaxies_centres.json`
holds the companions (their centres are *fixed*, not merely initialized). The tier a galaxy belongs to is
decided by which file it lands in.

For your own data, the pair's centres come from the centre-input GUI referenced in `multi_galaxy/start_here.py`,
and the companions' from
`autogalaxy_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=dataset_path / "galaxies.json",
)

ag.output_to_json(
    obj=ag.Grid2DIrregular(galaxy_centres),
    file_path=dataset_path / "galaxy_centres.json",
)

ag.output_to_json(
    obj=ag.Grid2DIrregular(extra_galaxy_centres),
    file_path=dataset_path / "extra_galaxies_centres.json",
)
