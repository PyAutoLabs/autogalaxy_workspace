"""
Simulator: Multi Galaxy
=======================

This script simulates `Imaging` of a 'multi galaxy' system: two galaxies close enough on the sky that their
light blends together, so both must be modeled simultaneously. Neither galaxy is a minor contaminant to mask
or subtract — they are co-equal subjects of the fit, and every one of them gets its own free light model.

This is the defining feature of the multi-galaxy regime on the galaxy side, and it mirrors the regime ladder
of the lensing workspace (`autolens_workspace`): there, `multi_galaxy` lenses have 2+ co-dominant deflectors;
here, `multi_galaxy` systems have 2+ blended galaxies whose light is fitted together. PyAutoGalaxy deals in
neither mass models nor lensed sources, so the regime here is purely photometric: interacting pairs, close
projected pairs, and compact multiples — exactly the systems (like the merging pair SDSS J1011+0143) whose
blended light single-galaxy fitting cannot decompose.

This script simulates `Imaging` of a multi-galaxy system where:

 - The system is a close pair of galaxies (~1.5" separation) whose light distributions are `Sersic` profiles
   of comparable brightness.

__Contents__

- **Dataset Paths:** The dataset folder the simulated data is output to.
- **Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Galaxies:** The two blended galaxies.
- **Dataset:** Simulate and write the imaging dataset.
- **Galaxies json + Centres:** Truth records for the modeling scripts.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The dataset is output to `/autogalaxy_workspace/dataset/multi_galaxy/simple`.
"""
dataset_type = "multi_galaxy"
dataset_name = "simple"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid / PSF / Simulator__
"""
grid = ag.Grid2D.uniform(
    shape_native=(180, 180),
    pixel_scales=0.1,
)

galaxy_centres = [(0.0, -0.75), (0.0, 0.75)]

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=galaxy_centres,
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

The two blended galaxies: comparable brightness (the second is ~70% as bright as the first), overlapping
light distributions, different morphologies (one de Vaucouleurs-like bulge, one disky Sersic). The ~1.5"
separation is small enough that each galaxy's light contaminates the other's centre — the regime where
simultaneous fitting is mandatory and sequential fit-and-subtract loops struggle.
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

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

"""
__Dataset__

Simulate the imaging dataset from the two galaxies and write it to the dataset folder.
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
__Galaxies json + Centres__

Save the true galaxies and the galaxy centres. The centres initialize the per-galaxy centre priors in the
modeling scripts (for your own data, click them with the centre-input GUI referenced in `start_here.py`).
"""
ag.output_to_json(
    obj=galaxies,
    file_path=dataset_path / "galaxies.json",
)

ag.output_to_json(
    obj=ag.Grid2DIrregular(galaxy_centres),
    file_path=dataset_path / "galaxy_centres.json",
)

"""
Finished.
"""
