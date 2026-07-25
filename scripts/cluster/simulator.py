"""
Simulator: Cluster
==================

This script simulates `Imaging` of a 'cluster' field: a brightest cluster galaxy (BCG) surrounded by ten
lower-luminosity member galaxies. It is the top rung of PyAutoGalaxy's regime ladder — the point where
modeling every galaxy with its own free light model (the `multi_galaxy` approach) stops scaling, and the
member population is instead driven by a **catalogue**: a CSV of member centres and luminosities whose
photometry pins the faint galaxies while only shared normalizations stay free.

PyAutoGalaxy deals in neither mass models nor lensed sources, so this package is the photometric
counterpart of the lensing workspace's cluster package: the same population-scale bookkeeping (catalogue
CSVs, tiered composition), applied to the galaxies' LIGHT. Note the deliberate divergence between the two
libraries at this rung: **here the foreground galaxies' light IS the subject and is always modeled**,
whereas the PyAutoLens cluster workflow fits point-source image positions and does not model lens light.

This script simulates `Imaging` of a cluster field where:

 - The BCG's light is an elliptical `Sersic` (de Vaucouleurs-like).
 - Ten member galaxies have `SersicSph` light profiles whose intensities follow their catalogue
   luminosities.

__Contents__

- **Dataset Paths:** The dataset folder the simulated data is output to.
- **Grid / PSF / Simulator:** Standard imaging simulation setup.
- **BCG:** The brightest cluster galaxy.
- **Member Galaxies:** Ten members whose intensities follow the catalogue luminosities.
- **Dataset:** Simulate and write the imaging dataset.
- **Member Catalogue CSV:** Write `scaling_galaxies.csv` (y, x, luminosity) — the modeling scripts' input.
- **Galaxies json + Centres:** Truth records for the modeling scripts.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

import csv
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The dataset is output to `/autogalaxy_workspace/dataset/cluster/simple`.
"""
dataset_type = "cluster"
dataset_name = "simple"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid / PSF / Simulator__
"""
grid = ag.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

bcg_centre = (0.0, 0.0)

member_centres = [
    (5.5, -6.5),
    (-7.5, 3.0),
    (3.0, 8.0),
    (8.0, 5.0),
    (-6.5, -8.0),
    (-2.5, 6.5),
    (7.0, -2.0),
    (-8.0, 8.5),
    (2.0, -8.5),
    (-4.0, -3.5),
]

member_luminosities = [0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05]

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[bcg_centre] + member_centres,
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
__BCG__

The brightest cluster galaxy: a bright, extended de Vaucouleurs-like Sersic at the cluster centre. In the
modeling scripts it is the one galaxy modeled individually, with a free MGE.
"""
bcg = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=bcg_centre,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.5,
        effective_radius=2.5,
        sersic_index=4.0,
    ),
)

"""
__Member Galaxies__

Ten members whose central intensities equal their catalogue luminosities — so the rendered image visibly
traces the catalogue, and the modeling scripts' shared-normalization tier (intensity = scale * luminosity)
can recover the truth with `scale = 1`.
"""
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

"""
__Dataset__
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
__Member Catalogue CSV__

Write the member catalogue to `scaling_galaxies.csv` with the same three-column `y, x, luminosity` schema
the lensing workspace's cluster package uses (there it drives member MASSES via a scaling relation; here it
drives member LIGHT). The modeling scripts load it with `ag.galaxy_table_from_csv` — the catalogue-loading
API that makes the population a row-append away from scaling up.
"""
with open(dataset_path / "scaling_galaxies.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["y", "x", "luminosity"])
    for centre, luminosity in zip(member_centres, member_luminosities):
        writer.writerow([centre[0], centre[1], luminosity])

"""
__Galaxies json + Centres__
"""
ag.output_to_json(
    obj=galaxies,
    file_path=dataset_path / "galaxies.json",
)

ag.output_to_json(
    obj=ag.Grid2DIrregular([bcg_centre]),
    file_path=dataset_path / "bcg_centres.json",
)

"""
Finished.
"""
