"""
Multi Galaxy: Modeling
======================

This script models an example 'multi galaxy' system — two galaxies whose light blends together, each with
its own free light model — with the full modeling API spelled out.

The multi-galaxy regime keeps the **standard imaging analysis workflow**: the data is CCD imaging, the fit
is `AnalysisImaging`, and everything from `imaging/modeling.py` (priors, searches, linear light profiles,
result inspection) applies. What changes is only the model composition: **one free light model per blended
galaxy**, composed with the list-based `galaxy_0`, `galaxy_1`, ... API.

Two notes on the wider regime ladder:

 - Above this regime, the `cluster/` package models a brightest cluster galaxy plus tens-to-hundreds of
   members loaded from a CSV catalogue — the point where per-galaxy free models stop scaling and catalogue
   photometry takes over.
 - The lensing workspace (`autolens_workspace`) mirrors this package name for its multi-deflector regime;
   the list-based composition API is identical there, applied to light AND mass.

__Contents__

- **Dataset & Mask:** Standard set up of the dataset (auto-simulating if absent).
- **Model:** One MGE per galaxy via the list-based API; why MGEs suit blended fits.
- **Linear Light Profiles:** The MGE uses linear light profiles, solving intensities exactly.
- **Search + Analysis:** Configure the non-linear search and the `AnalysisImaging`.
- **Result:** Per-galaxy decomposition of the maximum likelihood fit.
- **Photometry:** Measuring each galaxy's flux from the decomposed fit.

__Simulation__

This script fits the simulated dataset produced by `autogalaxy_workspace/*/multi_galaxy/simulator.py`; if
the dataset is not found on disk it is simulated automatically before the fit.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset & Mask__
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

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

galaxy_centres = ag.from_json(file_path=dataset_path / "galaxy_centres.json")

mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(galaxy_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

One MGE light model per galaxy, composed in a loop over the centres. Where the two galaxies' light
overlaps their parameters are partially degenerate; two things keep the fit well-behaved:

 - The MGE's Gaussians share each galaxy's centre and ellipticity structure, so each galaxy contributes
   few non-linear parameters despite its flexibility.
 - Each galaxy's centre prior is initialized on its observed centre (from `galaxy_centres.json`), which
   breaks the labeling degeneracy between the two components.

__Linear Light Profiles__

The MGE's Gaussians are linear light profiles (`lp_linear`): their `intensity` values are solved for
exactly by linear inversion at every likelihood evaluation, rather than sampled as free parameters. For
blended systems this is a major win — the flux ratio between the two galaxies, the quantity most degenerate
in a blend, is solved exactly instead of explored stochastically.
"""
galaxy_dict = {}

for i, centre in enumerate(galaxy_centres):

    bulge = ag.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    galaxy_dict[f"galaxy_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

print(model.info)

"""
__Search + Analysis__

This example uses `Nautilus` because it returns the **full posterior** — every parameter's errors and the
covariances between them. The folder's `start_here.py` instead fits with `af.MultiStartProdigy`, a multi-start
gradient optimizer which is far faster but returns only a single best-fit model with no errors at all. Use that
one to check a model quickly, and this one when you need results you can quote.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy"),
    name="modeling",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=1000,
)

analysis = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The fit's `subplot_fit_imaging_of_galaxy` decomposes the blend, showing each galaxy's modeled light (and
the data with the OTHER galaxies subtracted) — the core deliverable of a multi-galaxy fit.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

for i in range(len(galaxy_centres)):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=result.max_log_likelihood_fit, galaxy_index=i
    )

"""
__Photometry__

With the blend decomposed, per-galaxy photometry is direct: each galaxy's model image contains only its own
light, so summing it gives that galaxy's flux uncontaminated by its neighbour.
"""
galaxies = result.max_log_likelihood_galaxies

for i, galaxy in enumerate(galaxies):
    image = galaxy.image_2d_from(grid=dataset.grids.lp)
    print(f"galaxy_{i}: total model flux = {float(image.array.sum()):.3f}")

"""
__Wrap Up__

- `autogalaxy_workspace/*/imaging/features`: linear light profiles, MGE variations, sky handling — all
  apply per-galaxy here unchanged.
- `autogalaxy_workspace/*/cluster`: the top rung — a BCG plus a catalogue-loaded member population.
"""
