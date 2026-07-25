"""
Cluster: Modeling
=================

This script models the simulated cluster field (1 BCG + 10 catalogue members) with the modeling API spelled
out, and shows the refinements a real cluster analysis reaches for once the `start_here.py` composition
converges:

 - **Freeing the member tier's shape** — the shared effective radius / Sersic index of the member
   population promoted from fixed values to shared free parameters.
 - **Promoting bright members** — giving the brightest catalogue members their own free light models
   alongside the BCG, exactly as the lensing workspace promotes members near multiple images.
 - **Truth comparison** — checking the recovered shared normalization against the simulation truth.

Everything runs through the standard `AnalysisImaging`; the cluster regime changes model composition, not
the analysis. (And note the divergence from the lensing workspace's cluster package: there the same
catalogue drives member MASSES and no galaxy light is modeled at all; here the light is the entire
subject.)

__Contents__

- **Dataset & Mask:** Standard set up (auto-simulating if absent).
- **Member Catalogue:** Load centres + luminosities via `ag.galaxy_table_from_csv`.
- **Model:** BCG MGE + catalogue tier with shared free normalization AND shared free shape.
- **Promoted Members:** The two brightest members given their own free models.
- **Search + Analysis:** Configure the search and analysis, and fit.
- **Truth Comparison:** Recovered `intensity_scale` vs the simulator truth of 1.0.

__Simulation__

Fits the dataset produced by `autogalaxy_workspace/*/cluster/simulator.py` (auto-simulated if absent).
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
dataset_path = Path("dataset", "cluster", dataset_name)

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

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(bcg_centres) + list(member_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

As `start_here.py`, plus the first refinement: the member tier's SHAPE parameters (effective radius,
Sersic index) are promoted from fixed values to parameters SHARED by the whole tier. Every member still
inherits them from the same two priors, so the member tier contributes 3 free parameters
(`intensity_scale` + shared `effective_radius` + shared `sersic_index`) regardless of its size — 8
members here after the promotion below, and still 3 parameters with 200.

__Promoted Members__

The second refinement: the two brightest catalogue members are promoted OUT of the tier and given their
own free `SersicSph` models — the light-side analogue of the lensing workspace freeing members that sit
near multiple images. Promotion costs their full per-galaxy parameters, so promote sparingly: brightest
first, and only while the data keeps constraining them.
"""
# BCG:

bulge = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=20,
    centre_prior_is_uniform=True,
    centre=(bcg_centres[0][0], bcg_centres[0][1]),
)

galaxy_dict = {"bcg": af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)}

# Promoted members: own free models. (The shipped CSV is sorted brightest-first, so rows 0-1 are
# the brightest; for your own catalogue, sort by luminosity first or select rows explicitly.)

n_promote = 2

for i in range(n_promote):
    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = tuple(member_centres[i])
    bulge.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=5.0)

    galaxy_dict[f"member_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

# Catalogue tier (remaining members): shared normalization + shared shape.

intensity_scale = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
tier_effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=2.0)
tier_sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=5.0)

for i in range(n_promote, len(member_centres)):
    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = tuple(member_centres[i])
    bulge.intensity = intensity_scale * float(member_luminosities[i])
    bulge.effective_radius = tier_effective_radius
    bulge.sersic_index = tier_sersic_index

    galaxy_dict[f"member_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

print(model.info)

"""
__Search + Analysis__
"""
search = af.Nautilus(
    path_prefix=Path("cluster"),
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
__Result + Truth Comparison__

The simulator built every member with `intensity = luminosity` — i.e. a true shared normalization of 1.0
(and true shape 0.6 / 3.0). A real (non-bypass) run recovers these; under PYAUTO_TEST_MODE the printed
values are prior medians and not meaningful.
"""
print(result.info)

instance = result.max_log_likelihood_instance

recovered_scale = (
    instance.galaxies.member_2.bulge.intensity / float(member_luminosities[2])
)
print(f"Recovered member-tier intensity_scale = {float(recovered_scale):.3f} (truth: 1.0)")

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.subplot_fit_imaging_of_galaxy(fit=result.max_log_likelihood_fit, galaxy_index=0)

"""
__Wrap Up__

- Scale up by appending rows to `scaling_galaxies.csv` — the tier's free-parameter count is unchanged.
- `autolens_workspace/*/cluster`: the lensing counterpart — the same catalogue schema driving member
  masses via the (Bergamini et al. 2019 tied-exponent) scaling relation, point-source source fitting, no
  galaxy light in the model.
"""
