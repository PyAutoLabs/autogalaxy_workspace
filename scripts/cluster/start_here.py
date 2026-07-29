"""
Start Here: Cluster
===================

Cluster fields contain a **brightest cluster galaxy (BCG) plus tens-to-hundreds of member galaxies**, far
too many to give each its own free light model. The cluster regime therefore changes how the model is
composed: the BCG (and any other dominant galaxies) are modeled individually, while the member population
is driven by a **catalogue** — a CSV of centres and luminosities whose photometry pins the faint members,
leaving only shared normalizations free. Adding a member is a row append; the model dimensionality does
not grow with the population.

This script gets you fitting a simulated cluster field (1 BCG + 10 members) in roughly 15 minutes, and
scales to real fields by swapping the CSV.

__The Regime Ladder (and a note for lensing users)__

PyAutoGalaxy organises galaxy-light modeling into a ladder of three regimes, mirroring the
`autolens_workspace` ladder: single galaxy (`imaging/`) → blended systems with one free model per galaxy
(`multi_galaxy/`) → catalogue-driven populations (this package).

If you come from lensing, note the one deliberate divergence between the two doc trees, and it is at this
rung: **in PyAutoGalaxy the cluster workflow models the foreground galaxies' light — that is its entire
subject.** In PyAutoLens the cluster workflow does NOT model lens light at all: it fits the point-source
multiple-image positions of the lensed background sources (lens-light modeling will arrive there later as
a feature). The population bookkeeping — catalogue CSVs, tiered composition, shared normalizations — is
the same machinery in both.

Why model a cluster's light? BCG + intracluster light assembly, member luminosity functions, photometry
uncontaminated by neighbours — and, for lensing applications, the light model that lensing analyses
subtract before fitting arcs.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** Run this example in a web browser without local installation.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load (auto-simulating if absent) and plot the cluster field.
- **Member Catalogue:** Load the member centres + luminosities from `scaling_galaxies.csv`.
- **Masking:** Mask the field.
- **Model:** BCG = free MGE; members = catalogue-pinned profiles with one shared free normalization.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Result:** Overview of the results, including BCG-only decomposition.
- **Model Your Own Cluster:** Adapting this script to your own imaging + catalogue.
- **Wrap Up:** Summary.

__JAX__

PyAutoGalaxy runs model-fits on JAX by default — `ag.AnalysisImaging` auto-enables `use_jax=True` if you
installed `autogalaxy[jax]`. The member tier adds fixed-shape profiles, which JIT-compile into the same
batched likelihood, so population size barely moves the fit cost.

__Google Colab Setup__

The `start_here` examples are runnable on Google Colab without local installation. The block below installs
the dependencies and downloads the example files if you're on Colab; running it locally is a no-op.
"""

try:
    import google.colab
except ImportError:
    from autogalaxy import setup_colab as _setup_colab
else:
    import importlib
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autonerves", "--no-deps"]
    )
    _setup_colab = importlib.import_module("autonerves.setup_colab")

_setup_colab.for_autogalaxy(
    raise_error_if_not_gpu=False  # Switch to True to require GPU on Colab.
)

"""
__Imports__
"""
from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the cluster dataset `simple` (1 BCG + 10 members). If the dataset is not found on disk it is simulated
automatically by `cluster/simulator.py`, so this script runs with no manual setup.
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Member Catalogue__

Load the member catalogue from `scaling_galaxies.csv` with `ag.galaxy_table_from_csv` — the
catalogue-loading API. The three-column `y, x, luminosity` schema is shared with the lensing workspace's
cluster package (where the catalogue drives member masses; here it drives member light). In a real
analysis this CSV comes straight from your photometry catalogue — it is spreadsheet-editable, and adding a
member is a row append that adds zero free parameters.
"""
scaling_table = ag.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

member_centres = scaling_table.centres.in_list
member_luminosities = scaling_table.luminosities

bcg_centres = ag.from_json(file_path=dataset_path / "bcg_centres.json")

"""
__Masking__

Mask the field generously — the members span the frame, and the model must account for every galaxy inside
the mask. We oversample the centre of every galaxy (BCG + members).
"""
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

The model has two tiers — the cluster regime's signature composition:

 - **BCG**: modeled individually with a free MGE, exactly as a single galaxy would be.
 - **Members**: one `SersicSph` per catalogue row with its centre fixed to the catalogue position, its
   shape fixed (effective radius, Sersic index), and its intensity TIED to the catalogue luminosity
   through a single shared free normalization:

       intensity_i = intensity_scale * luminosity_i

   The whole 10-member tier therefore contributes ONE free parameter, and would still contribute one with
   200 members. (Freeing per-member shapes, or promoting the brightest members to their own free models,
   are the natural refinements — see `cluster/modeling.py`.)
"""
# BCG:

bulge = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=20,
    centre_prior_is_uniform=True,
    centre=(bcg_centres[0][0], bcg_centres[0][1]),
)

galaxy_dict = {"bcg": af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)}

# Members: one shared free normalization for the whole tier.

intensity_scale = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

for i, (centre, luminosity) in enumerate(zip(member_centres, member_luminosities)):

    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = tuple(centre)
    bulge.intensity = intensity_scale * float(luminosity)  # tied to the catalogue
    bulge.effective_radius = 0.6
    bulge.sersic_index = 3.0

    galaxy_dict[f"member_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
Print the model — the BCG's MGE parameters are free, every member's intensity shows as tied to the single
shared `intensity_scale`, and the total dimensionality stays small despite 11 galaxies.
"""
print(model.info)

"""
__Model Fit__

Fit with `MultiStartProdigy`, a multi-start gradient optimizer, via `AnalysisImaging` — the same analysis as
every other rung of the ladder; only the composition changed.

__Multi Start Gradient Optimization__

`MultiStartProdigy` launches `n_starts` independent optimizations from broad starting points spread across the
parameter space, all of which descend the likelihood in parallel via `jax.vmap`, and returns the best one. A
single starting point would frequently get stuck in a local maximum, which is very likely for a cluster field
where many galaxies contribute light. Running a wide population of starts is what makes a gradient optimizer
reliable (the GIGA-Lens approach, Gu, Huang et al. 2022, arXiv:2202.07663). Prodigy is *learning-rate free*
(Mishchenko & Defazio 2024, arXiv:2306.06101), so there is nothing to tune.

__Posterior__

`MultiStartProdigy` is a maximum a posteriori (MAP) optimizer: it returns the **single best-fit galaxy model** and
nothing else — no posterior, no error bars, no covariances between parameters.

To get uncertainties, run `autogalaxy_workspace/scripts/cluster/modeling.py`, which fits this same model with the
nested sampling algorithm `Nautilus` and returns the **full posterior**. Use the fast optimizer here to check
your model and data are sensible, then `Nautilus` when you need results you can quote.
"""
search = af.MultiStartProdigy(
    path_prefix=Path("cluster"),
    name="start_here",
    unique_tag=dataset_name,
    n_starts=48,
    n_steps=300,
    iterations_per_quick_update=50,
    live_visual_update=False,
)

analysis = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

Plot the fit, and the BCG's decomposed light (galaxy index 0) — the member-subtracted BCG photometry that
motivates cluster light modeling.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.subplot_fit_imaging_of_galaxy(fit=result.max_log_likelihood_fit, galaxy_index=0)

"""
__Model Your Own Cluster__

- Supply your own CCD image, PSF, and RMS noise-map via `Imaging.from_fits()`.
- Replace `scaling_galaxies.csv` with your member catalogue (three columns: y, x, luminosity — any
  consistent luminosity units work, since only the shared normalization is fitted).
- Put the BCG centre(s) in `bcg_centres.json`; promote any other dominant galaxies to their own free
  MGEs alongside the BCG.
- Widen the mask to your field and check every catalogued member inside it is modeled.

__Wrap Up__

This script has shown the cluster regime's composition: individually-modeled BCG + a catalogue-driven
member tier with one shared free normalization.

Where to go next:

- `autogalaxy_workspace/*/cluster/modeling`: refinements — freeing member shapes, promoting bright
  members, comparing against the simulation truth.
- `autogalaxy_workspace/*/multi_galaxy`: the rung below — few blended galaxies, each fully free.
- `autolens_workspace/*/cluster`: the lensing counterpart — same catalogue machinery, applied to member
  masses, with the lensed sources fitted as point-source positions (and no lens light modeled).
"""
