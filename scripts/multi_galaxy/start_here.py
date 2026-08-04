"""
Start Here: Multi Galaxy
========================

Multi-galaxy systems have **two or more galaxies whose light blends together on the sky**, so that all of
them must be modeled simultaneously — every galaxy is a co-equal subject of the fit, with its own free
light model.

This script shows you how to model a multi-galaxy system using **PyAutoGalaxy** with as little setup as
possible. In about 15 minutes you'll be able to point the code at your own FITS files and fit your first
blended pair.

__Which Regime Is My System?__

PyAutoGalaxy organises galaxy-light modeling into a ladder of three regimes, mirroring the lensing regime
ladder of `autolens_workspace`:

 - **Single galaxy** (`imaging/start_here.ipynb`): one galaxy dominates the image; any neighbours are
   contaminants to mask out.

 - **Multi galaxy** (this package): 2+ galaxies of comparable brightness whose light overlaps — interacting
   pairs, close projected pairs, compact multiples. Each gets its own free light model, fitted together.

 - **Cluster** (`cluster/start_here.ipynb`): a brightest cluster galaxy plus tens-to-hundreds of member
   galaxies loaded from a catalogue — the population is modeled collectively, with catalogue photometry
   pinning the faint members.

A note for lensing users coming from `autolens_workspace`: the two doc trees mirror each other, with one
deliberate divergence at the top rung. In **PyAutoGalaxy the cluster workflow models the foreground
galaxies' light — that is its entire subject.** In **PyAutoLens the cluster workflow does not model lens
light at all** (it fits point-source multiple-image positions; lens-light modeling will arrive later as a
feature). Keep that in mind when moving between the libraries.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** Run this example in a web browser without local installation.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load (auto-simulating if absent) and plot the imaging dataset.
- **Galaxy Centres:** Load the centres used to initialize each galaxy's model.
- **Masking:** Mask the region of the image the model is fitted to.
- **Model:** Compose the model — one free MGE light model per galaxy.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Result:** Overview of the results.
- **Model Your Own System:** Adapting this script to your own imaging data.
- **Wrap Up:** Summary and the ladder up to clusters.

__JAX__

PyAutoGalaxy runs model-fits on JAX by default — `ag.AnalysisImaging` auto-enables `use_jax=True` if you
installed `autogalaxy[jax]`. Two blended MGE models fit comfortably on a GPU in minutes.

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

Load the multi-galaxy dataset `simple`: imaging of a simulated close pair of blended galaxies. If the
dataset is not found on disk it is simulated automatically by `multi_galaxy/simulator.py`, so this script
runs with no manual setup.
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Load the centres of the galaxies from a `.json` file in the dataset folder; they initialize each galaxy's
centre priors. For your own data, the centre-input GUI in the lensing workspace
(`autolens_workspace/*/group/start_here.ipynb`) writes this file from mouse clicks — the same file format
is used here.
"""
galaxy_centres = ag.from_json(file_path=dataset_path / "galaxy_centres.json")

"""
__Masking__

Define a circular mask enclosing BOTH galaxies' light — the fit decomposes the blend, so the mask must
cover the full system, not one galaxy.

We also oversample the central pixels of each galaxy, which improves modeling accuracy without adding
unnecessary cost elsewhere.
"""
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

Each galaxy's light is a Multi Gaussian Expansion (MGE) — flexible enough to capture two different
morphologies without many free parameters, which matters here because the two galaxies' parameters are
partially degenerate where their light overlaps.

__List-Based Model Composition__

Each galaxy is created in a loop over the galaxy centres and stored as `galaxy_0`, `galaxy_1`, etc. — the
same list-based API the lensing workspace's multi_galaxy and group packages use, so nothing needs
re-learning when you move between the libraries. The composition scales to any number of blended galaxies.
"""
galaxy_dict = {}

for i, centre in enumerate(galaxy_centres):

    bulge = ag.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    galaxy_dict[f"galaxy_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
Print the model — note `galaxy_0` and `galaxy_1` each carry their own free MGE, the signature of the
multi-galaxy regime.
"""
print(model.info)

"""
__Model Fit__

Fit the data using `MultiStartProdigy`, a multi-start gradient optimizer, via an `AnalysisImaging` object — the
same analysis used for single galaxies; only the model composition changed.

__Multi Start Gradient Optimization__

`MultiStartProdigy` launches `n_starts` independent optimizations from broad starting points spread across the
parameter space, all of which descend the likelihood in parallel via `jax.vmap`, and returns the best one. A
single starting point would frequently get stuck in a local maximum, which is especially likely here because
several co-dominant galaxies give a highly multi-modal parameter space. Running a wide population of starts is
what makes a gradient optimizer reliable (the GIGA-Lens approach, Gu, Huang et al. 2022, arXiv:2202.07663).
Prodigy is *learning-rate free* (Mishchenko & Defazio 2024, arXiv:2306.06101), so there is nothing to tune.

__Posterior__

`MultiStartProdigy` is a maximum a posteriori (MAP) optimizer: it returns the **single best-fit galaxy model** and
nothing else — no posterior, no error bars, no covariances between parameters.

To get uncertainties, run `autogalaxy_workspace/scripts/multi_galaxy/modeling.py`, which fits this same model
with the nested sampling algorithm `Nautilus` and returns the **full posterior**. Use the fast optimizer here to
check your model and data are sensible, then `Nautilus` when you need results you can quote.
"""
search = af.MultiStartProdigy(
    path_prefix=Path("multi_galaxy"),
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

Print the result info and plot the maximum likelihood fit. `subplot_fit_imaging_of_galaxy` shows each
galaxy's decomposed light separately — the deliverable of a blended-pair fit.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

for i in range(len(galaxy_centres)):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=result.max_log_likelihood_fit, galaxy_index=i
    )

"""
__Model Your Own System__

Adapt the code above by inputting the paths to your own .fits files into `Imaging.from_fits()`:

- Supply your own CCD image, PSF, and RMS noise-map.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to enclose every blended galaxy.
- Provide the galaxy centres in a `galaxy_centres.json` file.
- Start with the default model — one MGE per galaxy works very well for pretty much all blended systems!

__Wrap Up__

This script has shown how to model a multi-galaxy system: the standard imaging workflow, with one free
light model per blended galaxy.

Where to go next:

- `autogalaxy_workspace/*/multi_galaxy/modeling`: the full modeling API and how to customize the fit.
- `autogalaxy_workspace/*/multi_galaxy/simulator`: how the example dataset was simulated.
- `autogalaxy_workspace/*/cluster`: the top rung of the ladder — a BCG plus a member population loaded
  from a CSV catalogue.
- `autogalaxy_workspace/*/imaging/features`: linear light profiles, MGE variations, sky subtraction —
  all apply per-galaxy here unchanged.
"""
