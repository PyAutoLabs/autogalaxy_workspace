"""
Modeling Features: Extra Galaxies (Interferometer)
====================================================

There may be extra galaxies nearby the main galaxy whose light blends with the main galaxy's emission.
For interferometer data — unlike strong-lensing imaging — we typically *want* to detect these extra
galaxies and fit their light, because autogalaxy is a morphology-fitting tool for multi-galaxy fields
rather than a gravitational-lensing tool.

This script shows how to perform galaxy modeling which includes the light profiles of the extra galaxies.
The centres of each extra galaxy (loaded from `extra_galaxies_centres.json`, produced by `simulator.py`)
are used as the centre of the light profiles of these galaxies, in order to reduce model complexity.

Multi-galaxy interferometer fits were previously impractical because every iteration has to NUFFT each
galaxy's light profile separately into the uv-plane, and prior NUFFT backends were not JAX-friendly.
With `nufftax` (https://github.com/GragasLab/nufftax) — a JAX-native NUFFT — the full set of light
profiles is transformed inside the same jit/vmap pipeline as the rest of the model, amortising the
per-iteration NUFFT cost on the GPU. Multi-galaxy fits to visibilities are now routine even at
ALMA-class visibility counts.

__Contents__

- **Data Preparation:** Data standards required for fitting with PyAutoGalaxy.
- **Mask:** Define the `real_space_mask` which sets the grid the galaxy field is evaluated on.
- **Dataset:** Load the `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Extra Galaxies Centres:** Load the JSON file of (y,x) centres for the extra galaxies.
- **Model:** Compose the model for the main galaxy (linear Sersic bulge + linear Exponential disk).
- **Extra Galaxies Model:** Compose the model for the extra galaxies (one linear `SersicSph` per extra,
  with fixed centre; an MGE alternative is described in the prose).
- **Search + Analysis:** Configure the non-linear search and `AnalysisInterferometer`.
- **VRAM:** Memory budget for multi-galaxy interferometer fits on GPU.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Approaches to Extra Galaxies:** Comparison with the imaging approach (noise scaling vs modeling).
- **Wrap Up:** Summary of the script and next steps.

__Data Preparation__

To perform modeling which accounts for extra galaxies, a list of the (y,x) centre of each extra galaxy
is used to set up the model-fit. For the example dataset used here, this metadata has already been
produced and saved by the companion `simulator.py` script as `extra_galaxies_centres.json` in the
dataset folder.

For real interferometer data where extras are too faint to detect, you typically need an auxiliary
imaging dataset to locate them. The autogalaxy data-preparation tutorials describe how to mark these
centres on a `.json` file from imaging data.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script, see
`autogalaxy_workspace/*/imaging/features/extra_galaxies/modeling.py`.

__Lensing Equivalent__

For the strong-lensing version (extras contribute mass, not light), see
`autolens_workspace/*/interferometer/features/extra_galaxies/modeling.py`.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Mask__

Define the `real_space_mask` which sets the grid the galaxy field is evaluated on. We use a larger
6.0" mask than the typical 3.5" interferometer mask because the extra galaxies lie at offsets of
~3.5" from the main galaxy's centre, and we want their emission included in the real-space grid.
"""
mask_radius = 6.0

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the `Interferometer` dataset `extra_galaxies` from .fits files, which we will fit with the
model.

We use `TransformerNUFFT`, the JAX-native NUFFT backed by `nufftax`, which is required for fast
multi-galaxy modeling and scales efficiently from a few hundred visibilities to tens of millions.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/features/extra_galaxies/simulator.py"],
        check=True,
    )

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerNUFFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Extra Galaxies Centres__

To set up a model including each extra galaxy with a light profile, we input manually the centres of
the extra galaxies.

In principle, a model including the extra galaxies could be composed without these centres. For example,
if there were two extra galaxies in the data, we could simply add two additional light profiles into the
model. The modeling API does support this, but we will not use it in this example.

This is because models where the extra galaxies have free centres are often too complex to fit. It is
likely the fit will infer an inaccurate model and local maxima, because the parameter space is too
complex. A common failure is that one extra-galaxy light profile recenters itself onto the main galaxy.

Therefore, when modeling extra galaxies we input the centre of each, in order to fix their light profile
centres.

For this example the centres are loaded from the `.json` file written by the companion `simulator.py`.
"""
extra_galaxies_centres = ag.Grid2DIrregular(
    ag.from_json(file_path=Path(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)

"""
__Model__

Perform the normal steps to set up the main model of the galaxy: a linear Sersic bulge + linear
Exponential disk, with the disk centre aligned to the bulge centre.

A full description of model composition is provided by the model cookbook:

  https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

"""
__Extra Galaxies Model__

We now use the modeling API to create the model for the extra galaxies. The extra galaxies API requires
that the centres of the light profiles are fixed to the input centres (other parameters remain free).

In this example each extra galaxy is fitted with a single linear `SersicSph` (spherical Sersic), with
its centre fixed to the value loaded from `extra_galaxies_centres.json`. This is the classic API and
suits a small handful of bright, roughly-symmetric companions.

For irregular or asymmetric companion morphologies, you can swap the per-extra `SersicSph` for an MGE
basis via `ag.model_util.mge_model_from(centre_fixed=...)` (commented-out block below). The MGE keeps the
same dimensionality cost in the linear-light limit (only the shared `ell_comps` are free), but is far
more flexible at capturing irregular morphology.

In this example the model is:

 - The main galaxy's bulge is a linear parametric `Sersic` [6 parameters].
 - The main galaxy's disk is a linear parametric `Exponential` [5 parameters].
 - Each extra galaxy's light is a linear parametric `SersicSph` with fixed centre
   [2 extra galaxies x 2 parameters = 4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
"""
# Extra Galaxies (Option A — default): one SersicSph per extra galaxy.
extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:
    extra_galaxy = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.SersicSph,
    )

    extra_galaxy.bulge.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

# Option B (uncomment to use): MGE bulges per extra galaxy.
# extra_galaxies_list = []
# for extra_galaxy_centre in extra_galaxies_centres:
#     bulge = ag.model_util.mge_model_from(
#         mask_radius=mask_radius,
#         total_gaussians=10,
#         centre_fixed=tuple(extra_galaxy_centre),
#     )
#     extra_galaxies_list.append(
#         af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)
#     )

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Model:
model = af.Collection(
    galaxies=af.Collection(galaxy=galaxy), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes extra galaxies that we defined above.
"""
print(model.info)

"""
__Search + Analysis__

The code below performs the normal steps to set up a model-fit.

Given the extra model parameters due to the extra galaxies, we use 150 live points.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=20,  # GPU model fits are batched and run simultaneously, see VRAM section below.
)

analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=True)

"""
__VRAM__

The `interferometer/modeling.py` example explains how VRAM is used during GPU-based fitting and how to
print the estimated VRAM required by a model.

Adding extra galaxies increases VRAM usage modestly, because each additional linear light profile adds
a column to the visibility-plane mapping matrix. For 2-5 extra galaxies this is negligible (each adds
~5-15 MB per batched likelihood). For dozens of extras you may need to monitor VRAM more carefully.

VRAM on interferometer datasets is driven primarily by the visibility count and the real-space mask
size, not the number of galaxies in the field.

__Run Time__

Adding extra galaxies to the model increases the likelihood evaluation time, because each galaxy's
light profile must be evaluated on the real-space grid and its image NUFFT'd to the uv-plane.

With `nufftax`, the per-NUFFT cost is small enough that adding 2 extras typically slows down the
per-likelihood by ~10-30% compared to the single-galaxy case — paid back in better fit quality because
the extras are no longer biasing the main galaxy fit.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitInterferometer` object we can confirm the extra galaxies
contribute to the fit.
"""
aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

"""
Checkout `autogalaxy_workspace/*/guides/results` for a full description of analysing results.

These examples show how the results API can be extended to investigate extra galaxies in the results.

__Approaches to Extra Galaxies__

For CCD imaging data, autogalaxy supports two approaches to extra galaxies:

- **Masking / noise scaling**: mask the extras' pixels (or scale their noise to large values), so they
  don't contribute to the fit. Suitable when you only care about the main galaxy.
- **Modeling**: include the extras in the model as additional light profiles, so they are fit and
  subtracted simultaneously. Suitable when you want their morphology or when their light blends with the
  main galaxy.

For interferometer data the masking approach is less straightforward — the data lives in the uv-plane,
not directly tied to image-plane pixels, so masking a region of the image-plane doesn't cleanly remove
that emission from the likelihood. The modeling approach (this script) is therefore the recommended
way to handle extras in interferometer data.

__Wrap Up__

The extra galaxies API makes it straight forward for us to model interferometer galaxy fields with
additional light components for nearby galaxies. Thanks to `nufftax`, the per-extra-galaxy NUFFT cost
is amortised on the GPU and multi-galaxy fits are now routine at any visibility count.
"""
