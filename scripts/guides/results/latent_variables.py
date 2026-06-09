"""
Results: Latent Variables
=========================

A latent variable is a quantity derived from the model parameters that is not itself sampled by the non-linear
search — its posterior is induced by the posterior over the parameters. PyAutoGalaxy ships a curated catalogue
of galaxy-level latent variables out of the box, controlled by ``autogalaxy/config/latent.yaml``. This tutorial
shows what the catalogue contains, how to toggle individual latents on or off via the workspace config, how to
load latent results from a completed fit, and how to extend ``AnalysisImaging`` with a user-defined latent
when the curated set isn't enough.

The Bayesian framing of latent variables — what they are, why their errors are empirical posterior quantiles
rather than analytic Gaussian propagation, the trade-off between every-sample and N-draws-from-PDF output
modes — lives in the foundational autofit_workspace tutorial at
``../../../autofit_workspace/scripts/cookbooks/latent_variables.py``. Read that first if any of these terms
look unfamiliar; the tutorial here focuses on the galaxy-specific catalogue and the workspace ergonomics.

__Contents__

 - Galaxy Latents in PyAutoGalaxy: The curated default catalogue from ``autogalaxy/config/latent.yaml``.
 - Toggling Latents: The workspace ``config/latent.yaml`` override and why it shadows the library default.
 - Model Fit: A quick fit that produces real latent output for the loading section below.
 - Loading Latent Results: Reading ``latent/samples.csv`` / ``latent_summary.json`` via
   ``analysis.compute_latent_samples(result.samples)`` and the standard ``Samples`` API.
 - Extending with a Custom Latent: Subclass ``ag.AnalysisImaging``, override ``LATENT_KEYS`` and
   ``compute_latent_variables`` to add your own derived quantity.
 - Contributing Upstream: When your custom latent is general enough, consider promoting it to the library.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autogalaxy as ag

"""
__Galaxy Latents in PyAutoGalaxy__

The library ships a flat registry of named latent functions at ``autogalaxy.imaging.model.latent.LATENT_FUNCTIONS``,
backed by the toggle file ``autogalaxy/config/latent.yaml``. Each entry maps a snake-case latent name to a Python
function that takes a fit, magzero, and ``xp`` (numpy / jax.numpy) and returns a scalar value.

The catalogue contains two related latents that pair as a raw-flux / unit-converted couplet:

 - ``total_galaxy_0_flux`` — total integrated flux of the first galaxy in the fit (``fit.galaxies[0]``) in the
   fit's raw image units (typically e- s^-1 for HST, MJy/sr for JWST). Requires no instrument inputs and
   ships default-on in the library yaml. Returns NaN when galaxy 0 has no light profile. See
   ``scripts/guides/units/flux.py`` for how to convert this raw flux to microjanskies or AB magnitudes in post.

 - ``total_galaxy_0_flux_mujy`` — the same total flux converted to microjanskies via the ``magzero`` keyword
   argument passed to ``ag.AnalysisImaging``. Default-off in the library yaml — opt in only when you have a
   stable per-instrument zero-point. If you enable it without supplying ``magzero`` the column populates with
   NaN and the library emits a single warning per process (your fit completes uninterrupted).

The catalogue is intentionally narrow at launch — the long-tail of useful galaxy latents will accrue over
time as users contribute the ones their science needs. See the "Contributing Upstream" section at the end of
this tutorial for how to add new ones.

Future library releases may extend the registry. The toggle layer below means new entries can default either
way safely: a new raw-flux latent that requires no inputs can default-on, while anything needing instrument
metadata defaults-off so existing instrument-naive fits stay unchanged on upgrade.
"""

"""
__Toggling Latents__

The library defaults ``total_galaxy_0_flux: true`` (no instrument inputs needed) and
``total_galaxy_0_flux_mujy: false`` (you need to supply ``magzero``) in ``autogalaxy/config/latent.yaml``. The
default-off µJy variant is deliberate: it requires a ``magzero`` value via ``AnalysisImaging(..., magzero=...)``,
and the workspace fit you're running may not have one to hand. To opt in to the µJy column, edit your
workspace's ``config/latent.yaml`` and set ``total_galaxy_0_flux_mujy: true``. This workspace ships such a file
at ``autogalaxy_workspace/config/latent.yaml`` with both keys ``true``.

Workspace ``config/`` values shadow the library defaults — PyAutoFit's ``conf.instance`` searches the workspace
``config/`` directory first, then falls back to the library's bundled defaults. So toggling a latent in your
workspace yaml is enough to enable it without modifying the library install.

To disable a latent for a specific fit (e.g. a quick test where you don't want to incur the latent
computation cost), set its key to ``false`` in ``autogalaxy_workspace/config/latent.yaml``. The list of valid
keys lives in ``autogalaxy.imaging.model.latent.LATENT_FUNCTIONS``.
"""

"""
__Model Fit__

To make the loading and extending sections below concrete, we run a quick model fit on the standard simple-
imaging dataset that ships with the workspace. The model is a single galaxy with a Sersic bulge — keeping it
small so the example runs in a reasonable time.

We pass ``magzero=25.0`` to ``ag.AnalysisImaging`` so ``total_galaxy_0_flux_mujy`` populates with a real value
rather than NaN. If you forget it, ``total_galaxy_0_flux_mujy`` will be NaN and the library will log a single
warning per process noting the conversion was skipped — the fit itself is unaffected, and the raw
``total_galaxy_0_flux`` column populates normally.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

galaxy_model = af.Model(ag.Galaxy, redshift=0.5, bulge=af.Model(ag.lp.Sersic))
model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)

search = af.Nautilus(
    name="cookbook_latent_variables",
    n_live=50,
    n_like_max=300,
)

result = search.fit(model=model, analysis=analysis)

"""
__Loading Latent Results__

The fit above produces ``latent/samples.csv`` and ``latent/latent_summary.json`` under the search's output
directory. You can read them back into a ``Samples`` object via ``analysis.compute_latent_samples(result.samples)``.
The returned object exposes the same API as the parameter ``Samples`` — ``median_pdf``, ``max_log_likelihood``,
``values_at_sigma_1``, and so on — but reports on the induced latent posterior rather than the parameter posterior.

Because both ``total_galaxy_0_flux`` and ``total_galaxy_0_flux_mujy`` are enabled in this workspace, the
returned instance exposes both attributes. If you enable additional latents in the workspace yaml, they all
appear as attributes on the same instance.
"""
latent_samples = analysis.compute_latent_samples(result.samples)

median_instance = latent_samples.median_pdf()
print(f"Median PDF total_galaxy_0_flux: {median_instance.total_galaxy_0_flux}")
print(
    f"Median PDF total_galaxy_0_flux_mujy: {median_instance.total_galaxy_0_flux_mujy}"
)

max_likelihood_instance = latent_samples.max_log_likelihood()
print(
    f"Max log-likelihood total_galaxy_0_flux_mujy: {max_likelihood_instance.total_galaxy_0_flux_mujy}"
)

"""
The 1σ / 3σ intervals on these latents are *empirical quantiles of the induced posterior* — they are NOT
analytic Gaussian propagation of the parameter errors through the latent function. When the parameter
posterior is skewed (e.g. ``intensity`` near zero, or a banana-shaped degeneracy between Sersic ``sigma`` and
``intensity``), the latent interval inherits that asymmetry faithfully. See
``../../../autofit_workspace/scripts/cookbooks/latent_variables.py`` for the foundational treatment of why.
"""

"""
__Extending with a Custom Latent__

The library catalogue is intentionally small. If you want a different derived quantity — a Sersic effective
radius converted to kiloparsecs, a colour from two-band fits, a velocity dispersion from a virial estimate —
subclass ``ag.LatentGalaxy`` (override ``keys`` and ``variables``) and declare it on your analysis through the
``Latent`` class attribute. This is the same first-class, swappable mechanism you use to customise a
``Visualizer`` — the latent catalogue is a component of the analysis, not a pair of methods you monkey-patch.

The example below adds ``bulge_axis_ratio`` (the axis ratio of the Sersic bulge, computed from its
``ell_comps`` parameters). The same pattern works for any function of the model instance. Calling
``ag.LatentGalaxy.keys(analysis)`` / ``ag.LatentGalaxy.variables(...)`` from your overrides keeps the
config-driven library latents alongside your custom one (subclass the base ``ag.Latent`` instead if you want a
catalogue from scratch). The Euclid pipeline (``euclid_strong_lens_modeling_pipeline/util.py``, ``LatentEuclid``)
uses this exact composition pattern in production.
"""

import numpy as np


class LatentBulgeAxisRatio(ag.LatentGalaxy):
    """
    The library galaxy latents plus a custom ``bulge_axis_ratio`` — the axis
    ratio of the Sersic bulge derived from its ``ell_comps``. Demonstrates adding
    a user-defined latent without modifying the library: subclass
    ``ag.LatentGalaxy`` and compose its ``keys`` / ``variables`` static methods.
    """

    @staticmethod
    def keys(analysis):
        return list(ag.LatentGalaxy.keys(analysis)) + ["bulge_axis_ratio"]

    @staticmethod
    def variables(analysis, parameters, model):
        library_values = ag.LatentGalaxy.variables(analysis, parameters, model)

        instance = model.instance_from_vector(vector=parameters)
        xp = analysis._xp
        try:
            ell_y, ell_x = instance.galaxies.galaxy.bulge.ell_comps
            axis_ratio = (1.0 - np.sqrt(ell_y**2 + ell_x**2)) / (
                1.0 + np.sqrt(ell_y**2 + ell_x**2)
            )
        except AttributeError:
            axis_ratio = xp.nan

        return library_values + (axis_ratio,)


class AnalysisImagingWithAxisRatio(ag.AnalysisImaging):
    """
    ``AnalysisImaging`` that swaps in the custom ``LatentBulgeAxisRatio`` catalogue
    via the ``Latent`` class attribute — the same one-line mechanism used to
    declare a custom ``Visualizer`` or ``Result``. No library code is modified.
    """

    Latent = LatentBulgeAxisRatio


"""
With the subclass defined, running a fit that uses it produces a ``latent.csv`` with one extra column
(``bulge_axis_ratio``) on top of the library defaults. We don't actually run a second fit here — the pattern
above is the full recipe — but the workspace test suite (``autolens_workspace_test``) exercises identical
custom-latent subclasses end-to-end if you want a verified example.
"""

"""
__Contributing Upstream__

If your custom latent is general enough that other PyAutoGalaxy users would benefit from it (e.g. a Sersic
half-light radius in physical units, a galaxy SNR estimator, an aperture flux at a fixed radius), please
consider submitting it to the library. The flow is:

 1. Add the function to ``autogalaxy/imaging/model/latent.py``, following the signature
    ``(fit, magzero, xp=np) -> scalar``. NaN is the right fallback when the function can't apply
    (e.g. no light profile present).
 2. Register it in the module-level ``LATENT_FUNCTIONS`` dict.
 3. Add an entry to ``autogalaxy/config/latent.yaml`` defaulting it to ``false`` (the workspace yaml will
    opt users in).
 4. Add a unit test under ``test_autogalaxy/imaging/test_latent.py``.
 5. Open a PR.

Pipeline-specific latents that require non-standard kwargs (PSF-relative aperture fluxes, dataset-specific
photometric quantities) should stay in your pipeline's local Analysis subclass — see how
``euclid_strong_lens_modeling_pipeline/util.py`` keeps its FWHM aperture latents pipeline-local and
inherits the rest from the library.
"""
