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

The day-1 catalogue contains a single concrete latent:

 - ``total_galaxy_0_flux_mujy`` — total integrated flux of the first galaxy in the fit (``fit.galaxies[0]``),
   converted from the fit's linear flux units to microjanskies via the ``magzero`` keyword argument passed
   to ``ag.AnalysisImaging``. Returns NaN when galaxy 0 has no light profile.

The catalogue is intentionally narrow at launch — the long-tail of useful galaxy latents will accrue over
time as users contribute the ones their science needs. See the "Contributing Upstream" section at the end of
this tutorial for how to add new ones.

Future library releases may extend the registry. The toggle layer below means new entries default to ``false``
in the library yaml, so adding a latent never changes the behaviour of an existing fit — users opt in by
enabling the key in their workspace ``config/latent.yaml``.
"""

"""
__Toggling Latents__

The library defaults every latent to ``false`` in ``autogalaxy/config/latent.yaml``. This is deliberate: the
flux-derived latents require a ``magzero`` value via ``AnalysisImaging(..., magzero=...)`` kwargs, and enabling
them by default would crash any existing fit that doesn't pass ``magzero``. To opt in, you create or edit your
workspace's ``config/latent.yaml`` and set the keys you want to ``true``. This workspace ships such a file at
``autogalaxy_workspace/config/latent.yaml`` with ``total_galaxy_0_flux_mujy: true``.

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

We pass ``magzero=25.0`` to ``ag.AnalysisImaging``. This is required by ``total_galaxy_0_flux_mujy`` (the flux
needs a photometric zero-point to convert into microjanskies). If you forget it, the latent computation will
raise ``ValueError`` with a message pointing here.
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

Because ``total_galaxy_0_flux_mujy`` is the only enabled latent in this workspace, the returned instance only
exposes that single attribute. If you enable additional latents in the workspace yaml, they all appear as
attributes on the same instance.
"""
latent_samples = analysis.compute_latent_samples(result.samples)

median_instance = latent_samples.median_pdf()
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
you subclass ``ag.AnalysisImaging`` and override ``LATENT_KEYS`` + ``compute_latent_variables``.

The example below adds ``bulge_axis_ratio`` (the axis ratio of the Sersic bulge, computed from its
``ell_comps`` parameters). The same pattern works for any function of the model instance.

A subclass-level ``LATENT_KEYS`` shadows the library's config-driven ``@property``, so once you define your
own list the workspace yaml stops controlling it. If you want to keep the library latents AND add your own,
make ``LATENT_KEYS`` a property that returns ``super().LATENT_KEYS + ["your.key"]`` — the Euclid pipeline
(``euclid_strong_lens_modeling_pipeline/util.py``) shows this pattern in practice.
"""

import numpy as np


class AnalysisImagingWithAxisRatio(ag.AnalysisImaging):
    """
    AnalysisImaging extended with a custom ``bulge_axis_ratio`` latent — the axis ratio of the Sersic
    bulge derived from its ``ell_comps``. Demonstrates how to add a user-defined latent without modifying
    the library.
    """

    @property
    def LATENT_KEYS(self):
        return list(super().LATENT_KEYS) + ["bulge_axis_ratio"]

    def compute_latent_variables(self, parameters, model):
        from autogalaxy.imaging.model.latent import LATENT_FUNCTIONS

        xp = self._xp
        magzero = self.kwargs.get("magzero", None)
        instance = model.instance_from_vector(vector=parameters)
        fit = self.fit_from(instance=instance)
        context = {"fit": fit, "magzero": magzero, "xp": xp}

        library_keys = [k for k in super().LATENT_KEYS]
        library_values = tuple(LATENT_FUNCTIONS[k](**context) for k in library_keys)

        # Custom latent: axis ratio of the Sersic bulge from its ell_comps.
        try:
            ell_y, ell_x = instance.galaxies.galaxy.bulge.ell_comps
            axis_ratio = (1.0 - np.sqrt(ell_y**2 + ell_x**2)) / (
                1.0 + np.sqrt(ell_y**2 + ell_x**2)
            )
        except AttributeError:
            axis_ratio = xp.nan

        return library_values + (axis_ratio,)


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
