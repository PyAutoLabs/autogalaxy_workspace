"""
Light Profiles
==============

This guide is the single-page tour of every light profile **PyAutoGalaxy** ships with: how to
construct each one, how to evaluate its image on a grid, how to compose it into a model, and how
to retrieve an instance from that model.  Once you have read this guide you should be able to
recognise every profile referenced by the modelling examples and the API reference, and you
should know which family (standard / linear / operated / multipole / basis) any given profile
belongs to.

The guide is deliberately broad rather than deep — for each family it shows the *shape* of the
API and points you at the relevant `features/` package for the workflow details.

__Contents__

- **Overview & Docs URL:** Where the canonical API reference lives.
- **All Light Profiles (Survey):** A high-level run-through of every profile in `ag.lp.*` and
  the related namespaces, without yet evaluating any images.
- **Detailed Example: Sersic Image:** Build a `Grid2D`, instantiate `ag.lp.Sersic`, evaluate
  `image_2d_from`, plot it.
- **Linear Light Profiles:** One-line API for `ag.lp_linear.*` — intensity solved by inversion.
- **Operated Light Profiles:** One-line API for `ag.lp_operated.*` — emission post PSF.
- **Basis:** The grouping object that lets many profiles behave as a single composite — the
  building block of Multi-Gaussian Expansion (MGE) and shapelet decompositions.
- **Light Profile in a Model:** Wrap a profile in `af.Model`, compose it into an `af.Collection`
  via a `Galaxy`, inspect the model info.
- **Model Instance from Light Profile:** Realise an instance from the model's prior medians and
  evaluate `image_2d_from` on it.
- **Multipole Light Profiles:** The newer `SersicMultipole` and `GaussianMultipole`, with the
  m=3 / m=4 Fourier perturbation on the eccentric radius explained and plotted.
- **Remaining Profiles Walkthrough:** Compact `image_2d_from` block for every standard profile
  not yet shown, emphasising the API is the same as the Sersic example above.

__Units__

In this guide, all quantities use **PyAutoGalaxy**'s internal unit coordinates: spatial
coordinates in arc-seconds, luminosities in electrons per second, and mass quantities (e.g.
convergence) are dimensionless.

The `guides/units_and_cosmology.ipynb` guide illustrates how to convert these to physical
quantities (kiloparsecs, magnitudes, solar masses).

__Data Structures__

Images returned by `image_2d_from` are wrapped in **PyAutoGalaxy**'s `Array2D` data structure
with `slim` and `native` views.  The `guides/data_structures.py` guide covers this in detail;
here we only use the default `slim` 1D representation when printing values.

__Docs URL__

The published API reference for these classes lives at:

    https://pyautogalaxy.readthedocs.io/en/latest/api/light.html

The autosummary on that page is the authoritative list of every public light-profile class.
This guide mirrors it section-by-section, so a class shown here as `ag.lp.SersicCore` is
documented there under the `Standard [ag.lp]` autosummary, and so on for `ag.lp_linear`,
`ag.lp_operated`, `ag.lp_basis`.
"""

# from autoconf import setup_notebook; setup_notebook()

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


"""
__Grid__

To evaluate the image of any light profile we need a 2D Cartesian grid of (y,x) coordinates.
We build a 100x100 grid here at a 0.05" pixel scale — used by every section below.
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,
)

"""
__All Light Profiles (Survey)__

**PyAutoGalaxy** groups light profiles into five namespaces, each with a clear purpose:

- `ag.lp.*` — *Standard* parametric profiles.  `intensity` is a free model parameter.
- `ag.lp_linear.*` — *Linear* profiles.  `intensity` is removed from the model and instead
  solved analytically via a linear matrix inversion during each likelihood evaluation.
- `ag.lp_operated.*` — *Operated* profiles representing emission that has already had an
  instrument operation (e.g. PSF convolution) applied to it; `operated_only` on the fit
  classes controls inclusion.
- `ag.lp_basis.Basis` — A grouping object that bundles multiple light profiles into a single
  composite profile (e.g. an MGE built from many Gaussians).
- `ag.lp_snr.*` — Standard profiles parameterised by *signal-to-noise ratio* rather than
  intensity; useful when simulating a dataset with a target SNR.  Not covered further in this
  guide, but shares the API of the Standard profiles.

Below we construct each standard profile with default parameters.  No `image_2d_from` is
evaluated yet — that comes in the next section.  The goal here is purely a catalogue of what
is available, in the same order they appear in the API reference.
"""
# Sersic family
sersic = ag.lp.Sersic()
sersic_sph = ag.lp.SersicSph()
sersic_core = ag.lp.SersicCore()
sersic_core_sph = ag.lp.SersicCoreSph()
sersic_multipole = ag.lp.SersicMultipole()

# Exponential family (Sersic with sersic_index fixed to 1)
exponential = ag.lp.Exponential()
exponential_sph = ag.lp.ExponentialSph()
exponential_core = ag.lp.ExponentialCore()
exponential_core_sph = ag.lp.ExponentialCoreSph()

# de Vaucouleurs (Sersic with sersic_index fixed to 4)
dev_vaucouleurs = ag.lp.DevVaucouleurs()
dev_vaucouleurs_sph = ag.lp.DevVaucouleursSph()

# Gaussian / Moffat / Multipole-Gaussian
gaussian = ag.lp.Gaussian()
gaussian_sph = ag.lp.GaussianSph()
gaussian_multipole = ag.lp.GaussianMultipole()
moffat = ag.lp.Moffat()
moffat_sph = ag.lp.MoffatSph()

# Specialised: Chameleon (NFW-like double-isothermal) and Elson-Free-Fall (King-like)
chameleon = ag.lp.Chameleon()
chameleon_sph = ag.lp.ChameleonSph()
eff = ag.lp.ElsonFreeFall()
eff_sph = ag.lp.ElsonFreeFallSph()

# Shapelets — n_y, n_x (Cartesian) or n, m (Polar) pick the basis index
shapelet_cartesian = ag.lp.ShapeletCartesian(n_y=0, n_x=0)
shapelet_polar = ag.lp.ShapeletPolar(n=0, m=0)
shapelet_exponential = ag.lp.ShapeletExponential(n=0, m=0)

# Basis — bundles a list of light profiles into a single composite
basis = ag.lp_basis.Basis(profile_list=[ag.lp_linear.Gaussian(sigma=0.5)])

"""
Two things worth knowing about this list before we move on:

1. Every elliptical profile (e.g. `Sersic`, `Gaussian`) has a spherical sibling whose name
   ends in `Sph` (e.g. `SersicSph`, `GaussianSph`).  The spherical variant fixes the
   ellipticity components `ell_comps` to `(0, 0)`, which is useful when you want to model a
   round galaxy and avoid two redundant parameters in the non-linear search.
2. The `Multipole` variants (`SersicMultipole`, `GaussianMultipole`) only exist as
   *elliptical* profiles — the m=3 / m=4 perturbations are angular distortions and are not
   meaningful without an underlying elliptical reference frame.

We now move on to seeing what these profiles actually produce when evaluated on a grid.

__Detailed Example: Sersic Image__

The `Sersic` profile is the canonical galaxy light profile, controlled by:

- `centre` — the (y, x) arc-second coordinate of the profile's centre.
- `ell_comps` — the two ellipticity components `(e1, e2)`.  Use
  `ag.convert.ell_comps_from(axis_ratio=..., angle=...)` to convert from human-friendly
  axis ratio and position angle.
- `intensity` — overall brightness normalisation.
- `effective_radius` — the half-light radius (arc-seconds).
- `sersic_index` — the Sersic concentration.  `n=1` reduces to an exponential disc and
  `n=4` reduces to a de Vaucouleurs profile.

Build a Sersic and evaluate its image on our grid:
"""
sersic = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

image = sersic.image_2d_from(grid=grid)

aplt.plot_array(array=image, title="Sersic Image")

"""
The returned `image` is an `Array2D` — the `slim` view is a 1D numpy array of length
`total_pixels`, and `native` gives a 2D `(shape_native_y, shape_native_x)` array.

This same `image_2d_from(grid=grid)` call exists on every light profile in this guide,
returning an image of identical shape and units.  Every section below is a small variation
on this one — the API is uniform.

__Linear Light Profiles__

For a non-linear search, the `intensity` parameter of a standard light profile is a free
parameter sampled by the fitter.  This works fine for one or two profiles, but adds a free
dimension to the parameter space for every extra profile you bolt on.

Linear light profiles solve this by removing `intensity` from the model entirely and instead
recovering it analytically via a linear matrix inversion at each likelihood evaluation.  This
keeps the non-linear parameter space small even when you combine many profiles, and is the
default in our modern modelling examples.

The API is identical to the standard profile, just without `intensity`:
"""
linear_sersic = ag.lp_linear.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    effective_radius=0.6,
    sersic_index=3.0,
)

"""
Every standard profile in `ag.lp.*` has a `ag.lp_linear.*` counterpart, **including the
newer `SersicMultipole` and `GaussianMultipole`** — `ag.lp_linear.SersicMultipole` and
`ag.lp_linear.GaussianMultipole` both exist and behave the same way (the multipole comps
are non-linear parameters; only the overall intensity is solved by inversion).

The full workflow (likelihood function, fits, modeling) is documented in:

    scripts/imaging/features/linear_light_profiles/

That folder contains `fit.py`, `modeling.py`, and `likelihood_function.py` showing how to
build models with linear profiles and what the likelihood looks like under the hood.

__Operated Light Profiles__

Some emission components — chiefly the unresolved bright cores of AGN — are already PSF-
convolved by the time you receive the image.  Standard profiles get PSF-convolved during the
fit, so applying a PSF a second time double-blurs them.

Operated light profiles tell the fit "this profile's emission has already had the PSF
operation applied; do not blur it again".  The fit classes expose an `operated_only` flag
that controls whether these profiles are included or excluded from a given image computation.
"""
operated_gaussian = ag.lp_operated.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.3,
    sigma=0.05,
)

"""
Three operated profiles are available: `ag.lp_operated.Gaussian`, `ag.lp_operated.Moffat`,
and `ag.lp_operated.Sersic`.

The full workflow (simulating with operated profiles, modeling with them) is documented in:

    scripts/imaging/features/operated_light_profile/

That folder contains `simulator.py` and `modeling.py`.

__Basis__

A `Basis` is not a profile in its own right but a *grouping* of profiles that behave as a
single composite.  The classic application is the Multi-Gaussian Expansion (MGE), where a
galaxy's light is decomposed into a sum of many concentric Gaussians at fixed centres and
ellipticities but with increasing widths — together they reproduce arbitrary radial profiles
the standard parametric forms cannot capture.

The `Basis` constructor takes a `profile_list` of any light or mass profiles.  Below we
build a four-Gaussian MGE with shared centre and ellipticity, sigmas that span an order of
magnitude, and explicit decreasing `intensity` so the innermost Gaussian dominates the
core and the wider ones add the outer envelope.  Standard `ag.lp.Gaussian` profiles are
used here so the demo image is meaningful; in an actual fit you would swap these for
`ag.lp_linear.Gaussian` (see the note after the plot).
"""
basis = ag.lp_basis.Basis(
    profile_list=[
        ag.lp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            intensity=1.0,
            sigma=0.05,
        ),
        ag.lp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            intensity=0.5,
            sigma=0.15,
        ),
        ag.lp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            intensity=0.25,
            sigma=0.4,
        ),
        ag.lp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            intensity=0.1,
            sigma=1.0,
        ),
    ]
)

aplt.plot_array(
    array=basis.image_2d_from(grid=grid),
    title="Basis Image (4-Gaussian MGE)",
)

"""
Two things make `Basis` powerful:

- It slots into a `Galaxy` exactly like a `Sersic` would — once wrapped, the rest of the
  modelling code doesn't have to know it's looking at four Gaussians under the hood.
- When the constituents are `LightProfileLinear` instances (e.g. `ag.lp_linear.Gaussian`)
  rather than the standard `ag.lp.Gaussian` used in the demo above, all of their
  `intensity` values are solved together in a **single combined inversion** at each
  likelihood evaluation.  This means an MGE built from, say, 30 Gaussians adds only the
  shared geometric parameters to the non-linear search rather than 30 extra intensities.
  The demo above uses standard Gaussians purely so the image is non-zero on a static
  plot — in a real fit you would build the basis from `ag.lp_linear.Gaussian` and let the
  inversion solve the intensities.

The full MGE workflow — choosing how many Gaussians to use, how to space their `sigma`
values, and how the inversion plays with regularisation — is documented in:

    scripts/imaging/features/multi_gaussian_expansion/

Shapelet decompositions follow the same `Basis` pattern, using `ag.lp.ShapeletPolar` /
`ag.lp.ShapeletCartesian` / `ag.lp.ShapeletExponential` (and their linear counterparts).
The full shapelets workflow is documented in:

    scripts/imaging/features/shapelets/

__Light Profile in a Model__

So far we have been instantiating profiles with concrete parameter values.  When fitting a
real dataset we instead build a *model* of the profile and let the non-linear search find the
best-fit parameters.  This is what `af.Model` is for.
"""
sersic_model = af.Model(ag.lp.Sersic)

"""
The `af.Model` wraps the profile class.  Every constructor argument that has a numerical
default now becomes a *prior* — by default the priors are `UniformPriors` covering a sensible
range for each parameter (see `autogalaxy/config/priors/light.yaml` for the configured
ranges).

You can override individual priors before fitting:
"""
sersic_model.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
sersic_model.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=10.0)

"""
A model profile by itself is not yet a complete model — it has to be associated with a
`Galaxy` and an `af.Collection` so the search knows what dataset it's fitting:
"""
galaxy_model = af.Model(ag.Galaxy, redshift=0.5, bulge=sersic_model)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))

print(model.info)

"""
Printing `model.info` prints the full priors-and-defaults summary — useful before kicking off
a long fit to confirm the model looks the way you expect.

The model API is the same for **every** light profile in this guide — swap `ag.lp.Sersic`
for `ag.lp.SersicMultipole`, `ag.lp_linear.Gaussian`, `ag.lp_basis.Basis`, etc., and the rest
of the snippet is unchanged.  Multipole comps and Basis constituent lists are wired into the
prior machinery automatically.

Full modeling end-to-end examples live in `scripts/imaging/modeling.py` and the topic-
specific guides under `scripts/imaging/features/`.

__Model Instance from Light Profile__

A model is a description of *possible* profiles.  To get an actual profile back out — for
example to plot what the prior medians look like before running a fit — call
`instance_from_prior_medians()`:
"""
sersic_instance = sersic_model.instance_from_prior_medians()
print(type(sersic_instance))  # autogalaxy.profiles.light.standard.sersic.Sersic

image = sersic_instance.image_2d_from(grid=grid)
aplt.plot_array(array=image, title="Sersic Instance from Prior Medians")

"""
The instance returned from `instance_from_prior_medians()` is a real `ag.lp.Sersic` — the
same class we constructed by hand at the top of the guide — and supports the full API
including `image_2d_from`.

The same flow works at the galaxies level: realise an instance of the full model and pull
the light profile back out of it.
"""
model_instance = model.instance_from_prior_medians()

galaxies = ag.Galaxies(galaxies=[model_instance.galaxies.galaxy])

aplt.plot_array(
    array=galaxies.image_2d_from(grid=grid),
    title="Galaxies Instance Image",
)

bulge_instance = model_instance.galaxies.galaxy.bulge
print(type(bulge_instance))  # autogalaxy.profiles.light.standard.sersic.Sersic

"""
After a fit completes, `result.max_log_likelihood_instance` returns the same shape of
object, with the prior medians replaced by the fitted parameter values.  See
`scripts/guides/results/start_here.py` for the full results-introspection guide.

__Multipole Light Profiles__

`SersicMultipole` and `GaussianMultipole` are recent additions that bolt m=3 and m=4 Fourier
angular perturbations onto the eccentric radius of a base profile.  The perturbed radius is

    r' = r * (1 + c3 cos(3 theta) + s3 sin(3 theta)
                + c4 cos(4 theta) + s4 sin(4 theta))

where `theta` is the polar angle in the profile's elliptical reference frame, and the
`multipole_3_comps = (c3, s3)` and `multipole_4_comps = (c4, s4)` parameters control the
amplitude of each perturbation.

When both `multipole_*_comps` are `(0.0, 0.0)` (the defaults), the profile reduces exactly to
the base profile.  This is by design — you can swap a `Sersic` for a `SersicMultipole` in any
model without changing its predictions, and the multipole comps simply add four extra free
parameters that can capture boxy / discy / lopsided morphologies.  Plugging one into the
`af.Model` / `af.Collection` / `Galaxy` pattern shown above works exactly as it did for the
plain `Sersic` — the multipole comps are picked up as priors automatically.

Build a `SersicMultipole` with non-zero multipole components, alongside the unperturbed
Sersic that produced our reference image earlier in the guide:
"""
sersic_multipole = ag.lp.SersicMultipole(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
    multipole_3_comps=(0.05, 0.00),
    multipole_4_comps=(0.00, 0.04),
)

aplt.plot_array(
    array=sersic_multipole.image_2d_from(grid=grid),
    title="SersicMultipole Image (m=3 + m=4 perturbation)",
)

"""
For comparison, here is the unperturbed Sersic image — the two should look almost
identical with the perturbation showing as a subtle azimuthal modulation:
"""
aplt.plot_array(
    array=sersic.image_2d_from(grid=grid),
    title="Sersic Image (no multipole perturbation)",
)

"""
The `GaussianMultipole` profile applies the same perturbation to a Gaussian base — useful
when you want a multipole component inside a Multi-Gaussian Expansion (the `Basis` section
above shows how to bundle Gaussians together):
"""
gaussian_multipole = ag.lp.GaussianMultipole(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    sigma=0.4,
    multipole_3_comps=(0.05, 0.00),
    multipole_4_comps=(0.00, 0.04),
)

aplt.plot_array(
    array=gaussian_multipole.image_2d_from(grid=grid),
    title="GaussianMultipole Image",
)

"""
Two practical notes on the multipole variants:

- There is **no spherical (`*Sph`) variant** of either multipole.  The perturbation is an
  angular distortion measured in the elliptical reference frame, so it only makes sense for
  an elliptical profile (a spherical profile has no preferred angle).
- Both multipoles exist as **linear variants** too: `ag.lp_linear.SersicMultipole` and
  `ag.lp_linear.GaussianMultipole`.  In the linear form the multipole comps remain non-
  linear parameters but the overall intensity is solved by inversion, just like for the
  ordinary linear profiles.

__Remaining Profiles Walkthrough__

We have shown the full `image_2d_from` → `af.Model` → `instance` flow for the `Sersic`
profile.  Every remaining standard profile uses the **same API** — the only thing that
changes is which parameters appear in the constructor.

The compact tour below builds each remaining profile with sensible parameter values and
plots its image, so you can see what each looks like.  When you want to use any of these in
a model, repeat the `af.Model(...)` / `af.Collection(...)` pattern from the previous section.
"""

aplt.plot_array(
    array=ag.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
        radius_break=0.05,
        gamma=0.2,
        alpha=3.0,
    ).image_2d_from(grid=grid),
    title="SersicCore Image",
)

aplt.plot_array(
    array=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
    ).image_2d_from(grid=grid),
    title="Exponential Image",
)

aplt.plot_array(
    array=ag.lp.ExponentialCore(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
        radius_break=0.05,
        gamma=0.2,
        alpha=3.0,
    ).image_2d_from(grid=grid),
    title="ExponentialCore Image",
)

aplt.plot_array(
    array=ag.lp.DevVaucouleurs(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
    ).image_2d_from(grid=grid),
    title="DevVaucouleurs Image",
)

aplt.plot_array(
    array=ag.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        sigma=0.4,
    ).image_2d_from(grid=grid),
    title="Gaussian Image",
)

aplt.plot_array(
    array=ag.lp.Moffat(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        alpha=0.4,
        beta=2.5,
    ).image_2d_from(grid=grid),
    title="Moffat Image",
)

aplt.plot_array(
    array=ag.lp.Chameleon(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        core_radius_0=0.05,
        core_radius_1=0.3,
    ).image_2d_from(grid=grid),
    title="Chameleon Image",
)

aplt.plot_array(
    array=ag.lp.ElsonFreeFall(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        eta=2.0,
    ).image_2d_from(grid=grid),
    title="ElsonFreeFall Image",
)

"""
The spherical variants (`SersicSph`, `GaussianSph`, etc.) are constructed identically with
the `ell_comps` argument removed — they look like a rotationally symmetric version of the
corresponding elliptical plot.

The shapelet profiles are normally used inside a `Basis` rather than individually, but for
completeness here is the lowest-order Cartesian shapelet on its own:
"""
aplt.plot_array(
    array=ag.lp.ShapeletCartesian(
        n_y=0,
        n_x=0,
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        beta=0.2,
    ).image_2d_from(grid=grid),
    title="ShapeletCartesian (n_y=0, n_x=0) Image",
)

"""
And that completes the tour.  If you arrived here from the API reference and now want to use
any of these profiles in an actual fit, the next step is `scripts/imaging/modeling.py`,
which sets up an `AnalysisImaging` and runs a non-linear search end-to-end.  The
`scripts/imaging/features/` subpackages handle the family-specific workflows referenced
throughout this guide:

- `linear_light_profiles/` — using `ag.lp_linear.*` in a fit.
- `operated_light_profile/` — using `ag.lp_operated.*` in a fit.
- `multi_gaussian_expansion/` — building and fitting an MGE-style `Basis`.
- `shapelets/` — building and fitting a shapelet-style `Basis`.
"""
