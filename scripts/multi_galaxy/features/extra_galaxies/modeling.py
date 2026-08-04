"""
Modeling Features (Multi Galaxy): Extra Galaxies
================================================

A blended pair of galaxies rarely sits alone in the field. There are usually other, fainter galaxies whose
emission falls inside the mask, and which will bias the decomposition of the pair if nothing is done about them.

This script extends the multi-galaxy model with a lower tier for those galaxies. Everything the
`imaging/features/extra_galaxies` example teaches carries over unchanged — the same two levers (scale their
emission out of the fit, or model their light with a fixed centre), the same `extra_galaxies` collection, the
same `SersicSph`-vs-MGE choice. What is new here is that the model now has **two tiers at once**, and the
interesting question becomes which tier a given galaxy belongs in.

__The Two Tiers__

 - **Co-equal galaxies** (`galaxies=af.Collection(galaxy_0=..., galaxy_1=...)`). Each gets a full free light
   model with a free centre. Use this tier when a galaxy's light genuinely blends with the others and
   decomposing it is part of the science. This is the multi-galaxy regime's defining feature.

 - **Extra galaxies** (`extra_galaxies=af.Collection(...)`). A restricted light model with the centre **fixed**
   to the observed position. Use this tier when a galaxy has to be accounted for but is not itself a subject of
   the measurement.

The tiers are not a statement about brightness, they are a statement about intent. A galaxy you want photometry
for belongs in the first tier no matter how faint; a galaxy you merely need to stop contaminating the fit
belongs in the second no matter how bright. The `simulator.py` script places two faint, well-separated
companions in the second tier, which is the common case.

__Contents__

- **Dataset:** Load the blended pair plus companions (auto-simulating if absent).
- **Mask:** A larger mask than `multi_galaxy/modeling.py` uses, to admit the extra galaxies.
- **Over Sampling:** Applied at the centres of all four galaxies.
- **Extra Galaxies Noise Scaling:** The first lever — scale their emission out and fit the pair alone.
- **Extra Galaxies Dataset:** Reload without noise scaling for the modeling approach.
- **Centres:** Load the two centre files, one per tier.
- **Model:** One MGE per co-equal galaxy, via the list-based API.
- **Extra Galaxies Model: Two Options:** SersicSph (Option A, default) vs MGE (Option B, commented out).
- **Search + Analysis:** Configure the Nautilus search and the `AnalysisImaging`.
- **Run Time:** What the extra tier costs.
- **Model-Fit:** Run the fit.
- **Result:** Per-galaxy decomposition, with the extra galaxies separated from the pair.
- **Approaches to Extra Galaxies:** Choosing between the two levers at multi-galaxy scale.
- **Choosing a Tier:** The decision this feature exists to support.
- **Wrap Up:** Where to go next up the regime ladder.

__Model__

This script fits `Imaging` of a multi-galaxy system with a model where:

 - Each of the two co-equal galaxies has a Multi Gaussian Expansion bulge with a free centre.
 - Each extra galaxy has a linear `SersicSph` bulge with its centre fixed to the observed position
   [2 extra galaxies x 2 parameters = 4 parameters].

__Start Here Notebook__

If any code in this script is unclear, refer to the `multi_galaxy/start_here.ipynb` notebook, and to
`imaging/features/extra_galaxies/modeling.ipynb` for the single-galaxy version of this feature.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the multi-galaxy dataset `extra_galaxies`, which is a blended pair with two fainter companions.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/extra_galaxies/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
Visualization of this dataset shows the blended pair at the centre, with the two extra galaxies further out.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We define a bigger circular mask of 6.0" than the 3.0" mask used in `multi_galaxy/modeling.py`, to ensure the
extra galaxies' emission is included.

This is worth pausing on, because it is the step most easily skipped. Keeping the 3.0" mask would exclude the
companions and appear to solve the problem — but only for this dataset, where they happen to sit outside it.
Shrinking the mask until the contaminants fall outside is a fragile strategy: it throws away real signal from
the pair's outskirts, and it fails entirely for a companion projected close to the blend. The two levers below
handle the general case.
"""
mask_radius = 6.0

mask_main = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask_main)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres__

Load the centres of both tiers. Two separate files are used, and the split is the tier assignment made concrete:

 - `galaxy_centres.json` — the co-equal pair. These *initialize* free centre priors.
 - `extra_galaxies_centres.json` — the companions. These *fix* the light profile centres.

Both are loaded now because the over sampling below needs all four.
"""
galaxy_centres = ag.from_json(file_path=dataset_path / "galaxy_centres.json")

extra_galaxies_centres = ag.Grid2DIrregular(
    ag.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(f"Co-equal galaxy centres: {galaxy_centres}")
print(f"Extra galaxy centres:    {extra_galaxies_centres}")

"""
__Over Sampling__

Over sampling evaluates light profiles on a higher resolution grid than the image data, so the calculation is
accurate at the peaked centre of each galaxy. It is applied at the centres of **all four** galaxies — omitting
the extras would make their light inaccurate exactly where it is brightest, which then leaks into the pair's
model as residuals.

Once you are more experienced, read up on over-sampling via
`autogalaxy_workspace/*/guides/advanced/over_sampling.ipynb`.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(galaxy_centres) + extra_galaxies_centres.in_list,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Noise Scaling__

The first lever. Rather than masking the extra galaxies' pixels out of the fit entirely, their data values are
scaled to zero and their noise-map values inflated, so they contribute negligibly to the likelihood while their
pixels remain in the fit.

Keeping the pixels is what distinguishes this from simply shrinking the mask. Removing pixels leaves holes in
the fitted region, which causes problems for approaches that assume a contiguous grid — a pixelized
reconstruction, for instance, develops discontinuities at the hole edges. Scaling keeps the geometry intact.

We reload the dataset first to ensure the trimming performed for the previous mask does not affect the
noise-scaling.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

dataset = dataset.apply_mask(mask=mask_main)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
The subplot below shows the extra galaxies' data values scaled to zero and their noise-map values raised, making
their signal-to-noise effectively zero.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
We now fit the pair using the standard multi-galaxy API, with the extra galaxies absent from the model. The
noise scaling ensures they do not impact the fit.

Note that the model here is exactly the model of `multi_galaxy/modeling.py` — one MGE per co-equal galaxy. That
is the appeal of this lever: the extra galaxies are handled entirely in the data, so the model composition is
untouched.
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

search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="extra_galaxies_noise_scaling",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=1000,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

"""
__Extra Galaxies Dataset__

We now model the dataset with the extra galaxies **included** in the model, so we reload it and apply only the
circular mask — the extra galaxies mask is not used, because their emission is now fitted rather than scaled
away.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset = dataset.apply_mask(mask=mask_main)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

The co-equal tier, unchanged from `multi_galaxy/modeling.py`: one MGE per blended galaxy, composed in a loop
over `galaxy_centres`, each with its centre prior initialized on the observed position. The MGE's Gaussians
share each galaxy's centre and ellipticity, so each galaxy stays cheap in non-linear parameters despite being
flexible, and their `intensity` values are solved exactly by linear inversion rather than sampled.

A full description of model composition is provided by the model cookbook:

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html
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

"""
__Extra Galaxies Model: Two Options__

The second lever, and the lower tier of the model. The extra galaxies API requires the centres of the light
profiles to be fixed to the input centres, while the other parameters stay free.

Fixing the centres matters more here than in the single-galaxy case. A free-centre extra galaxy can wander, and
in a multi-galaxy field there is now more than one bright thing for it to wander onto — it may drift towards a
galaxy of the pair and start absorbing light that the co-equal tier is supposed to be measuring, quietly
corrupting the decomposition that is the whole point of the fit.

There are two equally-supported ways to compose the per-galaxy bulge, exactly as in
`imaging/features/extra_galaxies/modeling.py`:

**Option A (default) — one `SersicSph` per extra galaxy.** Each extra galaxy gets a single linear spherical
Sersic with fixed centre, adding ~2 free parameters per galaxy (the linear inversion solves intensity). Concise;
ideal for a small handful of bright, roughly-symmetric companions.

**Option B (commented out) — MGE bulges via `ag.model_util.mge_model_from(centre_fixed=...)`.** Each extra
galaxy gets ~10 Gaussians sharing centre and ellipticity, with fixed centre. The two free `ell_comps` parameters
per galaxy match Option A's count in the linear-light limit, but the basis captures irregular or asymmetric
morphologies far better. Recommended once the number of extras grows beyond a handful.

Both leave the centres fixed; the choice is purely which basis fits the bulge.
"""
# Extra Galaxies:

# Option A (default): one SersicSph per extra galaxy
extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:
    extra_galaxy = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.SersicSph,
    )

    extra_galaxy.bulge.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

# Option B (uncomment to use): MGE bulges via mge_model_from(centre_fixed=...)
# extra_galaxies_list = []
#
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
    galaxies=af.Collection(**galaxy_dict), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model has both tiers — the co-equal galaxies under `galaxies`, and the
companions under `extra_galaxies` with their centres fixed.
"""
print(model.info)

"""
__Search + Analysis__

The code below performs the normal steps to set up a model-fit.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=1000,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Time__

Adding extra galaxies increases the likelihood evaluation time, because each one's light profile image must be
evaluated and blurred. For profiles like `SersicSph` this is fast, so only a small increase is expected.

The bigger cost is the extra free parameters, which raise the dimensionality of parameter space and so the
number of iterations Nautilus needs. Note the cost is per-galaxy: the extras tier grows linearly with the number
of companions, which is why the MGE option (Option B) becomes attractive as that number rises — it buys much
more morphological flexibility at the same per-galaxy parameter count.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute confirms both tiers were fitted.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The per-galaxy decomposition is the core deliverable of a multi-galaxy fit, and it now covers every galaxy in
the model — the two of the pair and the two extras, in that order, since `extra_galaxies` are appended after
`galaxies`.

The first two subplots are the ones the science depends on: each shows a co-equal galaxy's modeled light with
the other galaxies subtracted. Compare them against the noise-scaling fit above to see what including the extras
changed.
"""
galaxies = result.max_log_likelihood_galaxies

for i in range(len(galaxies)):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=result.max_log_likelihood_fit, galaxy_index=i
    )

"""
__Photometry__

With every galaxy decomposed, per-galaxy photometry is direct — each galaxy's model image contains only its own
light. The extras are included below so you can check they came out faint, which is the sanity check that they
were assigned to the right tier.

Note the labelling. `max_log_likelihood_galaxies` is a flat list with the extras appended after the co-equal
tier, so index `i` alone does not tell you which tier a galaxy came from. We label them explicitly below using
the number of co-equal galaxies as the split point — worth doing in your own scripts too, since silently
reading `galaxies[2]` as "the third co-equal galaxy" is an easy mistake once a model has two tiers.
"""
n_co_equal = len(galaxy_centres)

for i, galaxy in enumerate(galaxies):
    image = galaxy.image_2d_from(grid=dataset.grids.lp)

    label = f"galaxy_{i}" if i < n_co_equal else f"extra_galaxy_{i - n_co_equal}"

    print(f"{label}: total model flux = {float(image.array.sum()):.3f}")

"""
Checkout `autogalaxy_workspace/*/results` for a full description of analysing results.

__Approaches to Extra Galaxies__

We illustrated the two levers:

- **Noise Scaling**: the extra galaxies' emission is scaled out of the fit, and the model is exactly the model
  of `multi_galaxy/modeling.py`.

- **Modeling**: the extra galaxies are included in the model as a second tier with fixed centres.

At multi-galaxy scale the choice tilts towards modeling more often than it does for a single galaxy, for one
reason: the deliverable is a *decomposition*, and a decomposition is more sensitive to unmodelled flux than a
single galaxy's parameters are. Light that the noise scaling does not quite remove has to be absorbed by
something, and with two free light models available it will be absorbed asymmetrically — biasing the flux ratio
between the pair, which is usually the quantity the analysis exists to measure.

Noise scaling remains the right choice when a companion is far from the pair, or is contaminated itself (a
diffraction spike, a satellite trail) so no light profile would describe it honestly.

There are approaches between the two extremes. You can noise-scale one companion and model another, make the
extra galaxies' `effective_radius` priors tighter, or promote a companion to the co-equal tier if the fit shows
it is brighter than expected.

__Choosing a Tier__

The decision this feature exists to support. Put a galaxy in the co-equal tier when you want a measurement of
it; in the extras tier when you only need it to stop contaminating the measurement; and use noise scaling when
you need neither and its light is cleanly separable.

The practical trap is the middle case being mistaken for the first. Adding a free light model with a free centre
for every visible galaxy in the field feels safer than deciding, but it is not — it inflates dimensionality and
introduces exactly the wandering-component failure the fixed centres are there to prevent.

__Wrap Up__

The extra galaxies API composes cleanly with the multi-galaxy model: the co-equal tier is unchanged, and the
extras are added alongside it.

Where to go next:

- `autogalaxy_workspace/*/imaging/features/extra_galaxies` — the single-galaxy version of this feature, with
  more detail on the API itself and on why the autolens scaling-relation tier does not transfer to a light-only
  fit.
- `autogalaxy_workspace/*/imaging/features` — linear light profiles, MGE variations and sky handling, all of
  which apply per-galaxy here unchanged.
- `autogalaxy_workspace/*/cluster` — the top rung of the ladder, where the member population is loaded from a
  CSV catalogue rather than modeled galaxy-by-galaxy. That is where to go when the extras tier stops scaling.
"""
