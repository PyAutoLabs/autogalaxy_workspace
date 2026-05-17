"""
Modeling Features: Shapelets (Interferometer)
==============================================

A shapelet is a basis function appropriate for capturing the exponential / disk-like features of a galaxy.
It has been employed in galaxy-structure studies to model galaxy light because it can represent disky
star-forming features that a single Sersic function cannot.

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T
- https://iopscience.iop.org/article/10.1088/0004-637X/813/2/102

Shapelets are described in full in:

  https://arxiv.org/abs/astro-ph/0105178

This script performs galaxy modeling of an `Interferometer` dataset using a polar shapelet basis for the
galaxy bulge. The `intensity` of every shapelet is solved for via linear algebra (see the
`linear_light_profiles` feature for a full description of this).

Shapelet fits to interferometer data were previously impractical because every likelihood evaluation has
to Fourier-transform each shapelet basis component into the uv-plane. With `nufftax`
(https://github.com/GragasLab/nufftax) — a JAX-native NUFFT — the full shapelet basis is transformed
inside the same jit/vmap pipeline as the rest of the model, amortising the per-iteration NUFFT cost on
the GPU. Shapelet fits are now routine even at ALMA-class visibility counts.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of a shapelet basis for interferometer data.
- **NUFFT (nufftax):** Why shapelets-on-visibilities is now practical thanks to nufftax.
- **Positive Negative Solver:** Why shapelets require a positive-negative solver (unlike MGE or linear
  Sersic).
- **Model:** Compose the galaxy model — a polar shapelet bulge.
- **Mask:** Define the `real_space_mask` which sets the grid the galaxy is evaluated on.
- **Dataset:** Load the `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Over Sampling:** Interferometer modeling does not use over-sampling.
- **Search:** Configure the non-linear search (Nautilus).
- **Analysis:** Create the `AnalysisInterferometer` object with the positive-negative solver enabled.
- **VRAM:** Memory budget for a multi-component shapelet basis on GPU.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals because they fail to
capture irregular and asymmetric morphology of galaxies (e.g. disky star formation, isophotal twists).
Shapelets capture some of these features and can therefore better represent complex galaxies.

The shapelet model has fewer non-linear parameters than an elliptical Sersic. In this example, ~10
shapelets composed in a model correspond to just N=3 non-linear parameters (centre + shared beta); a
linear Sersic would have N=6.

__Disadvantages__

- There are many types of galaxy structure which shapelets may struggle to represent, such as a bar or
  asymmetric knots of star formation. Shapelets also rely on the galaxy having a distinct centre over
  which the basis can be centred.

- The linear algebra used to solve for the `intensity` of each shapelet has to allow for negative values
  of intensity. Negative surface brightnesses are unphysical, and are often inferred in a shapelet
  decomposition. Other approaches (MGE, pixelization) can force positive-only intensities on the
  solution.

- Computationally slower than a single linear `Sersic` because each shapelet must be NUFFT'd to the
  uv-plane per likelihood. With `nufftax` this is no longer a blocker.

__NUFFT (nufftax)__

The image-to-visibilities Fourier transform is performed by a Non-Uniform Fast Fourier Transform (NUFFT),
exposed in **PyAutoGalaxy** as `TransformerNUFFT`. The default backend is `nufftax`, a pure-JAX NUFFT:

  https://github.com/GragasLab/nufftax

Because `nufftax` is JAX-native, NUFFT-ing every shapelet basis image happens inside the same compiled
likelihood that does the inversion and chi-squared sum. There is no host round-trip between NUFFT calls,
so a model with N shapelets costs only N forward-NUFFTs per iteration on the GPU.

If `nufftax` is not installed, install it via `pip install nufftax`.

__Positive Negative Solver__

In other examples which use linear algebra to fit the data — linear light profiles, the Multi-Gaussian
Expansion (MGE) — we use a positive-only solver, which forces all solved-for intensities to be positive.
This is physical and sensible because the surface brightnesses of a galaxy cannot be negative.

Shapelets **cannot** be solved with a positive-only solver. Their ability to decompose the light of a
galaxy relies on being able to use negative intensities — shapelets are not physically motivated light
profiles but a mathematical basis that can represent any light profile, including via cancellations
between positive and negative basis-function amplitudes.

The `Settings` object passed to the analysis below uses `use_positive_only_solver=False` to allow for
negative intensities.

__Model__

This script fits an `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's bulge is a superposition of polar `ShapeletPolar` profiles, with all shapelets sharing a
   centre, elliptical components, and a single `beta` size scale.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script, see `autogalaxy_workspace/*/imaging/features/shapelets/modeling.py`.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Mask__

We define the `real_space_mask` which defines the grid the image of the galaxy is evaluated on.
"""
mask_radius = 4.0

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, using `TransformerNUFFT`
backed by `nufftax`.
"""
dataset_name = "simple"
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
        [sys.executable, "scripts/interferometer/simulator.py"],
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
__Over Sampling__

Interferometer data does not observe galaxies in a way where over sampling is necessary, therefore all
interferometer calculations are performed without over sampling.

__Model__

We compose a model where:

 - The galaxy's bulge is a superposition of linear `ShapeletPolar` profiles [3 parameters total].
   - All shapelets share a centre, elliptical components, and a single `beta` size scale.
   - The shapelet (n, m) quantum numbers are assigned procedurally.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(ag.lp_linear.ShapeletPolar) for _ in range(total_n + total_m + 1)
)

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    if i == 0:
        shapelet.n = 0
        shapelet.m = 0
    else:
        shapelet.n = n_count
        shapelet.m = m_count

        m_count += 2

        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
    shapelet.beta = shapelets_bulge_list[0].beta

bulge = af.Model(ag.lp_basis.Basis, profile_list=shapelets_bulge_list)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a
full description).
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="shapelets",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=20,  # GPU galaxy model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how Nautilus fits the model to the data.

Note `use_positive_only_solver=False` is set on the `Settings` — shapelets require the positive-negative
solver, as discussed above.
"""
analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    settings=ag.Settings(use_positive_only_solver=False),
    use_jax=True,
)

"""
__VRAM__

For each linear shapelet, extra VRAM is used to store its NUFFT'd mapping matrix column. For around 30
shapelets this typically requires a modest amount of VRAM (e.g. 10-50 MB per batched likelihood). Models
that use hundreds of shapelets, especially in combination with a large batch size, may therefore exceed
GBs of VRAM and require you to adjust the batch size to fit within your GPU's VRAM.

__Run Time__

The likelihood evaluation time for a shapelet basis is slower than a single linear `Sersic`, because
the image of every shapelet must be evaluated and NUFFT'd to the uv-plane. With `nufftax`, the per-NUFFT
cost is small enough that the total slow-down per likelihood is typically 2-5x for a 30-shapelet basis —
paid back in fewer iterations because the parameter space is simpler.

If shapelets are too slow for your science case, consider the MGE feature
(`interferometer/features/multi_gaussian_expansion/modeling.py`), which uses an even simpler basis (no
quantum-number indexing, just log-spaced sigmas) and is often faster.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the
output folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format.
"""
print(result.info)

aplt.subplot_galaxies(galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

This script has illustrated how to use shapelets to model the light of a galaxy in interferometer data.
For most science cases an MGE bulge (see `features/multi_gaussian_expansion/`) will be faster and give
higher quality results. Shapelets may perform better for disky / star-forming morphologies that the
smoother MGE basis struggles with, but this is not guaranteed — try both and compare.
"""
