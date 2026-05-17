"""
Modeling Features: Shapelets Fit (Interferometer)
==================================================

A shapelet is a basis function appropriate for capturing the exponential / disk-like features of a galaxy.
The `intensity` of every shapelet is solved for via linear algebra ("inversion") rather than being a free
parameter of the non-linear search.

This script illustrates how to perform a single `FitInterferometer` of a shapelet galaxy model — that is,
not the full Nautilus model-fit, but a single likelihood evaluation given known basis parameters.

For an explanation of why shapelet fits to visibility data are now practical thanks to the JAX-native
NUFFT `nufftax` (https://github.com/GragasLab/nufftax), and why shapelets require the positive-negative
solver, see the companion `modeling.py` example.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of a shapelet basis for interferometer data.
- **Positive Negative Solver:** Why shapelets require the positive-negative solver.
- **Model:** The galaxy model whose `intensity` values we solve for via inversion.
- **Mask:** Define the `real_space_mask`.
- **Dataset:** Load the `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Basis:** Build the linear polar shapelet basis used as the galaxy bulge.
- **Fit:** Perform a single `FitInterferometer` and inspect the inversion.
- **Intensities:** Extract the per-shapelet solved-for `intensity` values.
- **Visualization:** Build the helper galaxies object and plot.
- **Wrap Up:** Summary of the script and next steps.

__Model__

This script fits an `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's bulge is a superposition of linear `ShapeletPolar` profiles with shared centre,
   ell_comps, and beta.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
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
__Basis__

We build a `Basis` of linear polar shapelets for the galaxy bulge. All shapelets share centre, ell_comps,
and a single `beta` size scale; the (n, m) quantum numbers are assigned procedurally.

We use `lp_linear.ShapeletPolar`, which solves for each shapelet's `intensity` analytically via the
inversion.
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = []
n_count = 1
m_count = -1

for i in range(total_n + total_m + 1):
    if i == 0:
        n, m = 0, 0
    else:
        n, m = n_count, m_count
        m_count += 2
        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet = ag.lp_linear.ShapeletPolar(
        n=n,
        m=m,
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        beta=0.5,
    )
    shapelets_bulge_list.append(shapelet)

bulge = ag.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__

We now illustrate the API for performing a single shapelet fit using standard `Galaxy`, `Galaxies` and
`FitInterferometer` objects. Once we have a `Basis`, we can treat it like any other light profile.

Note `Settings(use_positive_only_solver=False)` is passed to the fit — shapelets require the
positive-negative solver to function.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(
    dataset=dataset,
    galaxies=galaxies,
    settings=ag.Settings(use_positive_only_solver=False),
)

"""
The fit's `subplot_fit_interferometer` shows the visibility-plane fit and dirty-image residuals.
"""
aplt.subplot_fit_interferometer(fit=fit)

aplt.subplot_fit_dirty_images(fit=fit)

"""
__Intensities__

The fit contains the solved-for `intensity` value of every shapelet in the basis. Print the first few
entries for brevity.
"""
print(fit.linear_light_profile_intensity_dict)

for shapelet in shapelets_bulge_list[:5]:
    intensity = fit.linear_light_profile_intensity_dict[shapelet]
    print(f"  n={shapelet.n}  m={shapelet.m}  intensity = {intensity:+.6e}")

print(
    f"\n  number of negative-intensity shapelets: "
    f"{sum(1 for s in shapelets_bulge_list if fit.linear_light_profile_intensity_dict[s] < 0)} "
    f"/ {len(shapelets_bulge_list)}"
)

"""
A `Galaxies` object where all linear light profile objects are replaced with ordinary light profiles using
the solved-for `intensity` values is also accessible from a fit.
"""
galaxies = fit.model_obj_linear_light_profiles_to_light_profiles

"""
__Visualization__

The helper-galaxies object created above replaces every linear `ShapeletPolar` with an ordinary
`ShapeletPolar` carrying its solved-for `intensity` — that object can be plotted directly.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=dataset.grid), title="Galaxy Image (shapelet bulge)")


"""
__Wrap Up__

Checkout `autogalaxy_workspace/*/guides/results` for a full description of analysing results.
"""
