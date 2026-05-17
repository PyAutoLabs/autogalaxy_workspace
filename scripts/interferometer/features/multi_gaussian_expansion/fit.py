"""
Modeling Features: Multi Gaussian Expansion Fit (Interferometer)
=================================================================

A multi-Gaussian expansion (MGE) decomposes a galaxy's light into ~15-100 Gaussians, where the `intensity`
of every Gaussian is solved for via linear algebra using a process called an "inversion".

This script illustrates how to perform a single `FitInterferometer` of an MGE galaxy model — that is, not
the full Nautilus model-fit, but a single likelihood evaluation given known basis parameters. This is
useful for understanding how the inversion produces the per-Gaussian solved-for `intensity` values, and
how to extract them from the resulting fit.

For an explanation of why MGE fits to visibility data are now practical thanks to the JAX-native NUFFT
`nufftax` (https://github.com/GragasLab/nufftax), see the companion `modeling.py` example.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of an MGE for interferometer data.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **Model:** The galaxy model whose `intensity` values we solve for via inversion — a 30-Gaussian MGE
  galaxy bulge.
- **Mask:** Define the `real_space_mask` which sets the grid the galaxy is evaluated on.
- **Dataset:** Load the `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Basis:** Build the linear Gaussian basis used as the galaxy bulge.
- **Fit:** Perform a single `FitInterferometer` and inspect the inversion.
- **Intensities:** Extract the per-Gaussian solved-for `intensity` values via
  `fit.linear_light_profile_intensity_dict`.
- **Visualization:** Build the helper galaxies object where linear light profiles are replaced with
  ordinary light profiles carrying their solved-for `intensity`, then plot.
- **Wrap Up:** Summary of the script and next steps.

__Positive Only Solver__

**PyAutoGalaxy** uses a positive only linear algebra solver for the MGE inversion which has been
extensively optimized to ensure it is as fast as positive-negative solvers. This ensures that all
Gaussian intensities are positive and therefore physical — without it, an unconstrained MGE inversion can
produce a positive-negative "ringing" pattern across the basis.

__Model__

This script fits an `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's bulge is a multi-Gaussian expansion of 30 linear `Gaussian` profiles, all sharing the
   same centre and `ell_comps`, with `sigma` values spanning 0.01" to the mask radius in log-spaced
   increments.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
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

We build a `Basis` of 30 linear `Gaussian` profiles, all sharing the same centre and `ell_comps`, with
`sigma` values spanning 0.01" to the mask radius in log-spaced increments.

We use linear light profile Gaussians (`lp_linear.Gaussian`), which solve for each Gaussian's `intensity`
analytically via the inversion. This is essential for MGE — a wide range of positive `intensity` values
are needed to decompose the galaxy's morphology, and we cannot guess them by eye.

Linear light profiles are described in detail in the `linear_light_profiles.py` example; you should
familiarize yourself with that example before using the multi-Gaussian expansion.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius.
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []
for i in range(total_gaussians):
    gaussian = ag.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        sigma=10 ** log10_sigma_list[i],
    )
    bulge_gaussian_list.append(gaussian)

bulge = ag.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
__Fit__

We now illustrate the API for performing a single MGE fit using standard `Galaxy`, `Galaxies` and
`FitInterferometer` objects. Once we have a `Basis`, we can treat it like any other light profile.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

"""
The fit's `subplot_fit_interferometer` shows the visibility-plane fit and dirty-image residuals. Because
the bulge is a Basis of linear light profiles, the inversion has solved for each Gaussian's `intensity` to
maximize the fit to the observed visibilities.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
The `subplot_fit_dirty_images` provides a real-space view of the data, model and residuals via
inverse-NUFFT of the visibility-plane quantities. This is generally more interpretable to the human eye
than the uv-plane plots above.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Intensities__

The fit contains the solved-for `intensity` value of every Gaussian in the MGE basis.

These are computed via `fit.linear_light_profile_intensity_dict`, which maps each linear light profile in
the model to its inferred `intensity`.

The code below prints the first five Gaussian intensities for brevity — for an MGE of N Gaussians the
full dict has N entries.
"""
print(fit.linear_light_profile_intensity_dict)

for gaussian in bulge_gaussian_list[:5]:
    print(
        f"  sigma = {gaussian.sigma:.4f}  "
        f"intensity = {fit.linear_light_profile_intensity_dict[gaussian]:.6e}"
    )

"""
A `Galaxies` object where all linear light profile objects are replaced with ordinary light profiles
using the solved-for `intensity` values is also accessible from a fit.

The benefit of this helper-galaxies object is that it can be visualised (linear light profiles cannot be
plotted by default because they do not have `intensity` values).
"""
galaxies = fit.model_obj_linear_light_profiles_to_light_profiles

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies) cannot be plotted because they do not
have an `intensity` value.

The helper-galaxies object created above replaces every linear `Gaussian` with an ordinary `Gaussian`
carrying its solved-for `intensity` — that object can be plotted directly.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=dataset.grid), title="Galaxy Image (MGE bulge)")


"""
__Wrap Up__

Checkout `autogalaxy_workspace/*/guides/results` for a full description of analysing results.
"""
