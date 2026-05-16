"""
Modeling Features: Linear Light Profiles Fit (Interferometer)
==============================================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved
for via linear algebra every time the model is fitted to the data. This uses a process called an "inversion"
and it always computes the `intensity` values that give the best fit to the data (e.g. maximize the
likelihood) given the light profile's other parameters.

This script illustrates how to perform a single `FitInterferometer` of a linear light profile model — that
is, not the full Nautilus model-fit, but a single likelihood evaluation given known light profile parameters.
This is useful for understanding how the inversion produces the solved-for `intensity` values, and how to
extract them from the resulting fit.

For an explanation of why linear light profile fits are now practical against visibility data thanks to the
JAX-native NUFFT `nufftax` (https://github.com/GragasLab/nufftax), see the companion `modeling.py` example.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of linear light profiles for interferometer data.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **Model:** The galaxy model whose `intensity` values we solve for via inversion.
- **Mask:** Define the `real_space_mask` which sets the grid the galaxy is evaluated on.
- **Dataset:** Load the `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Fit:** Perform a single `FitInterferometer` using the model and inspect the inversion.
- **Intensities:** Extract the solved-for `intensity` values via `fit.linear_light_profile_intensity_dict`.
- **Visualization:** Build the helper galaxies object where linear light profiles are replaced with ordinary
  light profiles carrying their solved-for `intensity`, then plot.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example by 2
dimensions).

This also removes the degeneracies that occur between the `intensity` and other light profile parameters
(e.g. `effective_radius`, `sersic_index`), which are difficult degeneracies for the non-linear search to map
out accurately.

The inversion has a relatively small computational cost on top of the NUFFT, so we reduce the model
complexity without much slow-down.

__Disadvantages__

Although the computation time of the inversion is small, it is not non-negligible. It is approximately 3-4x
slower per inversion-only term than using a standard light profile with a fixed `intensity`. The NUFFT
typically dominates the total per-likelihood cost on interferometer data, so the overall slow-down is
usually smaller.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and
negative values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a
galaxy's light, which is clearly unphysical.

**PyAutoGalaxy** uses a positive only linear algebra solver which has been extensively optimized to ensure
it is as fast as positive-negative solvers. This ensures that all light profile intensities are positive and
therefore physical.

__Model__

This script fits an `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's bulge is a linear parametric `Sersic`.
 - The galaxy's disk is a linear parametric `Exponential`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.
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

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, using `TransformerNUFFT` backed
by `nufftax`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
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
__Fit__

We now illustrate how to perform a fit to the dataset using linear light profiles, with the bulge and disk
shape parameters fixed to known values.

The API follows closely the standard use of a `FitInterferometer` object, but simply uses linear light
profiles (via the `lp_linear` module) instead of standard light profiles.

Note that the linear light profiles below do not have `intensity` parameters input — we let the inversion
solve for them.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp_linear.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=ag.lp_linear.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        effective_radius=1.6,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

"""
The fit's `subplot_fit_interferometer` shows the visibility-plane fit and dirty-image residuals. Because the
bulge and disk are linear light profiles, the inversion has solved for their `intensity` values to maximize
the fit to the observed visibilities.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
The `subplot_fit_dirty_images` provides a real-space view of the data, model and residuals via inverse-NUFFT
of the visibility-plane quantities. This is generally more interpretable to the human eye than the uv-plane
plots above.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Intensities__

The fit contains the solved-for `intensity` values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the `max_log_likelihood` quantities
covered in `modeling.py`.
"""
bulge = galaxies[0].bulge
disk = galaxies[0].disk

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of bulge (lp_linear.Sersic) = {fit.linear_light_profile_intensity_dict[bulge]}"
)

print(
    f"\n Intensity of disk (lp_linear.Exponential) = {fit.linear_light_profile_intensity_dict[disk]}"
)

"""
A `Galaxies` object where all linear light profile objects are replaced with ordinary light profiles using the
solved-for `intensity` values is also accessible from a fit.

For example, the linear `Sersic` of the bulge above has a solved-for `intensity`. The `galaxies` object
created below instead has an ordinary `Sersic` light profile with that solved-for `intensity`.

The benefit of this galaxies object is that it can be visualised (linear light profiles cannot be plotted by
default because they do not have `intensity` values).
"""
galaxies = fit.model_obj_linear_light_profiles_to_light_profiles

print(galaxies[0].bulge.intensity)
print(galaxies[0].disk.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies) cannot be plotted because they do not have
an `intensity` value.

Therefore, the helper-galaxies object created above (with all linear light profiles replaced by ordinary
light profiles carrying their solved-for `intensity`) must be used for visualization:
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=dataset.grid), title="Galaxy Image")


"""
__Wrap Up__

Checkout `autogalaxy_workspace/*/guides/results` for a full description of analysing results.
"""
