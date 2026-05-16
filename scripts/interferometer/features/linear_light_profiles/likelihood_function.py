"""
__Log Likelihood Function: Linear Light Profile (Interferometer)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit
`Interferometer` data with parametric linear light profiles (e.g. a linear `Sersic` bulge and linear
`Exponential` disk).

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved
for via linear algebra every time the model is fitted to the data. This uses a process called an "inversion"
and it always computes the `intensity` values that give the best fit to the data (e.g. maximize the
likelihood) given the light profile's other parameters.

For interferometer data this is now practical thanks to the JAX-native NUFFT `nufftax`
(https://github.com/GragasLab/nufftax) — the image-to-uv Fourier transform of each linear basis component
happens inside the same jit/vmap pipeline as the rest of the model, so per-iteration NUFFT cost is amortised
on the GPU.

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoGalaxy**, so that readers can
   understand the likelihood function (including references to the previous literature from which it is
   defined) without having to write large quantities of text and equations.

 - To make linear inversions in **PyAutoGalaxy** less of a "black-box" to users.

__Prerequisites__

The likelihood function of a linear light profile builds on the standard parametric likelihood function and
on the interferometer-specific NUFFT step, so you must read the following notebooks before this script:

 - `interferometer/log_likelihood_function.ipynb` — the standard interferometer parametric likelihood
   function (NUFFT of a real-space image, visibility-plane $\\chi^2$).
 - `imaging/features/linear_light_profiles/likelihood_function.ipynb` — the linear-inversion linear algebra
   (data vector, curvature matrix, positive-only solver) for CCD imaging.

This script repeats just enough setup that you can follow it without rereading those two — but if anything
is unclear, those are the places to look first.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Mask:** Define the `real_space_mask` which sets the grid the galaxy is evaluated on.
- **Dataset:** Load and plot the `Interferometer` dataset using `TransformerNUFFT` (nufftax).
- **Galaxy Linear Light Profiles:** A linear `Sersic` bulge and linear `Exponential` disk.
- **Internal Intensity:** Why each linear light profile carries an internal `intensity=1.0`.
- **Mapping Matrix:** Real-space mapping matrix — one column per linear basis component, equal to the
  image of each component evaluated with `intensity=1.0`.
- **Transformed Mapping Matrix ($f$):** NUFFT each column to give the visibility-space mapping matrix used
  in the inversion.
- **Data Vector (D):** Compute $D$ from the transformed mapping matrix, visibilities, and noise map.
- **Curvature Matrix (F):** Compute $F$ separately for real and imaginary components, then sum.
- **Reconstruction (Positive-Negative):** Solve $s = F^{-1} D$ via NumPy.
- **Reconstruction (Positive Only):** Solve with the fast non-negative least squares (`fnnls`) algorithm.
- **Visibilities Reconstruction:** Map $s$ back to visibility space.
- **Likelihood Function:** Visibility-plane $\\chi^2$ and noise normalization.
- **Chi Squared:** Sum chi-squared contributions over real and imaginary components.
- **Noise Normalization Term:** The fixed noise normalization term.
- **Calculate The Log Likelihood:** Combine into the final log likelihood.
- **Fit:** Cross-check via `FitInterferometer`.
- **Galaxy Modeling:** How this likelihood is sampled in the full Nautilus fit.
- **Wrap Up:** Summary and next steps.
"""

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
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

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit with the
model. We use `TransformerNUFFT` (backed by `nufftax`), the JAX-native NUFFT that makes linear inversions in
the visibility plane practical at any visibility count.
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
__Galaxy Linear Light Profiles__

The galaxy is fitted using two **linear** light profiles: a linear `Sersic` bulge and a linear `Exponential`
disk. Compared to the standard parametric interferometer likelihood guide, we drop the `intensity` arguments
— they are solved for via the inversion below.
"""
bulge = ag.lp_linear.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp_linear.Exponential(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)
galaxies = ag.Galaxies(galaxies=[galaxy])

"""
__Internal Intensity__

Internally in the source code, linear light profiles still have an `intensity` parameter — its value is
fixed to 1.0. The inversion later rescales each column of the mapping matrix by the solved-for intensity.
Without an internal value of 1.0 the image evaluated below would be ambiguous in normalisation.
"""
print("Bulge Internal Intensity:")
print(bulge.intensity)

print("Disk Internal Intensity:")
print(disk.intensity)

"""
Like standard light profiles, we can compute images of each linear light profile, but their overall
normalisation is arbitrary given that the internal `intensity` value of 1.0 is used.
"""
image_2d_bulge = bulge.image_2d_from(grid=dataset.grids.lp)
image_2d_disk = disk.image_2d_from(grid=dataset.grids.lp)

"""
__Mapping Matrix__

For interferometer data the mapping matrix is the same conceptually as for imaging data: each column holds
the real-space image of one linear basis component, evaluated with `intensity=1.0`. No PSF convolution
(unlike imaging); the NUFFT will be applied to each column below.

We have two linear light profiles (the bulge and disk), so the mapping matrix has dimensions
`(total_real_space_pixels, 2)`.
"""
lp_linear_func = ag.LightProfileLinearObjFuncList(
    grid=dataset.grids.lp,
    blurring_grid=None,
    psf=None,
    light_profile_list=[bulge, disk],
    regularization=None,
)

mapping_matrix = lp_linear_func.mapping_matrix

print("Mapping matrix shape (real-space pixels, linear basis count):")
print(mapping_matrix.shape)

"""
Printing the first column of the mapping matrix shows the image of the bulge light profile. The second
column would show the disk.
"""
bulge_image = mapping_matrix[:, 0]
print(bulge_image)
print(image_2d_bulge.slim)

"""
A 2D plot of the `mapping_matrix` shows each light profile image in 1D, which is a bit odd to look at but
is a good way to think about the linear algebra.
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
__Transformed Mapping Matrix ($f$)__

To fit visibilities, every column of the real-space `mapping_matrix` must be NUFFT'd to the uv-plane. The
result is the `transformed_mapping_matrix` — a *complex-valued* matrix with dimensions
`(total_visibilities, total_linear_basis_components)`.

In our case it has two columns: the visibilities the bulge and the disk each contribute per unit
`intensity`. The inversion's job is to find the two scalars that, when multiplied by their respective
columns, best fit the observed visibilities.

The NUFFT here uses `TransformerNUFFT` (nufftax) — this is exactly the operation that used to be slow and
which nufftax has made fast. The whole pipeline (image → `transform_mapping_matrix` → linear inversion) is
JIT-compilable under JAX.
"""
transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

print("Transformed mapping matrix shape (visibilities, linear basis count):")
print(transformed_mapping_matrix.shape)

"""
Plot the real and imaginary components of the transformed mapping matrix. Each row is one observed
visibility; the values are the per-unit-intensity contributions of the bulge and disk at that uv point.
"""
plt.imshow(
    transformed_mapping_matrix.real,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.title("Re(f) — real component of transformed mapping matrix")
plt.show()
plt.close()

plt.imshow(
    transformed_mapping_matrix.imag,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.title("Im(f) — imaginary component of transformed mapping matrix")
plt.show()
plt.close()

"""
Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) (hereafter WD03) introduce the linear inversion
formalism used to compute the intensity values of the linear light profiles. WD03 indexes the transformed
mapping matrix as $f_{ij}$ where $i$ maps over all $I$ linear basis components and $j$ maps over all $J$
visibilities.

The indexing of the `transformed_mapping_matrix` array is reversed compared to the WD03 convention (rows
are visibilities, columns are basis components).

__Data Vector (D)__

To solve for the linear light profile intensities we pose the problem as a linear inversion.

This requires us to convert the `transformed_mapping_matrix` and our `data` and `noise_map` into matrices of
the right dimensions. The `data_vector`, $D$, has dimensions `(total_linear_basis_components,)`.

In WD03 the data vector is given by:

 $\\vec{D}_{i} = \\sum_{\\rm  j=1}^{J} f_{ij}\\, d_{j} / \\sigma_{j}^2 \\, ,$

where $d_j$ are the observed visibility values, $\\sigma_j^2$ are the visibility variances, and the sum runs
over real and imaginary components. The interferometer helper handles the real/imaginary split internally.

For our two-basis model $D$ is a length-2 vector — the noise-weighted overlaps of the bulge and the disk
contributions with the observed visibilities.
"""
data_vector = (
    ag.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=dataset.data,
        noise_map=dataset.noise_map,
    )
)

print("Data Vector D:")
print(data_vector)
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ has dimensions
`(total_linear_basis_components, total_linear_basis_components)`.

In WD03 the curvature matrix is given by:

 ${F}_{ik} = \\sum_{\\rm  j=1}^{J} f_{ij}\\, f_{kj} / \\sigma_{j}^2 \\, .$

Because visibilities (and therefore $f$) are complex-valued, the curvature is computed separately for the
real and imaginary parts and summed. For our two-basis model $F$ is a 2x2 matrix; the off-diagonal entries
$F_{01}$ and $F_{10}$ quantify how much the bulge and disk visibility patterns overlap (and therefore how
correlated their solved-for intensities will be).
"""
real_curvature_matrix = ag.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.real,
    noise_map=dataset.noise_map.real,
)

imag_curvature_matrix = ag.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.imag,
    noise_map=dataset.noise_map.imag,
)

curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)

print("Curvature Matrix F:")
print(curvature_matrix)

"""
__Reconstruction (Positive-Negative)__

The following chi-squared is minimized when we perform the inversion and solve for the bulge and disk
intensities:

$\\chi^2 = \\sum_{\\rm  j=1}^{J} \\bigg[ \\frac{(\\sum_{\\rm  i=1}^{I} s_{i} f_{ij}) - d_{j}}{\\sigma_{j}} \\bigg]^2$

Where $s_i$ is the solved-for `intensity` of the $i$-th linear basis component. The solution is given by
(equation 5 WD03):

 $s = F^{-1} D$

Computed with NumPy:
"""
reconstruction = np.linalg.solve(curvature_matrix, data_vector)

print("Reconstruction s (positive-negative solver):")
print(reconstruction)

"""
For linear light profiles fit to a clean dataset like ours, the solved-for `intensity` values are positive
and the positive-negative solution is physical. For more complex models (e.g. many components or noisy data)
the positive-negative solver can return negative `intensity` values — unphysical for a light profile.

__Reconstruction (Positive Only)__

The linear algebra can be solved with the constraint that all `intensity` values are positive. The naive
approach is `scipy.optimize.nnls`, which is iterative and works directly on the transformed mapping matrix
— it does not use `data_vector` or `curvature_matrix`. This is slow, especially with many linear components.

The source code therefore uses a "fast nnls" algorithm — an adaptation of:
  https://github.com/jvendrow/fnnls

`fnnls` *does* use `data_vector` $D$ and `curvature_matrix` $F$, which is why it is much faster. The
function `reconstruction_positive_only_from` wraps `fnnls`.
"""
reconstruction = ag.util.inversion.reconstruction_positive_only_from(
    data_vector=data_vector,
    curvature_reg_matrix=curvature_matrix,  # ignore the _reg_ tag in this guide
)

print("Reconstruction s (positive-only solver):")
print(reconstruction)

"""
__Visibilities Reconstruction__

Using the reconstructed `intensity` values we can map the reconstruction back to the visibility plane via
the `transformed_mapping_matrix`, producing the model visibilities.
"""
mapped_reconstructed_visibilities = (
    ag.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = ag.Visibilities(
    visibilities=mapped_reconstructed_visibilities
)

aplt.plot_grid(grid=mapped_reconstructed_visibilities.in_grid, title="Model Visibilities")


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our linear-light-profile reconstruction.

The likelihood function for parametric galaxy modeling, even with linear light profiles, consists of two
terms in the visibility plane:

 $-2 \\mathrm{ln} \\, \\epsilon = \\chi^2 + \\sum_{\\rm  j=1}^{J} { \\mathrm{ln}} \\left [2 \\pi (\\sigma_j)^2 \\right]  \\, .$

We now explain each term.

__Chi Squared__

The first term is a $\\chi^2$ statistic computed in the visibility plane:

 - `model_data` = `mapped_reconstructed_visibilities`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = `normalized_residual_map` ** 2.0
 - `chi_squared` = sum(`chi_squared_map`)

Visibilities are complex-valued, so we split into real and imaginary components, compute $\\chi^2$ for each,
and sum.
"""
model_visibilities = mapped_reconstructed_visibilities

residual_map = dataset.data - model_visibilities

chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2

chi_squared_real = np.sum(chi_squared_map_real)
chi_squared_imag = np.sum(chi_squared_map_imag)
chi_squared = chi_squared_real + chi_squared_imag

print(f"chi_squared (real + imag) = {chi_squared}")

"""
__Noise Normalization Term__

The likelihood function assumes the visibility data consists of independent Gaussian noise on every
visibility (real and imaginary parts treated independently).

The `noise_normalization` term is the sum of the log of every noise-map value squared. Because the
`noise_map` is fixed, this term does not change during modeling and has no impact on the inferred model —
it is included so that the absolute value of `log_likelihood` has the correct calibration for model
comparison.
"""
noise_normalization_real = float(np.sum(np.log(2 * np.pi * dataset.noise_map.real ** 2.0)))
noise_normalization_imag = float(np.sum(np.log(2 * np.pi * dataset.noise_map.imag ** 2.0)))
noise_normalization = noise_normalization_real + noise_normalization_imag

"""
__Calculate The Log Likelihood__

Combine the two terms to compute the `log_likelihood`:
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(f"log_likelihood (figure of merit) = {figure_of_merit}")


"""
__Fit__

The exact same likelihood evaluation is performed inside the `FitInterferometer` object. We construct one,
print its `figure_of_merit`, and confirm it matches the value we computed by hand above.
"""
fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)
print(f"FitInterferometer.figure_of_merit = {fit.figure_of_merit}")

aplt.subplot_fit_interferometer(fit=fit)

"""
The fit contains an `Inversion` object, which handles all the linear algebra we have covered in this script.
"""
print(fit.inversion)
print(fit.inversion.data_vector)
print(fit.inversion.curvature_matrix)
print(fit.inversion.reconstruction)

"""
__Galaxy Modeling__

To fit a galaxy model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus`
(https://github.com/johannesulf/nautilus); multiple MCMC and optimization algorithms are also supported.

For linear light profiles, the reduced number of free parameters (the `intensity` values are solved for via
the inversion instead of being non-linear search dimensions) means that the sampler converges in fewer
iterations and is less likely to be confused by intensity-shape degeneracies.

__Wrap Up__

We have presented a visual step-by-step guide to the linear light profile interferometer likelihood
function. The pipeline:

  image (linear profile, intensity=1) → `mapping_matrix`
  → NUFFT → `transformed_mapping_matrix` → $D$ and $F$ → solve $s = F^{-1} D$
  → `mapped_reconstructed_visibilities` → visibility-plane $\\chi^2$ → log likelihood

is the same as for CCD imaging linear light profiles, with the PSF convolution step replaced by the NUFFT
step. The NUFFT is the operation that nufftax made fast on the GPU, which is why this entire workflow is
now practical on interferometer data at any visibility count.

There are a number of other inputs which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in the `guides` package:

 - `over_sampling`: Not applicable to interferometer data (over-sampling is an imaging-only technique).
 - `pixelization`: For galaxies whose morphology cannot be captured by analytic light profiles, see the
   interferometer pixelization examples in `interferometer/features/pixelization/`.
"""
