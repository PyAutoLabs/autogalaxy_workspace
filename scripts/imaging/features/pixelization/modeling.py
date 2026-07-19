"""
Features: Pixelization Modeling
===============================

This is the canonical example of when (and why) you should reach for a pixelization in **PyAutoGalaxy**.

The dataset (`dataset/imaging/clumpy`) shows a galaxy with two very different kinds of light:

 - A smooth, symmetric central **bulge** that is well described by a single Sersic profile.
 - **Asymmetric clumpy star formation** spread irregularly across the galaxy, which no parametric profile (or even a
   combination of profiles) can fit cleanly.

We therefore use a hybrid model: a linear `Sersic` for the bulge, and a pixelization (with a `RectangularAdaptDensity`
mesh and `GaussianKernel` regularization scheme) for the clumpy component. The Sersic captures the smooth bulge with
just a handful of parameters; the pixelization reconstructs whatever the Sersic cannot fit on a flexible pixel grid.

This split is the canonical use-case for a pixelization in galaxy modeling: parametric profile for the smooth part,
pixelization for the irregular part.

You may wish to first read the `pixelization/fit.py` example, which demonstrates how a bulge + pixelization galaxy
reconstruction is applied to a single dataset.

Pixelizations are covered in detail in chapter 4 of the **HowToGalaxy** lectures.

__Run Time Overview__

Pixelized galaxy reconstructions are computed using either GPU acceleration via JAX or CPU acceleration via `numba`.

The faster option depends on two crucial factors:

#### **1. GPU VRAM Limitations**
JAX only provides significant acceleration on GPUs with **large VRAM (≥16 GB)**.
To avoid excessive VRAM usage, examples often restrict pixelization meshes (e.g. 20 × 20).
On consumer GPUs with limited memory, **JAX may be slower than CPU execution**.

#### **2. Sparse Matrix Performance**

Pixelized inversions require operations on **very large, highly sparse matrices**.

- JAX currently lacks sparse-matrix support and must compute using **dense matrices**, which scale poorly.
- PyAutoGalaxy’s CPU implementation (via `numba`) fully exploits sparsity, providing large speed gains
  at **high image resolution** (e.g. `pixel_scales <= 0.03`).

As a result, CPU execution can outperform JAX even on powerful GPUs for high-resolution datasets.

The example `pixelization/cpu_fast_modeling` shows how to set up a pixelization to use efficient CPU calculations
via the library `numba`.

__Rule of Thumb__

For **low-resolution imaging** (for example, datasets with `pixel_scales > 0.05`), modeling is generally faster using
**JAX with a GPU**, because the computations involve fewer sparse operations and do not require large amounts of VRAM.

For **high-resolution imaging** (for example, `pixel_scales <= 0.03`), modeling can be faster using a **CPU with numba**
and multiple cores. At high resolution, the linear algebra is dominated by sparse matrix operations, and the CPU
implementation exploits sparsity more effectively, especially on systems with many CPU cores (e.g. HPC clusters).

**Recommendation:** The best choice depends on your hardware and dataset. If your data has resolution of 0.1" per pixel
(e.g. Euclid imaging) or lower, JAX will often be the most efficient. For higher resolution imaging (e.g. HST, JWST),
it is worth benchmarking both approaches (GPU+JAX vs CPU+numba) to determine which performs fastest for your case.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of using a pixelization to model galaxy light.
- **Positive Only Solver:** How a positive solution to the reconstructed pixel fluxes is ensured.
- **Dataset & Mask:** Standard setup of the clumpy imaging dataset that is fitted.
- **Pixelization:** How to create a pixelization, including a description of its inputs.
- **Model:** Composing a hybrid model with a parametric `Sersic` bulge and a pixelization for the clumps.
- **Search & Analysis:** Standard setup of non-linear search and analysis.
- **Run Time:** Profiling of pixelization run times and discussion of how they compare to analytic light profiles.
- **Model-Fit:** Performs the model fit using the standard API.
- **Result:** Galaxy reconstruction results and visualization.
- **Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve modeling with pixelizations.
- **Result (Advanced):** API for various pixelization outputs.
- **Wrap Up:** Summary and pointers to further reading.

__Advantages__

Many galaxies exhibit complex, asymmetric, and irregular morphologies — spiral arms, star-forming clumps, tidal
features, low-surface-brightness substructure. The clumpy dataset used here is a clean example: parametric profiles
fit the smooth bulge well but leave structured residuals where the clumps live.

Alternative basis-function approaches (shapelets, multi-Gaussian expansions) can absorb some of this irregular light
but typically struggle when the substructure is highly localised or asymmetric. A pixelization places no smoothness
assumption beyond the regularization prior, and so reconstructs irregular features directly on a flexible pixel grid.

Combining a parametric profile for the smooth component with a pixelization for the irregular component gives the
best of both worlds: a low-dimensional, physically interpretable description of the bulge, and a flexible flux map
for everything the bulge cannot explain.

Finally, many science applications aim to study galaxy morphology itself in detail, particularly for faint or
low-surface-brightness features. Pixelizations reconstruct the intrinsic galaxy light distribution, enabling these
studies.

__Disadvantages__

Pixelized galaxy reconstructions are computationally more expensive than analytic light-profile models. For
high-resolution imaging data (e.g. Hubble Space Telescope observations), fits using pixelizations can require
significantly longer run times.

Modeling galaxy light with pixelizations is also conceptually more complex, with additional failure modes compared to
parametric models, such as overfitting noise or producing overly complex reconstructions if regularization is not
chosen carefully.

As a result, learning to successfully fit galaxy models with pixelizations typically requires more time and
experience than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

Many codes which use linear algebra rely on solvers that allow both positive and negative values of the solution
(e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it allows negative surface-brightness values to represent a galaxy’s light, which is clearly
unphysical. For a pixelization, this often produces negative pixels that over-fit the data, leading to unphysical
solutions.

All pixelized galaxy reconstructions therefore use a positive-only solver, meaning that every pixel is only allowed
to reconstruct positive flux values. This ensures that the reconstruction is physical and prevents unphysical
negative solutions.

Enforcing this efficiently requires non-trivial linear algebra, so a bespoke fast non-negative solver was developed;
many methods in the literature omit this and therefore allow unphysical solutions that can degrade galaxy modeling
results.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's smooth central bulge is fit with a linear `Sersic` light profile.
 - The galaxy's asymmetric clumpy star formation is reconstructed using a pixelization with a
   `RectangularAdaptDensity` mesh and `GaussianKernel` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load and plot the `clumpy` imaging dataset, a galaxy with a smooth central bulge and asymmetric clumpy star formation.
"""
dataset_name = "clumpy"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/pixelization/simulator.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)


"""
__Over Sampling__

A pixelization uses a separate grid for light evaluation, with its own over sampling scheme. Below we use an
adaptive scheme that over-samples heavily (8x8) near the bulge centre, transitions through 4x4, then drops to
1x1 in the outskirts. The bulge centre is the only region where high over-sampling matters for the parametric
Sersic — the clumps are reconstructed by the pixelization itself and do not need profile over-sampling.

Note that the over sampling is input into the `over_sample_size_pixelization` because we are using a `Pixelization`.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_pixelization=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mesh Shape__

The `mesh_shape` parameter defines the number of pixels used by the rectangular mesh to reconstruct the galaxy,
set below to 28 x 28.

The `mesh_shape` must be fixed before modeling and cannot be a free parameter of the model, because JAX uses the
mesh shape to define static shaped arrays which use the mesh to reconstruct the galaxy. For a rectangular
mesh, the same number of pixels must be used in the y and x directions.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this example we fit
a single galaxy with two components:

 - The galaxy's smooth central **bulge** is fit with a linear elliptical `Sersic` light profile [6 parameters: centre,
   ell_comps, effective_radius, sersic_index]. Using `lp_linear.Sersic` rather than `lp.Sersic` means the bulge's
   `intensity` is solved for via the same linear inversion that solves for the pixelization reconstruction, removing
   one non-linear parameter and avoiding the bulge/pixelization brightness degeneracy.

 - The galaxy's asymmetric **clumpy star formation** is reconstructed with a 28 x 28 `RectangularAdaptDensity` mesh
   [0 parameters], regularized with a `GaussianKernel` scheme that smooths the reconstruction [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.

The pixelization absorbs whatever the Sersic cannot fit. The combination is dramatically more parsimonious than
trying to add more and more parametric light profiles (Sersics, Gaussians) until every clump is described —
a 20+ parameter approach that would still struggle with truly irregular substructure.
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.Sersic,
    pixelization=pixelization,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU model fits are batched and run simultaneously, see VRAM section below.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data. 
"""
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)


"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM
required by a model.

Pixelizations use a lot more VRAM than light profile-only models, with the amount required depending on the size of
the dataset and the number of mesh pixels in the pixelization. For 400 reconstruction pixels, around 0.05 GB per
batched likelihood of VRAM is used.

This is why the `batch_size` above is 20, lower than other examples, because reducing the batch size ensures a more
modest amount of VRAM is used. If you have a GPU with more VRAM, increasing the batch size will lead to faster run times.

Given VRAM use is an important consideration, we print out the estimated VRAM required for this
model-fit and advise you do this for your own pixelization model-fits.

The method below prints the VRAM usage estimate for the analysis and model with the specified batch size,
it takes about 20-30 seconds to run so you may want to comment it out once you are familiar with your GPU's VRAM limits.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

The run time of a pixelization are fast provided that the GPU VRAM exceeds the amount of memory required to perform
a likelihood evaluation.

Assuming the use of a 20 x 20 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. If VRAM is exceeded, the run time will be significantly longer (3+ hours). CPU run
times are also of order hours, but can be sped up using the `numba` library (see the `pixelization/cpu` example).

The run times of pixelizations slow down as the data becomes higher resolution. In this example, data with a pixel
scale of 0.1" gives of order 10 minute run times (when VRAM is under control), for a pixel scale of 0.05" this
becomes around 30 minutes, and an hour for 0.03".

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
We plot the maximum likelihood fit, galaxy images and posteriors inferred via Nautilus.

The reconstructed bulge image and the pixelized reconstruction of the clumps should together reproduce the data to
roughly the noise level, with the bulge component absorbing the smooth central light and the pixelization absorbing
the off-centre clumpy structure.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_galaxies(galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)


"""
__Adding More Light Profiles__

The model above pairs a single `Sersic` bulge with the pixelization. Galaxies with additional smooth components
(for example a bulge plus an extended disk) can include those components in exactly the same way by adding them as
extra attributes on the `Galaxy` model alongside the bulge and pixelization. We use linear light profiles
(`lp_linear.*`) to maximize computational efficiency — their intensities are solved for simultaneously with the
pixelization reconstruction.

For brevity, we do not perform a second model fit here. The code below demonstrates how to extend the model with a
disk, which can then be fitted using the same search and analysis objects introduced above.
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.Sersic,
    disk=ag.lp_linear.Exponential,
    pixelization=pixelization,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Mask Extra Galaxies__

There may be extra galaxies nearby the main galaxy, whose emission blends with it.

If their emission is significant, and close enough to the galaxy, we may simply remove the emission from the data
to ensure it does not impact the model-fit. A standard masking approach would be to remove the image pixels containing
the emission of these galaxies altogether. This is analogous to what the circular masks used throughout the examples
does.

For fits using a pixelization, masking regions of the image in a way that removes their image pixels entirely from
the fit can produce discontinuities in the pixelization used to reconstruct the galaxy and produce unexpected
systematics and unsatisfactory results. In this case, applying the mask in a way where the image pixels are not
removed from the fit, but their data and noise-map values are scaled such that they contribute negligibly to the fit,
is a better approach.

We illustrate the API for doing this below, using the `extra_galaxies` dataset which has extra galaxies whose emission
needs to be removed via scaling in this way. We apply the scaling and show the subplot imaging where the extra
galaxies mask has scaled the data values to zeros, increasing the noise-map values to large values and in turn made
the signal to noise of its pixels effectively zero.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/extra_galaxies/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=0.1, centre=(0.0, 0.0), radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
We do not explictly fit this data, for the sake of brevity, however if your data has these nearby galaxies you should
apply the mask as above before fitting the data.

__Result Use__

There are many things you can do with the result of a pixelization fit, including analysing the galaxy reconstruction
and exporting it for downstream science.

These are documented in the `fit.py` and `galaxy_reconstruction.py` examples.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
__Wrap Up__

Pixelizations are the most complex but also the most powerful way to model a galaxy's light.

Whether you need to use them depends on the science you are doing. If you are only interested in measuring simple
global quantities (for example, total flux, size, or axis ratio), analytic light profiles such as a Sérsic, MGE, or
shapelets are often sufficient. For low-resolution data, pixelizations are also unnecessary, as the irregular
structure of the galaxy is not resolved.

However, modeling galaxies with complex, irregular, or highly structured light distributions — like the asymmetric
clumpy galaxy fit here — requires this level of flexibility. Furthermore, if you are interested in studying the
detailed morphology of a galaxy itself, there is no better approach than combining a parametric bulge with a
pixelization for the irregular component.

__Chaining__

Modeling with a pixelization can be made more efficient, robust, and automated using the non-linear chaining feature
to compose a pipeline that begins by fitting a simpler model using parametric light profiles.

More information on chaining is provided in the
`autogalaxy_workspace/notebooks/guides/modeling/chaining` folder and in chapter 3 of the **HowToGalaxy** lectures.

__HowToGalaxy__

A full description of how pixelizations work—which relies heavily on linear algebra, Bayesian statistics, and
2D geometry—is provided in chapter 4 of the **HowToGalaxy** lectures.

__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More diagnostic quantities for reconstructed galaxy light.
- Gradient calculations of the reconstructed light distribution.
- Quantifying spatial variations in galaxy structure across the image.
"""
