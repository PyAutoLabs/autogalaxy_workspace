"""
Features: Pixelization Modeling
===============================

A pixelization reconstructs a galaxy’s light using a pixel grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a galaxy model which uses a pixelization to reconstruct the galaxy’s light.

A rectangular mesh and constant regularization scheme are used, which are the simplest forms of mesh and regularization
and provide computationally fast and accurate solutions.

For simplicity, the galaxy is modeled using only a pixelized light component. Including additional parametric
light components is straightforward and can be done within the same framework.

You may wish to first read the `pixelization/fit.py` example, which demonstrates how a pixelized galaxy reconstruction
is applied to a single dataset.

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

**Advantages & Disadvantages:** Benefits and drawbacks of using a pixelization to model galaxy light.
**Positive Only Solver:** How a positive solution to the reconstructed pixel fluxes is ensured.
**Dataset & Mask:** Standard setup of the imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard setup of non-linear search and analysis.
**Run Time:** Profiling of pixelization run times and discussion of how they compare to analytic light profiles.
**Model-Fit:** Performs the model fit using the standard API.
**Result:** Pixelization results and visualization.
**Including Smooth Components:** How to combine a pixelization with parametric light profiles to model both smooth and complex galaxy structures.
**Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve lens modeling with pixelizaitons.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a strong lens dataset with the inferred pixelized source.

__Advantages__

Many galaxies exhibit complex, asymmetric, and irregular morphologies. Such structures cannot be well approximated by
analytic light profiles such as a Sérsic profile, or even combinations of multiple Sérsic components. Pixelizations are
therefore required to accurately reconstruct this irregular galaxy light.

Even alternative basis-function approaches, such as shapelets or multi-Gaussian expansions, struggle to accurately
reconstruct galaxies with highly complex morphologies or multiple distinct components.

Pixelized galaxy models are therefore essential when the goal is to recover detailed structure in galaxy light
distributions beyond what is possible with parametric profiles.

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

 - The galaxy’s surface brightness is reconstructed using a pixelization.
 - A `RectangularMagnification` mesh and `Constant` regularization scheme are used.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__


Load and plot the strong lens dataset `simple__sersic` via .fits files
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()


"""
__Over Sampling__

A pixelization uses a separate grid for light evaluation, with its own over sampling scheme, which below we set to a 
uniform grid of values of 4. 

Note that the over sampling is input into the `over_sample_size_pixelization` because we are using a `Pixelization`.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_pixelization=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.
"""
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = ag.Preloads(
    mapper_indices=ag.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=ag.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a model where:

 - The galaxy's light uses a 20 x 20 `RectangularMagnification` mesh [0 parameters]. 
 
 - This pixelization is regularized using a `GaussianKernel` scheme which smooths every source [2 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=2. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (3 parameters) than 
fitting this complex galaxy using parametric light profiles would (20+ parameters). 
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

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
    n_batch=20,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data. 
"""
analysis = ag.AnalysisImaging(dataset=dataset, preloads=preloads, use_jax=True)


"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

pixelizations use a lot more VRAM than light profile-only models, with the amount required depending on the size of
dataset and the number of source pixels in the pixelization's mesh. For 400 source pixels, around 0.05 GB per batched
likelihood of VRAM is used. 

This is why the `batch_size` above is 20, lower than other examples, because reducing the batch size ensures a more 
modest amount of VRAM is used. If you have a GPU with more VRAM, increasing the batch size will lead to faster run times.

Given VRAM use is an important consideration, we print out the estimated VRAM required for this 
model-fit and advise you do this for your own pixelization model-fits.
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

The galaxy bulge and disk appear similar to those in the data, confirming that the `intensity` values inferred by
the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Including Smooth Components__

By combining a pixelization with parametric light profiles, we can model galaxies whose light consists of both smooth
and complex, irregular components. For example, we can quantify the light in bulge and disk components while
simultaneously reconstructing irregular features such as spiral arms or star-forming clumps using a pixelization.
This allows a physically meaningful decomposition of a galaxy into its main structural components and a robust
measurement of their properties.

Combining a pixelization with parametric light profiles is straightforward: we simply add light profiles to the
galaxy model alongside the pixelization using the standard modeling API. Below, we use linear light profiles to
maximize computational efficiency, although non-linear light profiles can also be used.

For brevity, we do not perform the model fit here. The code below demonstrates how to construct such a model, which
can then be fitted using the same search and analysis objects introduced above.
"""
pixelization = af.Model(
    ag.Pixelization,
    bulge=ag.lp_linear.Sersic,
    disk=ag.lp_linear.Exponential,
    mesh=ag.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Mask Extra Galaxies__

There may be extra galaxies nearby the main galaxy, whose emission blends with it.

If their emission is significant, and close enough to the galaxy, we may simply remove the emission from the data
to ensure it does not impact the model-fit. A standard masking approach would be to remove the image pixels containing
the emission of these galaxies altogether. This is analogous to what the circular masks used throughout the examples
does.

For fits using a pixelization, masking regions of the image in a way that removes their image pixels entirely from
the fit. This can produce discontinuities in the pixelixation used to reconstruct the source and produce unexpected
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

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We do not explictly fit this data, for the sake of brevity, however if your data has these nearby galaxies you should
apply the mask as above before fitting the data.

__Result Use__

There are many things you can do with the result of a pixelixaiton, including analysing the galaxy reconstruction.

These are documented in the `fit.py` example.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
__Wrap Up__

Pixelizations are the most complex but also the most powerful way to model a galaxy’s light.

Whether you need to use them depends on the science you are doing. If you are only interested in measuring simple
global quantities (for example, total flux, size, or axis ratio), analytic light profiles such as a Sérsic, MGE, or
shapelets are often sufficient. For low-resolution data, pixelizations are also unnecessary, as the complex
structure of the galaxy is not resolved.

However, modeling galaxies with complex, irregular, or highly structured light distributions requires this level of
flexibility. Furthermore, if you are interested in studying the detailed morphology of a galaxy itself, there is no
better approach than using a pixelization.

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
