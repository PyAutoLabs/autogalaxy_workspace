"""
Features: Pixelization
======================

A pixelization reconstructs a galaxy’s light on a grid of pixels, which is regularized using a prior that
enforces a degree of smoothness in the solution.

This script fits a galaxy model that uses a pixelization to reconstruct the galaxy’s light. It employs a
rectangular mesh with a constant regularization scheme, which together form the simplest pixelization and
regularization choices available. Despite their simplicity, these choices provide fast and accurate solutions.

For simplicity, the galaxy’s light is modeled using only a pixelization. For interferometer datasets, additional
light components are rarely required and this is the common scenario.

You may wish to first read the pixelization/fit.py example, which demonstrates how a pixelized reconstruction
is applied to a single dataset.

Pixelizations are covered in detail in Chapter 4 of the HowToGalaxy lecture series.

__Run Time Overview__

Throughout the workspace, it has been emphasised that pixelized reconstructions are computed using GPU or CPU
via JAX, where the linear algebra fully exploits sparsity in a way which minimizes VRAM use. This example uses
this functionality, and therefore is suitable for datasets with a low number of visibilities (e.g. < 10000) or
many visibilities (E.g. tens of millions).

This example fits the dataset with 273 visibilities used throughout the workspace, so the modeling runs in under 10
minutes. Fitting a higher resolution dataset will only take an hour to a few hours.

If your dataset contains many visibilities (e.g. millions), setting up the matrices for pixelized reconstruction
which speed up the linear algebra may take tens of minutes, or hours. Once you are comfortable with the API introduced
in this example, the `feature/pixelization/many_visibilities_preparation` explains how this initial setup can be
performed before galaxy modeling and saved to hard disk for fast loading before the model fit.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the reconstructed pixel fluxes can be ensured, but is often disabled for interferometer data.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Model:** Composing a model using a pixelization and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**VRAM:** Profiling of pixelization VRAM use and discussion of how it compares to standard light profiles.
**Run Time:** Profiling of pixelization run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Pixelization results and visualizaiton.
**Chaining:** How the advanced modeling feature, non-linear search chaining, can significantly improve galaxy modeling with pixelizaitons.
**Result (Advanced):** API for various pixelization outputs which requires some polishing.
**Simulate (Advanced):** Simulating an interferometer dataset with the inferred pixelized galaxy.

__Advantages__

Many galaxies exhibit complex, asymmetric, and irregular morphologies. Such structures cannot be well approximated by
analytic light profiles such as a Sérsic profile, or even combinations of multiple Sérsic components. Pixelizations are
therefore required to accurately reconstruct this irregular galaxy light.

Even alternative basis-function approaches, such as shapelets or multi-Gaussian expansions, struggle to accurately
reconstruct galaxies with highly complex morphologies or multiple distinct components.

Pixelized galaxy models are also essential for robustly constraining detailed components of a galaxy’s light
distribution. By fitting all of the galaxy light, they reduce degeneracies between different components of the model.

Finally, many science applications aim to study the galaxy itself in detail, in order to learn about distant and
intrinsically faint galaxies. Pixelizations reconstruct the intrinsic galaxy emission, enabling detailed studies of
galaxy structure.

For CCD imaging, a disadvantage of pixelized reconstructions is they are the most computationally expensive
modeling approach. However, for interferometer datasets, the way that JAX and GPUs can exploit the sparsity in the
linear algebra means pixelized reconstructions are both significantly faster than other approaches (E.g.
light profiles) and can scale to millions of visibilities.

__Disadvantages__

Galaxy modeling with pixelizations is conceptually more complex. There are additional failure modes, such as
solutions where the galaxy is reconstructed in an unphysical configuration. These issues are discussed in detail
later in the workspace.

As a result, learning to successfully fit galaxy models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This could be problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative pixels which over-fit
the data, producing unphysical solutions.

For CCD imaging datsets pixelized reconstructions use a positive-only solver, meaning that every pixel
is only allowed to reconstruct positive flux values. This ensures that the reconstruction is physical and
that we don't reconstruct negative flux values that don't exist in the real galaxy.

However, for interferometer datasets this positive-only solver is often disabled, because negative pixel values
can be observed from the measurement process. All interferometer examples therefore disable the positive only solver,
but you may want to consider if using the positive-only solver is appropriate for your dataset.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy’s light is modeled using a pixelization.
 - The galaxy’s surface-brightness is reconstructed using a `RectangularUniform` mesh
   and `Constant` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autogalaxy_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autogalaxy_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

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
__Mask__

Define the ‘real_space_mask’ which defines the grid the image is evaluated using.
"""
mask_radius = 3.5

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the `Interferometer` dataset `simple` from .fits files, which we will fit
with the galaxy model.

This includes the method used to Fourier transform the real-space image to the uv-plane and compare
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for
interferometer datasets containing ~1-10 million visibilities.

If you want to use the high resolution ALMA dataset, uncomment the relevant lines of code below after downloading
the data from the repository described in the "High Resolution Dataset" section above.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Sparse Operators__

Pixelized modeling requires dense linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **sparse linear algebra formalism**.

You do not need to understand the full details of the method, but the key point is:

- It exploits the **sparsity** of the matrices used in pixelized source reconstruction.
- This leads to a **significant speed-up on GPU or CPU**, using JAX to perform the linear algebra calculations.

To enable this feature, we call `apply_sparse_operator()` on the dataset. This computes and stores a NUFFT operator 
matrix.

On GPU via JAX, this computation is fast even for large datasets with many visibilities, with profiling
of high resolution datasets with over 1 million visibilities showing that computation takes under 20 seconds. For
10s or 100s of millions of visibilities computation on a GPU may stretch to minutes, but this is still very fast.

On CPU, for datasets with over 100000 visibilities and many pixels in their real-space mask, this computation
can take 10 minutes or hours (for the small dataset loaded above its miliseconds). The `show_progress` input outputs
a progress bar to the terminal so you can monitor the computation, which is useful when it is slow.

When computing it is slow, it is recommend you compute it once, save it to hard-disk, and load it
before modeling. The example `pixelization/many_visibilities_preparation.py` illustrates how to do this.
"""
dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

"""
__Settings__

As discussed above, disable the default position only linear algebra solver so the
reconstruction can have negative pixel values.
"""
settings_inversion = ag.SettingsInversion(use_positive_only_solver=False)

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used,
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data.

- `total_mapper_pixels`: The number of pixels in the rectangular pixelization mesh. This is required to set up
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of pixels on its edge, which when the reconstruction is computed
  are forced to values of zero.
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

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this
example fits a model where:

 - The galaxy's light uses a 20 x 20 `RectangularUniform` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme which smooths every pixel equally [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=1.

It is worth noting the pixelization fits the galaxy using significantly fewer parameters (1 parameter for
regularization) than fitting the galaxy using light profiles or an MGE (4+ parameters).

The model therefore includes a mesh and regularization scheme, which are used together to create the
pixelization.
"""
# Galaxy:
mesh = af.Model(ag.mesh.RectangularUniform, shape=mesh_shape)
regularization = af.Model(ag.reg.GaussianKernel)

pixelization = af.Model(ag.Pixelization, mesh=mesh, regularization=regularization)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

# Overall Model:
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the galaxy has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a
full description).
"""
search = af.Nautilus(
    path_prefix=Path("interferometer"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=50000,
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how via Nautilus the model is fitted to the data.

The `preloads` are passed to the analysis, which contain the static array information JAX needs to perform
the pixelization calculations.
"""
analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    preloads=preloads,
    settings_inversion=settings_inversion,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM
required by a model.

Pixelizations use a lot less VRAM than light profile-only models, provided the sparse operator
formalism is used (as it is above). In this mode, datasets with tens of millions of visibilities and real space
masks with pixel scales below 0.05" can be stored in just GB's of VRAM, which is remarkable given how much
data they contain.

In sparse operator mode, the **amount of VRAM used is independent of the number of visibilities in the dataset**.
This is because the sparse operator method compresses all the visibility information into sparse operator matrices,
whose size depends only on the number of pixels in the real-space mask. VRAM use is therefore mostly driven by
how many pixels are in the real space mask.

VRAM does scale with batch size though, and for high resoluiton datasets may require you to reduce from the value of
20 set above if your GPU does not have too much VRAM (e.g. < 4GB).
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

The run time of a pixelization are fast provided that the GPU VRAM exceeds the amount of memory required to perform
a likelihood evaluation.

The **run times of a pixelization are independent of the number of visibilities in the dataset**. This is again
because the sparse operator method compresses all the visibility information into the `nufft_precision_operator` matrix, 
whose size depends only on the number of pixels in the real-space mask.

Therefore, like VRAM, the main driver of run time is the number of pixels in the real-space mask,
not the number of visibilities in the dataset. The calculation also runs the same speed irrespective of whether
the real space mask is circular, or irregularly shaped, therefore using a circlular mask is recommended as it is
simpler to set up.

Assuming the use of a 20 x 20 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. Increasing the batch size will speed up the fit, provided VRAM allows it.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this
does not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms that the galaxy has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(result.info)

"""
We plot the maximum likelihood fit and posteriors inferred via Nautilus.

The end of this example provides a detailed description of all result options for a pixelization.
"""
print(result.max_log_likelihood_instance)

fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
The example `pixelization/fit` provides a full description of the different calculations that can be performed
with the result of a pixelization model-fit.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
__Wrap Up__

Pixelizations are the most complex but also most powerful way to model a galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in measuring a
simple quantity, you can get away with using light profiles like a Sersic, MGE or shapelets to model a galaxy. Low
resolution data also means that using a pixelization is not necessary, as the complex structure of the galaxy is not
resolved anyway.

However, modeling complex galaxy light distributions requires this level of flexibility. Furthermore, if you are
interested in studying the properties of the galaxy itself, you won't find a better way to do this than using a
pixelization.

__Chaining__

Modeling using a pixelization can be more efficient, robust and automated using the non-linear chaining feature to
compose a pipeline which begins by fitting a simpler model using a parametric galaxy.

More information on chaining is provided in the `autogalaxy_workspace/notebooks/guides/modeling/chaining` folder,
chapter 3 of the **HowToGalaxy** lectures.

__HowToGalaxy__

A full description of how pixelizations work, which comes down to a lot of linear algebra, Bayesian statistics and
2D geometry, is provided in chapter 4 of the **HowToGalaxy** lectures.

__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More diagnostic calculations.
- Gradient calculations.
- A calculation which shows differential effects across the reconstruction.
"""
