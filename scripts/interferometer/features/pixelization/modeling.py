"""
Features: Pixelization
======================

This is the canonical example of when (and why) you should reach for a pixelization to model interferometer
visibilities of a galaxy in **PyAutoGalaxy**.

The `clumpy` interferometer dataset is the Fourier transform of a galaxy with two very different kinds of light:

 - A smooth, symmetric central **bulge** that is well described by a single Sersic profile.
 - **Asymmetric clumpy star formation** spread irregularly across the galaxy, which no parametric profile (or
   combination of profiles) can fit cleanly.

We therefore use a hybrid model: a linear `Sersic` for the bulge, and a pixelization (with a `RectangularUniform`
mesh and `GaussianKernel` regularization scheme) for the clumpy component. The Sersic captures the smooth bulge
with just a handful of parameters; the pixelization reconstructs whatever the Sersic cannot fit on a flexible
pixel grid.

This split is the canonical use-case for a pixelization in galaxy modeling, and mirrors the imaging counterpart in
`autogalaxy_workspace/scripts/imaging/features/pixelization/modeling.py`. It relies on the sparse operator
formalism supporting linear light profiles alongside a pixelization, so that the bulge `intensity` and the mesh
pixel fluxes are solved for together in a single linear solve on the sparse path.

You may wish to first read the `pixelization/fit.py` example, which demonstrates how a bulge plus pixelization
galaxy reconstruction is applied to a single dataset.

Pixelizations are covered in detail in Chapter 3 of the HowToGalaxy lecture series.

__Run Time Overview__

Throughout the workspace, it has been emphasised that pixelized reconstructions are computed using GPU or CPU
via JAX, where the linear algebra fully exploits sparsity in a way which minimizes VRAM use. This example uses
this functionality, and therefore is suitable for datasets with a low number of visibilities (e.g. < 10000) or
many visibilities (E.g. tens of millions).

This example fits the dataset with 273 visibilities used throughout the workspace, so the modeling runs in under 10
minutes. Fitting a higher resolution dataset will only take an hour to a few hours.

If your dataset contains many visibilities (e.g. millions), setting up the matrices for pixelized reconstruction
which speed up the linear algebra may take tens of minutes, or hours. Once you are comfortable with the API introduced
in this example, the `features/pixelization/many_visibilities_preparation` explains how this initial setup can be
performed before galaxy modeling and saved to hard disk for fast loading before the model fit.

__Contents__

- **Advantages:** Benefits of using pixelizations to model galaxy light.
- **Disadvantages:** Drawbacks and additional complexity of pixelized galaxy modeling.
- **Positive Only Solver:** How a positive solution to pixel fluxes is ensured, and why it is often disabled for interferometer data.
- **Model:** Description of the hybrid bulge plus pixelization galaxy model fitted in this example.
- **Mask:** Defining the real-space mask for the interferometer grid.
- **Dataset:** Loading the interferometer dataset from FITS files.
- **Dataset Auto-Simulation:** Automatically simulating data if it does not exist.
- **Sparse Operators:** Computing the sparse NUFFT operator matrix for fast linear algebra.
- **Settings:** Disabling the positive-only solver for interferometer data.
- **Over Sampling:** Why over sampling is not needed for interferometer data.
- **Mesh Shape:** Defining the number of pixels used by the rectangular mesh.
- **Model:** Composing the model, a linear `Sersic` bulge plus a pixelization with a mesh and regularization scheme.
- **Search:** Configuring the Nautilus nested sampling non-linear search.
- **Analysis:** Setting up the AnalysisInterferometer object with JAX acceleration.
- **VRAM:** Estimating GPU VRAM requirements for pixelized modeling.
- **Run Time:** Profiling run times for pixelized modeling and how they scale.
- **Model-Fit:** Running the non-linear search to fit the model to data.
- **Result:** Inspecting the result object and maximum likelihood bulge and pixelized reconstruction.
- **Wrap Up:** Summary of when pixelizations are most useful.
- **Chaining:** Using non-linear search chaining to improve pixelized galaxy modeling.
- **HowToGalaxy:** Pointers to detailed pixelization tutorials in the lecture series.
- **Future Ideas / Contributions:** Potential future additions to this tutorial.

__Advantages__

Many galaxies exhibit complex, asymmetric, and irregular morphologies. Such structures cannot be well approximated by
analytic light profiles such as a Sérsic profile, or even combinations of multiple Sérsic components. Pixelizations are
therefore required to accurately reconstruct this irregular galaxy light.

Even alternative basis-function approaches, such as shapelets or multi-Gaussian expansions, struggle to accurately
reconstruct galaxies with highly complex morphologies or multiple distinct components.

Pixelized galaxy models are also essential for robustly constraining detailed components of a galaxy’s light
distribution. By pairing a parametric profile for the smooth bulge with a pixelization for the rest, we can
specifically estimate how much light is in the irregular components of a galaxy (e.g. spiral arms, star forming
clumps) compared to its smooth components, whilst reducing degeneracies between them.

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

This script fits an `Interferometer` dataset of a galaxy with a model where:

 - The galaxy's smooth central bulge is fit with a linear `Sersic` light profile.
 - The galaxy's asymmetric clumpy star formation is reconstructed using a pixelization with a
   `RectangularUniform` mesh and `GaussianKernel` regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/modeling.ipynb` notebook.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autogalaxy_workspace` repository:

https://github.com/PyAutoLabs/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autogalaxy_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

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
dataset_name = "clumpy"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/features/pixelization/simulator.py"],
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
__Sparse Operators__

Pixelized modeling requires dense linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **sparse linear algebra formalism**.

You do not need to understand the full details of the method, but the key point is:

- It exploits the **sparsity** of the matrices used in pixelized galaxy reconstruction.
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
settings = ag.Settings(use_positive_only_solver=False)

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used,
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

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

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this
example we fit a single galaxy with two components:

 - The galaxy's smooth central **bulge** is fit with a linear elliptical `Sersic` light profile [6 parameters:
   centre, ell_comps, effective_radius, sersic_index]. Using `lp_linear.Sersic` rather than `lp.Sersic` means the
   bulge's `intensity` is solved for via the same linear inversion that solves for the pixelization
   reconstruction, removing one non-linear parameter and avoiding the bulge / pixelization brightness degeneracy.

 - The galaxy's asymmetric **clumpy star formation** is reconstructed with a 28 x 28 `RectangularUniform` mesh
   [0 parameters], regularized with a `GaussianKernel` scheme that smooths the reconstruction [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.

It is worth noting how parsimonious the pixelization is. The clumpy component costs just the 2 regularization
parameters, however irregular it turns out to be, whereas describing the same clumps with light profiles or an
MGE would mean adding profile after profile (a 20+ parameter model) and would still struggle with truly
asymmetric substructure.

The model therefore includes a light profile, plus a mesh and regularization scheme which are used together to
create the pixelization.
"""
# Galaxy:
bulge = af.Model(ag.lp_linear.Sersic)

mesh = af.Model(ag.mesh.RectangularUniform, shape=mesh_shape)
regularization = af.Model(ag.reg.GaussianKernel)

pixelization = af.Model(ag.Pixelization, mesh=mesh, regularization=regularization)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, pixelization=pixelization)

# Overall Model:
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the galaxy has a linear `Sersic` bulge, plus a mesh and regularization scheme which are
combined into a pixelization.
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
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how via Nautilus the model is fitted to the data.
"""
analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    settings=settings,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM
required by a model.

Any model fitted through an inversion — a pixelization, linear light profiles, or both together (as in this
example) — uses a lot less VRAM once the sparse operator formalism is applied (as it is above). In this mode,
datasets with tens of millions of visibilities and real space masks with pixel scales below 0.05" can be stored
in just GB's of VRAM, which is remarkable given how much data they contain.

In sparse operator mode, the **amount of VRAM used is independent of the number of visibilities in the dataset**.
This is because the sparse operator method compresses all the visibility information into sparse operator matrices,
whose size depends only on the number of pixels in the real-space mask. VRAM use is therefore mostly driven by
how many pixels are in the real space mask.

VRAM does scale with batch size though, and for high resoluiton datasets may require you to reduce from the value of
20 set above if your GPU does not have too much VRAM (e.g. < 4GB).

The method below prints the VRAM usage estimate for the analysis and model with the specified batch size,
it takes about 20-30 seconds to run so you may want to comment it out once you are familiar with your GPU's VRAM limits.
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

Assuming the use of a 28 x 28 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. Adding the linear `Sersic` bulge to the model does not change this materially: it
contributes one more column to the same linear solve, and its 6 non-linear parameters are cheap compared to the
likelihood evaluation itself. Increasing the batch size will speed up the fit, provided VRAM allows it.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this
does not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms that the galaxy has a linear `Sersic` bulge, plus a mesh and regularization scheme which are
combined into a pixelization.
"""
print(result.info)

"""
We plot the maximum likelihood fit, galaxy images and posteriors inferred via Nautilus.

The reconstructed bulge image and the pixelized reconstruction of the clumps should together reproduce the data
to roughly the noise level, with the bulge absorbing the smooth central light and the pixelization absorbing the
off-centre clumpy structure.

The end of this example provides a detailed description of all result options for a pixelization.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_fit_dirty_images(fit=result.max_log_likelihood_fit)

aplt.subplot_galaxies(galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp)


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
pixelization. The combination of a linear parametric bulge with a pixelization for the irregular component — as in
the model fitted above — is the canonical **PyAutoGalaxy** approach.

__Chaining__

Modeling using a pixelization can be more efficient, robust and automated using the non-linear chaining feature to
compose a pipeline which begins by fitting a simpler model using a parametric galaxy.

More information on chaining is provided in the `autogalaxy_workspace/notebooks/guides/modeling/chaining` folder,
the end of chapter 2 of the **HowToGalaxy** lectures (tutorials 9-10).

__HowToGalaxy__

A full description of how pixelizations work, which comes down to a lot of linear algebra, Bayesian statistics and
2D geometry, is provided in chapter 3 of the **HowToGalaxy** lectures.

__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More diagnostic calculations.
- Gradient calculations.
- A calculation which shows differential effects across the reconstruction.
"""
