"""
Features: Pixelization Fit
==========================

This script performs a single, direct fit (no non-linear search) of a galaxy that has two components:

 - A smooth central **bulge**, captured by a linear `Sersic` light profile.
 - **Asymmetric clumpy star formation**, reconstructed on a pixelization with a rectangular mesh and constant
   regularization scheme.

The `clumpy` imaging dataset used here is the canonical motivating example for pixelizations in galaxy modeling.
Parametric profiles fit the bulge well but leave structured residuals where the clumps live; the pixelization
absorbs whatever the Sersic cannot describe on a flexible pixel grid.

See `modeling.py` for the same model fit via a non-linear search, and `galaxy_reconstruction.py` for how to export
the reconstructed clumpy component to a CSV for downstream science.

Pixelizations are covered in detail in chapter 4 of the **HowToGalaxy** lectures.

__JAX GPU Run Times__

Pixelizations run time depends on how modern GPU hardware is. GPU acceleration only provides fast run times on
modern GPUs with large amounts of VRAM, or when the number of pixels in the mesh are low (e.g. < 500 pixels).

This script's default setup uses an adaptive 20 x 20 rectangular mesh (400 pixels), which is relatively low resolution
and may not provide the most accurate modeling results. On most GPU hardware it will run in ~ 10 minutes,
however if your laptop has a large VRAM (GPU > 20 GB) or you can access a GPU cluster with better hardware you should use these
to perform modeling with increased mesh resolution.

__CPU Run Times__

JAX is not natively designed to provide significant CPU speed up, therefore users using CPUs to perform pixelization
analysis will not see fast run times using JAX (unlike GPUs).

The example `pixelization/cpu_fast_modeling` shows how to set up a pixelization to use efficient CPU calculations
via the library `numba`.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of using a pixelization to reconstruct a galaxy.
- **Positive Only Solver:** How a positive solution to the reconstructed pixel fluxes is ensured.
- **Dataset & Mask:** Standard set up of the clumpy imaging dataset that is fitted.
- **Mesh Shape**: Defining the shape of the mesh that reconstructs the clumpy component in advance, such that JAX knows static array shapes.
- **Pixelization:** How to create a pixelization, including a description of its inputs.
- **Fit:** Perform a fit to the dataset combining a `Sersic` bulge with a pixelization for the clumps.
- **Mask Extra Galaxies:** Using noise scaling to handle nearby galaxies whose emission would otherwise contaminate the fit.
- **Linear Objects / Grids / Reconstruction:** Inspecting the inversion output for the reconstructed clumpy component.

__Advantages__

Many galaxies are complex, and have asymmetric and irregular morphologies. These morphologies cannot be well
approximated by light profiles like a Sersic, or many Sersics, and thus a pixelization is required to reconstruct
the irregular component of the galaxy's light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a galaxy accurately
if its surface brightness has highly localised, asymmetric substructure.

With a pixelization, we can specifically estimate how much light is in the irregular components of a galaxy
(e.g. spiral arms, star forming clumps) compared to its smooth components (e.g. bulge, disk), by pairing a
parametric profile for the bulge with a pixelization for the rest.

__Disadvantages__

Pixelizations are computationally slow and run times are typically longer than a purely parametric galaxy model.
It is not uncommon for models using a pixelization to take hours to fit high resolution imaging data
(e.g. Hubble Space Telescope imaging), albeit on modern GPUs run times are often closer to < 20 minutes.

It will take you longer to learn how to successfully fit galaxy models with a pixelization than other methods
illustrated in the workspace!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelization, this often produces negative reconstruction pixels which
over-fit the data, producing unphysical solutions.

All pixelized reconstructions use a positive-only solver, meaning that every pixel is only allowed
to reconstruct positive flux values. This ensures that the reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real galaxy (a common systematic solution in this
analysis).

Enforcing positive reconstructions efficiently requires non-trivial linear algebra, so a bespoke JAX fast non-negative
solver was developed; many methods in the literature omit this and therefore allow unphysical negative solutions that
can degrade modeling results.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt
from autoarray.inversion.plot.inversion_plots import subplot_of_mapper

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

A pixelization uses a separate grid for light evaluation, with its own over sampling scheme, which below we set to a 
uniform grid of values of 4. 

Note that the over sampling is input into the `over_sample_size_pixelization` because we are using a `Pixelization`.
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mesh Shape__

The `mesh_shape` parameter defines the number of pixels used by the rectangular mesh to reconstruct the clumpy
component of the galaxy, set below to 30 x 30.

The `mesh_shape` must be fixed before modeling and cannot be a free parameter of the model, because JAX uses the
mesh shape to define static shaped arrays which use the mesh to reconstruct the galaxy. For a rectangular
mesh, the same number of pixels must be used in the y and x directions.
"""
mesh_pixels_yx = 30
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Pixelization__

We create a `Pixelization` object to perform the pixelized galaxy reconstruction, which is made up of two
components:

- `mesh:` Different types of mesh can be used to perform the reconstruction, where the mesh changes the
details of how the galaxy is reconstructed (e.g. interpolation weights). In this example, we use a rectangular mesh.

- `regularization:` A pixelization uses many pixels for the reconstruction, which will often lead to over-fitting
of the noise in the data and an unrealistically complex and structured solution. Regularization smooths the
reconstruction by penalizing solutions where neighboring pixels have large flux differences.
"""
mesh = ag.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = ag.reg.Constant(coefficient=1.0)

pixelization = ag.Pixelization(mesh=mesh, regularization=regularization)

"""
__Fit__

This illustrates the API for performing a single (no non-linear search) fit using a parametric bulge plus a
pixelization for the clumpy component, via standard PyAutoGalaxy objects: `Galaxy`, `Galaxies`, and `FitImaging`.

We create a `Galaxy` with both a parametric `Sersic` bulge and the `Pixelization` defined above. We use a
standard `lp.Sersic` here (rather than `lp_linear.Sersic` as in `modeling.py`) and hand-tune all of its parameters
so that the linear `Inversion` only contains the pixelization `Mapper` — keeping the rest of this script (which
walks through the mapper internals) easy to read. In `modeling.py` the bulge `intensity` is solved for via the
inversion alongside the pixelization reconstruction.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.5,
    effective_radius=0.6,
    sersic_index=2.5,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, pixelization=pixelization)

galaxies = ag.Galaxies([galaxy])

fit = ag.FitImaging(
    dataset=dataset,
    galaxies=galaxies,
)

"""
By plotting the fit, we see that the bulge captures the smooth central component while the pixelization absorbs
the clumpy off-centre light, fitting the data to roughly the noise level.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
Pixelizations have bespoke visualizations which show more details about the reconstruction, image-mesh
and other quantities.

The `subplot_of_mapper` function produces a comprehensive diagnostic subplot for the inversion.
"""
inversion = fit.inversion

subplot_of_mapper(inversion=inversion, mapper_index=0)

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
We do not explicitly fit this data, for the sake of brevity, however if your data has these nearby galaxies you
should apply the mask as above before fitting the data.

__Wrap Up__

Pixelizations are the most complex but also most powerful way to reconstruct a galaxy's light.

Whether you need to use them or not depends on the science you are doing. If you are only interested in fitting
the smooth and symmetric components of a galaxy's light (e.g. bulge, disk) then using parametric light profiles
is likely a better approach, as they are fast and accurate for this purpose.

However, fitting complex structures in galaxies (e.g. spiral arms, clumps) requires a pixelization, as parametric
light profiles cannot easily capture these features. The combination of a parametric bulge with a pixelization for
the irregular component — as in the model fit above — is the canonical PyAutoGalaxy approach.

__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`.

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear object used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper`
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectangularMapper` is used for a `RectangularAdaptDensity` mesh).

In this example, the bulge is a non-linear `lp.Sersic` so it does not enter the `Inversion`. The only linear object
used to fit the data is therefore the pixelization `Mapper`, and `linear_obj_list` contains just one entry:
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities come in lists or require that we specify the linear object
we wish to use.

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"Mapper = {inversion.linear_obj_list[0]}")

"""
__Grids__

The role of a mapper is to map between the image-plane data grid and the pixelization mesh used to reconstruct
the galaxy.

This includes mapping grids corresponding to the data grid (e.g. the centres of each image-pixel) and the
pixelization grid (e.g. the centre of each mesh pixel). For a galaxy reconstruction the `image-plane data grid` is
where the data lives and the `source-plane` naming used by the library API refers to the plane in which the
pixelization mesh is laid out — for `PyAutoGalaxy` (where there is no lensing) this is just the same image-plane
grid the data sits on.

All grids are available in a mapper via its `mapper` property.
"""
mapper = inversion.linear_obj_list[0]

# Centre of each masked image pixel in the image-plane.
print(mapper.image_plane_data_grid)

# Centre of each image pixel in the pixelization's mesh frame (identical to the image-plane grid in PyAutoGalaxy).
print(mapper.source_plane_data_grid)

# Centre of each pixelization pixel in the image-plane.
print(mapper.image_plane_mesh_grid)

# Centre of each pixelization pixel in the mesh frame.
print(mapper.source_plane_mesh_grid)

"""
__Reconstruction__

The reconstruction is also available as a 1D numpy array of values representative of the pixelization
itself (in this example, the reconstructed clump flux at each rectangular pixel).
"""
print(inversion.reconstruction)

"""
The (y,x) grid of coordinates associated with these values is given by the `Inversion`'s `Mapper` (which are
described in chapter 4 of **HowToGalaxy**).

Note above how we showed that the first entry of the `linear_obj_list` contains the inversion's `Mapper`.
"""
mapper = inversion.linear_obj_list[0]
print(mapper.source_plane_mesh_grid)

"""
The mapper also contains the (y,x) grid of coordinates that correspond to the imaging data's grid.
"""
print(mapper.source_plane_data_grid)

"""
__Mapped Reconstructed Images__

The reconstruction is mapped to the image grid in order to fit the model.

This mapped reconstructed image is also accessible via the `Inversion`.

Note that any parametric light profiles in the model (e.g. the `bulge` of the galaxy fitted above) are not
included in this image — it only contains the pixelized reconstruction of the clumpy component.
"""
print(inversion.mapped_reconstructed_operated_data.native)

"""
__Linear Algebra Matrices (Advanced)__

To perform an `Inversion` a number of matrices are constructed which use linear algebra to perform the reconstruction.

These are accessible in the inversion object.
"""
print(inversion.curvature_matrix)
print(inversion.regularization_matrix)
print(inversion.curvature_reg_matrix)

"""
__Evidence Terms (Advanced)__

In **HowToGalaxy** and the papers below, we cover how an `Inversion` uses a Bayesian evidence to quantify the goodness
of fit:

https://arxiv.org/abs/1708.07377
https://arxiv.org/abs/astro-ph/0601493

This evidence balances solutions which fit the data accurately, without using an overly complex regularized
reconstruction.

The individual terms of the evidence and accessed via the following properties:
"""
print(inversion.regularization_term)
print(inversion.log_det_regularization_matrix_term)
print(inversion.log_det_curvature_reg_matrix_term)

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More diagnostic quantities for the reconstructed clumpy light.
- Gradient calculations of the reconstructed light distribution.
"""
