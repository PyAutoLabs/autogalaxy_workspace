"""
Features: Pixelization Fit
==========================

A pixelization reconstructs a galaxy's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This can be used to explitcitly reconstruct complex and irregular components in galaxies (e.g. spiral arms, clumps)
which symmetric light profiles like a Sersic cannot easily fit.

This script fits a galaxy in a way which uses a pixelization to reconstruct the source's light. This uses a
rectangular mesh and constant regularization scheme are, which are the simplest forms of each, both providing
computationally fast and accurate solutions.

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

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**JAX & Preloads**: Preloading certain arrays for the pixelization's linear algebra, such that JAX knows their shapes in advance.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Fit:** Perform a fit to a dataset using a pixelization, and visualize its results.
**Interpolated Source:** Interpolate the source reconstruction from an irregular Voronoi mesh to a uniform square grid and output to a .fits file.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a dataset with the inferred pixelized source.

__Advantages__

Many galaxies are complex, and have asymmetric and irregular morphologies. These morphologies cannot be well
approximated by a light profiles like a Sersic, or many Sersics, and thus a pixelization is required to reconstruct
the source's irregular light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a galaxy accurately
if there are multiple sources of light, or if the galaxy has a very complex morphology.

With a pixeliation, we can specficially estimate how much light is in irregular components of a galaxy (e.g. spiral
arms, star forming clumps) compared to its smooth components (e.g. bulge, disk).

__Disadvantages__

Pixelizations are computationally slow and run times are typically longer than a parametric source model. It is not
uncommon for models using a pixelization to take hours to fit high resolution imaging data (e.g. Hubble Space
Telescope imaging), albeit on modern GPUs run times are often closer to < 20 minutes.

It will take you longer to learn how to successfully fit lens models with a pixelization than other methods illustrated
in the workspace!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

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

Load and plot the strong lens dataset `simple__sersic` via .fits files.
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
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

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
__Pixelization__

We create a `Pixelization` object to perform the pixelized source reconstruction, which is made up of three
components:

- `mesh:` Different types of mesh can be used to perform the source reconstruction, where the mesh changes the
details of how the source is reconstructed (e.g. interpolation weights). In this example, we use a rectangular mesh.

- `regularization:` A pixelization uses many pixels to reconstructed the source, which will often lead to over fitting
of the noise in the data and an unrealistically complex and structured source. Regularization smooths the source
reconstruction solution by penalizing solutions where neighboring pixels have
large flux differences.
"""
mesh = ag.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = ag.reg.Constant(coefficient=1.0)

pixelization = ag.Pixelization(mesh=mesh, regularization=regularization)

"""
__Fit__

This is to illustrate the API for performing a fit via a pixelization using standard autolens objects like 
the `Galaxy`, `Tracer` and `FitImaging` 

We simply create a `Pixelization` and pass it to the source galaxy, which then gets input into the tracer.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularAdaptDensity(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

galaxies = ag.Galaxies([galaxy])

fit = ag.FitImaging(
    dataset=dataset,
    galaxies=galaxies,
    preloads=preloads,
)

"""
By plotting the fit, we see that the pixelized source does a good job at capturing the appearance of the galaxy
and fitting the data to roughly the noise level.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Pixelizations have bespoke visualizations which show more details about the reconstruction, image-mesh
and other quantities.

These plots use an `InversionPlotter`, which gets its name from the internals of how pixelizations are performed in
the source code, where the linear algebra process which computes the source pixel fluxes is called an inversion.
"""
inversion_plotter = fit_plotter.inversion_plotter
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
The inversion can be extracted directly from the fit the perform these plots, which we also use below
for various calculations
"""
inversion = fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

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

__Wrap Up__

Pixelizations are the most complex but also most powerful way to model a galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in fitting
the smooth and symmetric components of a galaxy's light (e.g. bulge, disk) then using parametric light profiles
is likely a better approach, as they are fast and accurate for this purpose.

However, fitting complex structures in galaxies (e.g. spiral arms, clumps) requires a pixelization, as parametric 
light profiles cannot easily capture these features.

__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `RectangularAdaptDensity` mesh).

In this example, the only linear object used to fit the data was a `Pixelization`, thus the `linear_obj_list`
contains just one entry corresponding to a `Mapper`:
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"Mapper = {inversion.linear_obj_list[0]}")

"""
__Grids__

The role of a mapper is to map between the image-plane and source-plane. 

This includes mapping grids corresponding to the data grid (e.g. the centers of each image-pixel in the image and
source plane) and the pixelization grid (e.g. the centre of the Delaunay triangulation in the image-plane and 
source-plane).

All grids are available in a mapper via its `mapper_grids` property.
"""
mapper = inversion.linear_obj_list[0]

# Centre of each masked image pixel in the image-plane.
print(mapper.mapper_grids.image_plane_data_grid)

# Centre of each source pixel in the source-plane.
print(mapper.mapper_grids.source_plane_data_grid)

# Centre of each pixelization pixel in the image-plane (the `Overlay` image_mesh computes these in the image-plane
# and maps to the source-plane).
print(mapper.mapper_grids.image_plane_mesh_grid)

# Centre of each pixelization pixel in the source-plane.
print(mapper.mapper_grids.source_plane_mesh_grid)

"""
__Reconstruction__

The reconstruction is also available as a 1D numpy array of values representative of the pixelization
itself (in this example, the reconstructed source values at each rectangular pixel).
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
The mapper also contains the (y,x) grid of coordinates that correspond to the imaging data's grid
"""
print(mapper.source_plane_data_grid)

"""
__Mapped Reconstructed Images__

The reconstruction(s) are mapped to the image grid in order to fit the model.

These mapped reconstructed images are also accessible via the `Inversion`. 

Note that any parametric light profiles in the model (e.g. the `bulge` and `disk` of a galaxy) are not 
included in this image -- it only contains the source.
"""
print(inversion.mapped_reconstructed_image.native)

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

This evidence balances solutions which fit the data accurately, without using an overly complex regularization source.

The individual terms of the evidence and accessed via the following properties:
"""
print(inversion.regularization_term)
print(inversion.log_det_regularization_matrix_term)
print(inversion.log_det_curvature_reg_matrix_term)

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More 
- Source gradient calculations.
"""
