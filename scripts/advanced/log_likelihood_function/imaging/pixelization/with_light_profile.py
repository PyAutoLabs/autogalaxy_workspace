"""
__Log Likelihood Function: Pixelization With Light Profile__

This script describes how a pixelization, with light profiles or linear light profiles included, changes the likelihood
calculation of a pixelization.

It directly follows on from the `pixelization/log_likelihood_function.py` notebook and you should read through that
script before reading this script.

__Prerequisites__

You must read through the following likelihood functions first:

 - `pixelization/log_likelihood_function.py` the likelihood function for a pixelization.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Following the `pixelization/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
dataset_path = path.join("dataset", "imaging", "simple")

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

masked_dataset = dataset.apply_mask(mask=mask)

masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1, over_sample_size_pixelization=1
)

"""
__With Standard Light Profiles__

Combining standard light profiles with a pixelization is simple, and basically changes the pixelization likelihood
function as follows:

1) The image of the light profiles are computed, summed and convolved with the PSF using the standard methods.

2) This image is subtracted from the observed image to create a light subtracted image, which is the image that
   enters the pixelization linear inversion calculation.

3) The light profile images are addded back on to the pixelization's reconstructed image to create the model image
   that is compared to the observed image.

The code below repeats the `pixelization/log_likelihood_function.py` example, but with the addition of the light 
profiles.

Text is only included for steps which differ from the example in `pixelization/log_likelihood_function.py`.

__Galaxy__

The light profiles are created and combined with the `Pixelization` to create a `Galaxy` object.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.2,
    effective_radius=0.8,
    sersic_index=4.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=0.1,
    effective_radius=1.6,
)

pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)


"""
__Light Subtracted Image__

The image of the light profiles is computed, convolved with the PSF and subtracted from the observed image
to create the light subtracted image which will be input to the pixelization linear inversion.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk, pixelization=pixelization)

image = galaxy.image_2d_from(grid=masked_dataset.grid)
blurring_image_2d = galaxy.image_2d_from(grid=masked_dataset.grids.blurring)

convolved_image_2d = masked_dataset.convolver.convolve_image(
    image=image, blurring_image=blurring_image_2d
)

light_subtracted_image_2d = masked_dataset.data - convolved_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=light_subtracted_image_2d)
array_2d_plotter.figure_2d()

"""
__Mapping Matrix__

Steps creating the `mapping_matrix` and blurred_mapping_matrix` are identical to the previous example.
"""
grid_rectangular = ag.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=masked_dataset.grids.pixelization
)

mapper_grids = ag.MapperGrids(
    mask=mask,
    source_plane_data_grid=masked_dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = ag.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapping_matrix = ag.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

blurred_mapping_matrix = masked_dataset.convolver.convolve_mapping_matrix(
    mapping_matrix=mapping_matrix
)

"""
__Data Vector (D)__

The step which creates the `data_vector` is updated, as it now receives the light subtracted image as input.

The `data_vector`, $D$, is now defined algebraically as 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j} - b_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are again the image-pixel data flux values and $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.
 - $b_{\rm j}$ is a new quantity, they are the brightness values of the bulge and disk light model (therefore $d_{\rm  j} - b_{\rm j}$ is 
 the bulge and disk light subtracted image).

$i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 
"""
data_vector = ag.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(light_subtracted_image_2d),
    noise_map=np.array(masked_dataset.noise_map),
)

"""
__Linear Alegbra__

Steps creating the `curvature_matrix` and other quantities are identical.
"""
curvature_matrix = ag.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

regularization_matrix = ag.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

mapped_reconstructed_image_2d = (
    ag.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_image_2d = ag.Array2D(
    values=mapped_reconstructed_image_2d, mask=mask
)

array_2d_plotter = aplt.Array2DPlotter(array=mapped_reconstructed_image_2d)
array_2d_plotter.figure_2d()

"""
__Model Image__

Whereas previously the model image was only the reconstructed image, it now includes the light profile image.

The following chi-squared is therefore now minimized when we perform the inversion and reconstruct the galaxy:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) + b_{j} - d_{j}}{\sigma_{j}} \bigg]$
"""

model_image = convolved_image_2d + mapped_reconstructed_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
__Likelihood Function__

The overall likelihood function is the same as before, except the model image now includes the light profiles.
"""
model_image = convolved_image_2d + mapped_reconstructed_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
__With Linear Light Profiles__

Code examples of how linear light profiles are combined with a pixelization are not provided in this script, as they
are not yet written.

Conceptually, the main difference between linear light profiles and standard light profiles is that the linear light
profiles enter the linear algebra calculations and are solved for simultaneously with the pixelization. A summary of
how this changes the calculation is provided below:

- The data fitted is now the original data and does not have the light profiles subtracted from it, as the linear
  algebra calculation now solves for the light profiles simultaneously with the pixelization.

- The `mapping_matrix`, which for a pixelization has shape `(total_image_pixels, total_pixelization_pixels)`, has 
  rows added to it for every linear light profile, meaning its 
  shape is `(total_image_pixels, total_pixelization_pixels + total_linear_light_profiles)`.
 
- Each light profile column of this `mapping_matrix` is the image of each linear light profile, meaning that their
 `intensity` values are solved for simultaneously with the pixelization's `flux` values.

- The use of the positive only solver for the reconstruction is more important, because linear light profiles and
  pixelizations can trade-off negative values between one another and produce unphysical solutions.

Other than the above change, the calculation is performed in an identical manner to the pixelization example.
"""
