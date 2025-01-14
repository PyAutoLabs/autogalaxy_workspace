"""
__Log Likelihood Function: W Tilde__

This script describes how a pixelization can be computed using a different linear algebra calculation, but
one which produces an identical likelihood at the end.

This is called the `w_tilde` formalism, and for interferometer datasets it avoids storing the `operated_mapping_matrix`
in memory, meaning that in the regime of 1e6 or more visibilities this extremely large matrix does not need to be
stored in memory.

This can make the likelihood function significantly faster, for example with speed ups of hundreds of times or more
for tens or millions of visibilities. In fact, the run time does not scale with the number of visibilities at all,
meaning datasets of any size can be fitted in seconds.

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
real_space_mask = ag.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

"""
__W Tilde__

We now compute the `w_tilde` matrix.

The `w_tilde` matrix is applied to the `curvature_matrix`, and allows us to efficiently compute the curvature matrix
without computing the `transformed_mapping_matrix` matrix. 

The functions used to do this has been copy and pasted from the `inversion` module of PyAutoArray source code below,
so you can see the calculation in full detail.

REMINDER: for the `real_space_mask` above with shape (800, 800) the `w_tilde` matrix will TAKE A LONG
TIME TO COMPUTE.
"""
from autoarray import numba_util


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
We now compute the `w_tilde` matrices.
"""
w_tilde = w_tilde_curvature_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
)

"""
__Mapping Matrix__

The `w_tilde` matrix is applied directly to the `mapping_matrix` to compute the `curvature_matrix`.

Below, we perform the likelihood function steps described in the `pixelization/log_likelihood_function.py` script,
to create the `mapping_matrix` we will apply the `w_tilde` matrix to.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

grid_rectangular = ag.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

mapper_grids = ag.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
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

"""
__Curvature Matrix__

We can now compute the `curvature_matrix` using the `w_tilde` matrix and `mapping_matrix`, which amazingly uses
simple matrix multiplication.
"""


def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_curvature_preload_imaging_from` should be used
    instead.

    Parameters
    ----------
    w_tilde
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    return np.dot(mapping_matrix.T, np.dot(w_tilde, mapping_matrix))


curvature_matrix = curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde, mapping_matrix=mapping_matrix
)

"""
If you compare the `curvature_matrix` computed using the `w_tilde` matrix to the `curvature_matrix` computed using the
`operated_mapping_matrix` matrix in the other example scripts, you'll see they are identical.

__Data Vector__

The `data_vector` was computed in the `pixelization/log_likelihood_function.py` script using 
the `transformed_mapping_matrix`.

Fortunately, there is also an easy way to compute the `data_vector` which bypasses the need to compute the
`transformed_mapping_matrix` matrix, again using simple matrix multiplication.
"""
data_vector = np.dot(mapping_matrix.T, dataset.w_tilde.dirty_image)

"""
__Reconstruction__

The `reconstruction` is computed using the `curvature_matrix` and `data_vector` as per usual.
"""
regularization_matrix = ag.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
__Likelihood Step: Visibilities Reconstruction__

The mapped reconstructed visibilities were another quantity computed in the `pixelization/log_likelihood_function.py`
which used the `transformed_mapping_matrix` matrix.

This is again a step which can be performed without the need to compute the `transformed_mapping_matrix` matrix.

In the example below, this is computed by doing the following:

1) Compute the real-space image of the source galaxy, which is computed by the pixelization.
2) Perform an NUFFT on this image to compute the visibilities.

This NUFFT can actually be slow for large numbers of visibilities (e.g. > 1e7) and will become a bottleneck in the
likelihood function. However, this is another problem that can be solved using the "fast chi-squared" approach,
which is described in the `log_likelihood_function/intereferometer/pixelization/fast_chi_squared.py` script.
"""
mapped_reconstructed_data_dict = {}

mapped_reconstructed_image = (
    ag.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = dataset.transformer.visibilities_from(
    image=mapped_reconstructed_image
)

visibilities = ag.Visibilities(visibilities=mapped_reconstructed_visibilities)

"""
__Log Likelihood__

Finally, we verify that the log likelihood computed using the `curvature_matrix` and `data_vector` computed using the
`w_tilde` matrix is identical to the log likelihood computed using the `operated_mapping_matrix` matrix in the
other example scripts.
"""
model_visibilities = mapped_reconstructed_visibilities

residual_map = dataset.data - model_visibilities

chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2
chi_squared_map = chi_squared_map_real + 1j * chi_squared_map_imag


chi_squared_real = np.sum(chi_squared_map.real)
chi_squared_imag = np.sum(chi_squared_map.imag)
chi_squared = chi_squared_real + chi_squared_imag


regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]


noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
Finish.
"""
