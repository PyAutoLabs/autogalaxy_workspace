"""
__Log Likelihood Function: Pixelization__

This script provides a step-by-step guide of the **PyAutoGalaxy** `log_likelihood_function` which is used to fit
`Imaging` data with a pixelization (specifically a `RectangularAdaptDensity` mesh and `Constant` regularization scheme`).

This example combines does not include a light profile or linear light profiles, which can be combined with a
pixelization to fit an image. The inclusion of these components is described in the notebook
`log_likelihood_function/imaging/pixelization/with_light_profile.ipynb`.

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoGalaxy**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make pixelized reconstructions less of a "black-box" to users.

__Prerequisites__

The likelihood function of a pixelization builds on that used for standard parametric light profiles and
linear light profiles, therefore you must read the following notebooks before this script:

- `light_profile/likelihood_function.ipynb`.
- `linear_light_profile/likelihood_function.ipynb`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__JAX & Preloads__

The `autogalaxy_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.
"""
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = ag.Preloads(
    mapper_indices=ag.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=ag.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__Dataset__

In order to perform a likelihood evaluation, we first load a dataset.

This example fits a simulated galaxy which is simulated using a 0.1 arcsecond-per-pixel resolution (this is lower
resolution than the best quality Hubble Space Telescope imaging and close to that of the Euclid space satellite).
"""
dataset_path = Path("dataset", "imaging", "simple")

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
I will use in-built visualization tools for plotting. 

For example, using the `ImagingPlotter` I can plot the imaging dataset we performed a likelihood evaluation on.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
lens modeling.

Below, we define a 2D circular mask with a 3.0" radius.
"""
mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`. 

a full description of over sampling and how to use it is given in `autogalaxy_workspace/*/guides/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1, over_sample_size_pixelization=1
)

"""
__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

For light profiles these are given by `masked_dataset.lp`, which is a uniform grid of (y,x) Cartesian coordinates
which have had the 3.0" circular mask applied.

A pixelization uses a separate grid of (y,x) coordinates, called `masked_dataset.grids.pixelization`, which is
identical to the light profile grid but may of had a different over-sampling scale applied (but in this example
does not).

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to construct a pixelization there is a straight forward mapping between the image data and pixelization pixels.
"""
grid_plotter = aplt.Grid2DPlotter(grid=masked_dataset.grids.pixelization)
grid_plotter.figure_2d()

"""
__Galaxy__

We combine the pixelization into a single `Galaxy` object.

The galaxy includes the rectangular mesh and constant regularization scheme, which will ultimately be used
to reconstruct its star forming clumps.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularAdaptDensity(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

"""
__Source Galaxy Pixelization and Regularization__

The source galaxy is reconstructed using a pixel-grid, in this example a RectangularUniform mesh, which accounts for 
irregularities and asymmetries in the source's surface brightness. 

A constant regularization scheme is applied which applies a smoothness prior on the reconstruction. 
"""
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)

mesh_grid = overlay_grid_from(
    shape_native=galaxy.pixelization.mesh.shape, grid=masked_dataset.grids.pixelization
)

"""
__Interpolation__

We now combine grids computed above to create an `Interpolator`, which describes how image grid pixel maps to
every rectangular mesh pixel. 
"""
interpolator = pixelization.mesh.interpolator_from(
    source_plane_data_grid=masked_dataset.grids.pixelization,
    source_plane_mesh_grid=mesh_grid,
)

"""
For the rectangular mesh, the interpolation scheme is called bilinear interpolation, which means that every image 
pixel maps to the rectangular pixel it lands in and the three neighboring rectangular pixels. 

The weight of each mapping is determined by the bilinear interpolation scheme, which is a function of how close the 
image pixel is to the centre of the rectangular pixel it lands in and the three neighboring rectangular pixels.

We can print the mappings and weights, for example of the first image pixel, to confirm this.
"""
print(interpolator.mappings[0])
print(interpolator.weights[0])

"""
__Mapper__

The rectangular mesh will now be referred to interchangeably as the `source-plane`, to represent that it is a 
pixelization which will reconstruct a source galaxy.

We now use the interpolator to create a `Mapper`, which describes the mapping between every image pixel and every 
rectangular pixel, based on the interpolation scheme above.
"""
mapper = ag.Mapper(interpolator=interpolator, preloads=preloads)

"""
Plotting the rectangular mesh shows that the source-plane has been discretized into a grid of rectangular pixels.

Below, we plot the rectangular mesh without the image-grid pixels (for clarity) and with them as black dots in order
to show how each set of image-pixels fall within a rectangular pixel.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.figure_2d()

visuals = aplt.Visuals2D(
    grid=mapper.source_plane_data_grid,
)
mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.figure_2d()

"""
The `Mapper` contains:

 1) `source_plane_data_grid`: the grid of masked (y,x) image-pixel coordinate centres (`masked_dataset.grids.pixelization`).
 2) `source_plane_mesh_grid`: The rectangular mesh of (y,x) pixelization pixel coordinates (`mesh_grid`).

We have therefore discretized the source-plane into a rectangular mesh, and can pair every image-pixel coordinate
with the corresponding rectangular pixel it lands in.

These quantities are both in the source-plane, and do not by themselves describe the mapping between the image and 
source planes. The mapping is described by the `pix_indexes_for_sub_slim_index`, which maps every image-pixel index to 
every pixelization pixel index.

`pix_indexes` refers to the pixelization pixel indexes (e.g. rectangular pixel 0, 1, 2 etc.) and `sub_slim_index`  
refers to the index of an image pixel (e.g. image-pixel 0, 1, 2 etc.). 

For example, printing the first ten entries of `pix_indexes_for_sub_slim_index` shows the first ten rectangular
pixelization pixel indexes these image sub-pixels map too.
"""
pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])

"""
This array can be used to visualize how an input list of image-pixel indexes map to the rectangular pixelization.

It also shows that image-pixel indexing begins from the top-left and goes rightwards and downwards, accounting for 
all image-pixels which are not masked.
"""
visuals = aplt.Visuals2D(indexes=[list(range(2050, 2090))])

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)
mapper_plotter.subplot_image_and_mapper(image=masked_dataset.data)

"""
The reverse mappings of pixelization pixels to image-pixels can also be used.
"""
pix_indexes = [[200]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(indexes=indexes)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=masked_dataset.data)

"""
__Interpolation__

The right hand plot shows more laying over source pixel 200 than its retangular black lines. Pixels further 
out than the pixel appear to be mapped to this source pixel. 

This is because of the interpolation mapping scheme whereby each image pixels is paired with four source 
pixels.

We can confirm that every image pixel maps to four source pixels by printing 
the `pix_sizes_for_sub_slim_index`, which gives the number of mapped source pixels for every image pixel.

We can also confirm that the interpolation introduces weights to each mapping by printing the 
`pix_weights_for_sub_slim_index`, which gives the weight of each mapping for every image pixel.
"""
print(mapper.pix_sizes_for_sub_slim_index[0:9])
print(mapper.pix_weights_for_sub_slim_index[0:9])

"""
__Over Sampling__

Lets quickly think about what happens when we use over sampling in the pixelization (e.g. `sub_size>1`). For
the `sub_size=1` case above, each image pixel maps to 4 source pixels (due to bilinear interpolation)
with a weight determined from the bilinear interpolation scheme.

However, the default over sampling for a pixelization is `sub_size=4`, meaning each image pixel is divided
into a 4x4 grid of sub-pixels (16 sub-pixels in total). Each of these sub-pixels maps to 4 source pixels
(due to bilinear interpolation), where the weight of each mapping is determined by the bilinear interpolation
scheme divided by 16 (because there are 16 sub-pixels).

This example therefore used a `sub_size=1` to keep the explanation of the likelihood, visualization of the
arrays above and understanding of the mapping scheme as simple as possible. You can manually increase the
`sub_size` above and re-run the notebook to see how this changes the mapping scheme.

__Alternative Meshes__

We can briefly consider how this step differs for other mesh types. Above, we simply overlaid a uniform rectangular
grid to define the source pixel centres and then mapped image pixels to these source pixels.

The `RectangularAdaptDensity` mesh pretty much works exactly the same, its just that a calculation (which we don't
describe here) works out how to make a grid of rectangular pixels that adapt to the source-plane density and thus
vary in size. 

There is also a `RectangularAdaptImage` mesh which uses the image of the galaxy to adapt
the rectangular pixel sizes. This often puts even smaller pixels in the brightest regions of the galaxy,
even if it lies offset or away from the caustic.

There is also a `Delaunay` mesh which uses a Delaunay triangulation to define an irregular grid of source pixels.
This is described fully in the `delaunay` example including a likelihood function guide.

__Mapping Matrix__

The `mapping_matrix` represents the image-pixel to source-pixel mappings above in a 2D matrix. 

It has dimensions `(total_image_pixels, total_rectangular_pixels)`.
"""
mapping_matrix = ag.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.over_sampler.sub_fraction,
)

"""
A 2D plot of the `mapping_matrix` shows of all image-pixelization pixel mappings.

No row of pixels has more than one non-zero entry. It is not possible for two image pixels to map to the same 
pixelization pixel (meaning that there are no correlated pixels in the mapping matrix).
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
Each column of the `mapping_matrix` can therefore be used to show all image-pixels it maps too. 

For example, above, we plotted all image-pixels of pixelization pixel 200 (as well as 202 and 204). We can extract all
image-pixel indexes of pixelization pixels 200 using the `mapping_matrix` and use them to plot the image of this
pixelization pixel (which corresponds to only values of zeros or ones).
"""
indexes_pix_200 = np.nonzero(mapping_matrix[:, 200])

print(indexes_pix_200[0])

array_2d = ag.Array2D(values=mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
__Blurred Mapping Matrix ($f$)__

Each pixelization pixel can therefore be thought of as an image (where all entries of this image are zeros and ones). 

To incorporate the imaging data's PSF, we simply blur each one of these pixelization pixel images with the imaging 
data's Point Spread Function (PSF) via 2D convolution.

This operation does not change the dimensions of the mapping matrix, meaning the `blurred_mapping_matrix` also has
dimensions `(total_image_pixels, total_rectangular_pixels)`. It turns the values of zeros and ones into 
non-integer values which have been blurred by the PSF.
"""
blurred_mapping_matrix = masked_dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=masked_dataset.mask
)

"""
A 2D plot of the `blurred_mapping_matrix` shows all image-source pixel mappings including PSF blurring.

Note how, unlike for the `mapping_matrix`, every row of image-pixels now has multiple non-zero entries. It is now 
possible for two image pixels to map to the same source pixel, because they become correlated by PSF convolution.
"""
plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

"""
Each column of the `blurred_mapping_matrix` shows all image-pixels it maps to after PSF blurring. 
"""
indexes_pix_200 = np.nonzero(blurred_mapping_matrix[:, 200])

print(indexes_pix_200[0])

array_2d = ag.Array2D(values=blurred_mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
In Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) the `blurred_mapping_matrix` is denoted $f_{ij}$
where $i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 

For example: 

 - $f_{0, 2} = 0.3$ indicates that image-pixel $2$ maps to pixelization pixel $0$ with a weight of $0.3$ after PSF convolution.
 - $f_{4, 8} = 0$ indicates that image-pixel $8$ does not map to pixelization pixel $4$, even after PSF convolution.

The indexing of the `mapping_matrix` is reversed compared to the notation of WD03 (e.g. image pixels
are the first entry of `mapping_matrix` whereas for $f$ they are the second index).
"""
print(f"Mapping between image pixel 0 and rectangular pixel 2 = {mapping_matrix[0, 2]}")

"""
__Data Vector (D)__

To solve for the source pixel fluxes we now pose the problem as a linear inversion.

This requires us to convert the `blurred_mapping_matrix` and our `data` and `noise map` into matrices of certain dimensions. 

The `data_vector`, $D$, is the first matrix and it has dimensions `(total_source_pixels,)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) and N15 (https://arxiv.org/abs/1412.7436) the data vector 
is give by: 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j} - b_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are the image-pixel data flux values.
 - $b_{\rm j}$ are the brightness values of the lens light model (therefore $d_{\rm  j} - b_{\rm j}$ is the lens light
 subtracted image).
 - $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.

$i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 

NOTE: WD03 assume the data is already lens subtracted thus $b_{j}$ is omitted (e.g. all values are zero).
"""
data_vector = ag.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

"""
$D$ describes which deconvolved rectangular pixels trace to which image-plane pixels. This ensures the 
reconstruction fully accounts for the PSF when fitting the data.

We can plot $D$ as a column vector:
"""
plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
The dimensions of $D$ are the number of source pixels.
"""
print("Data Vector:")
print(data_vector)
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ is the second matrix and it has 
dimensions `(total_rectangular_pixels, total_rectangular_pixels)`.

In WD03 / N15 (https://arxiv.org/abs/astro-ph/0302587) the curvature matrix is a 2D matrix given by:

 ${F}_{ik} = \sum_{\rm  j=1}^{J}f_{ij}f_{kj}/\sigma_{j}^2 \, \, .$

NOTE: this notation implicitly assumes a summation over $K$, where $k$ runs over all pixelization pixel indexes $K$.

Note how summation over $J$ runs over $f$ twice, such that every entry of $F$ is the sum of the multiplication
between all values in every two columns of $f$.

For example, $F_{0,1}$ is the sum of every blurred image pixels values in $f$ of source pixel 0 multiplied by
every blurred image pixel value of source pixel 1.
"""
curvature_matrix = ag.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
For $F_{ik}$ to be non-zero, this requires that the images of rectangular pixels $i$ and $k$ share at least one
image-pixel, which we saw above is only possible due to PSF blurring.

For example, we can see a non-zero entry for $F_{100,101}$ and plotting their images
show overlap.
"""
source_pixel_0 = 0
source_pixel_1 = 1

print(curvature_matrix[source_pixel_0, source_pixel_1])

array_2d = ag.Array2D(
    values=blurred_mapping_matrix[:, source_pixel_0], mask=masked_dataset.mask
)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

array_2d = ag.Array2D(
    values=blurred_mapping_matrix[:, source_pixel_1], mask=masked_dataset.mask
)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
The following chi-squared is minimized when we perform the inversion and reconstruct the galaxy:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) - d_{j}}{\sigma_{j}} \bigg]$

Where $s$ is the reconstructed pixel fluxes in all $I$ rectangular pixels.

The solution for $s$ is therefore given by (equation 5 WD03):

 $s = F^{-1} D$

We can compute this using NumPy linear algebra:
"""

# Because we are no using regularizartion (see below) it is common for the curvature matrix to be singular and lead
# to a LinAlgException. The loop below mitigates this -- you can ignore it as it is not important for understanding
# the PyAutoGalaxy likelihood function.

for i in range(curvature_matrix.shape[0]):
    curvature_matrix[i, i] += 1e-8

reconstruction = np.linalg.solve(curvature_matrix, data_vector)

"""
We can plot this reconstruction -- it looks like a mess.

The pixelization pixels have noisy and unsmooth values, and it is hard to make out if a galaxy is even being 
reconstructed. 

In fact, the linear inversion is (over-)fitting noise in the image data, meaning this system of equations is 
ill-posed. We need to apply some form of smoothing on the reconstruction to avoid over fitting noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction)

"""
__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{\rm L}$ to the $\chi^2$ we solve for giving us a new merit 
function $G$ (equation 11 WD03):

 $G = \chi^2 + \lambda \, G_{\rm L}$

where $\lambda$ is the `regularization_coefficient` which describes the magnitude of smoothness that is applied. A 
higher $\lambda$ will regularize the source more, leading to a smoother galaxy reconstruction.

Different forms for $G_{\rm L}$ can be defined which regularize the reconstruction in different ways. The 
`Constant` regularization scheme used in this example applies gradient regularization (equation 14 WD03):

 $G_{\rm L} = \sum_{\rm  i}^{I} \sum_{\rm  n=1}^{N}  [s_{i} - s_{i, v}]$

This regularization scheme is easier to express in words -- the summation goes to each rectangular pixelization pixel,
determines all rectangular pixels with which it shares a direct vertex (e.g. its neighbors) and penalizes solutions 
where the difference in reconstructed flux of these two neighboring pixels is large.

The summation does this for all rectangular pixels, thus it favours solutions where neighboring rectangular 
pixels reconstruct similar values to one another (e.g. it favours a smooth galaxy reconstruction).

We now define the `regularization matrix`, $H$, which allows us to include this smoothing when we solve for $s$. $H$
has dimensions `(total_rectangular_pixels, total_rectangular_pixels)`.

This relates to $G_{\rm L}$ as (equation 13 WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial G_{\rm L}}{\partial s_{i} \partial s_{k}}$

$H$ has the `regularization_coefficient` $\lambda$ folded into it such $\lambda$'s control on the degree of smoothing
is accounted for.
"""
regularization_matrix = ag.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.neighbors,
    neighbors_sizes=mapper.neighbors.sizes,
)

"""
We can plot the regularization matrix and note that:

 - non-zero entries indicate that two rectangular pixelization pixels are neighbors and therefore are regularized 
 with one another.

 - Zeros indicate the two rectangular pixels do not neighbor one another.

The majority of entries are zero, because the majority of rectangular pixels are not neighbors with one another.
"""
plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__F + Lamdba H__

$H$ enters the linear algebra system we solve for as follows (WD03 equation (12)):

 $s = [F + H]^{-1} D$
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

"""
__Galaxy Reconstruction (s)__

We can now solve the linear system above using NumPy linear algebra. 

Note that the for loop used above to prevent a LinAlgException is no longer required.
"""
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
By plotting this galaxy reconstruction we can see that regularization has lead us to reconstruct a smoother galaxy,
which actually looks like the star forming clumps in the galaxy imaging data! 

This also implies we are not over-fitting the noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction)

"""
__Image Reconstruction__

Using the reconstructed pixel fluxes we can map the reconstruction back to the image plane (via 
the `blurred mapping_matrix`) and produce a reconstruction of the image data.
"""
mapped_reconstructed_operated_data = (
    ag.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_operated_data = ag.Array2D(
    values=mapped_reconstructed_operated_data, mask=mask
)

array_2d_plotter = aplt.Array2DPlotter(array=mapped_reconstructed_operated_data)
array_2d_plotter.figure_2d()

"""
__Likelihood Function__

We now quantify the goodness-of-fit of our pixelization galaxy reconstruction. 

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for galaxy modeling consists of five terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, \left[ \mathrm{det} (F + H) \right] - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right] + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

This expression was first derived by Suyu 2006 (https://arxiv.org/abs/astro-ph/0601493) and is given by equation (19).
It was derived into **PyAutoGalaxy** notation in Dye 2008 (https://arxiv.org/abs/0804.4002) equation (5).

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `mapped_reconstructed_operated_data`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_image = mapped_reconstructed_operated_data

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = ag.Array2D(values=chi_squared_map, mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=chi_squared_map)
array_2d_plotter.figure_2d()

"""
__Regularization Term__

The second term, $s^{T} H s$, corresponds to the $\lambda $G_{\rm L}$ regularization term we added to our merit 
function above.

This is the term which sums up the difference in flux of all reconstructed rectangular pixels, and reduces the 
likelihood of solutions where there are large differences in flux (e.g. the galaxy is less smooth and more likely to be 
overfitting noise).

We compute it below via matrix multiplication, noting that the `regularization_coefficient`, $\lambda$, is built into 
the `regularization_matrix` already.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

"""
__Complexity Terms__

Up to this point, it is unclear why we chose a value of `regularization_coefficient=1.0`. 

We cannot rely on the `chi_squared` and `regularization_term` above to optimally choose its value, because increasing 
the `regularization_coefficient` smooths the solution more and therefore:

 - Decreases `chi_squared` by fitting the data worse, producing a lower `log_likelihood`.

 - Increases the `regularization_term` by penalizing the differences between source pixel fluxes more, again reducing
 the inferred `log_likelihood`.

If we set the regularization coefficient based purely on these two terms, we would set a value of 0.0 and be back where
we started over-fitting noise!

The terms $\left[ \mathrm{det} (F + H) \right]$ and $ - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right]$ address 
this problem. 

They quantify how complex the reconstruction is, and penalize solutions where *it is more complex*. Reducing 
the `regularization_coefficient` makes the galaxy reconstruction more complex (because a galaxy that is 
smoothed less uses more flexibility to fit the data better).

These two terms therefore counteract the `chi_squared` and `regularization_term`, so as to attribute a higher
`log_likelihood` to solutions which fit the data with a more smoothed and less complex source (e.g. one with a higher 
`regularization_coefficient`).

In **HowToGalaxy** -> `chapter 4` -> `tutorial_4_bayesian_regularization` we expand on this further and give a more
detailed description of how these different terms impact the `log_likelihood_function`. 
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term ins the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the galaxy modeling process and has no impact on the 
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the galaxy model, by combining the five terms computed above using
the likelihood function defined above.
"""
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
__Fit__

This process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(
    dataset=masked_dataset,
    galaxies=galaxies,
    settings=ag.Settings(use_border_relocator=True),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Galaxy Modeling__

To fit a galaxy model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
multiple MCMC and optimization algorithms are supported.

__Log Likelihood Function: Pixelization With Light Profile__

The above example does not include a parametric light profile in the galaxy model, which the modeling example
shows may be useful if you want to fit both the smooth and clumpy components of a galaxy's light.

A step-by-step guide to the likelihood function is not provided, but in a nutshell one of the following two 
things happen:

- `standard light profile`: If normal light profiles are included in the galaxy model, their light is computed 
  on the image grid, subtracted from the data, and the pixelization is used to reconstruct the resulting light
  profile subtracted image. You can basically think of the pixelization as fitting the residuals left over after
  the light profile has been subtracted.

- `linear light profile`: If linear light profiles are included in the galaxy model, their light is computed
  on the image grid and linearly solved for simultaneously with the pixelization reconstruction. This means
  they trade off their solved for `intensity` values with the source pixel fluxes to fit the image data. The
  inversion is thus performed simultaneously for both the linear light profile intensities and source pixel fluxes!

__Wrap Up__

We have presented a visual step-by-step guide to the **PyAutoGalaxy** likelihood function, which uses a pixelization,
regularization scheme, and inversion to reconstruct a galaxy’s light.

There are a number of additional inputs and features which slightly change the behaviour of this likelihood function,
and which are described in further notebooks found in this package. In brief, these include:

 - **Sub-gridding**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually
   evaluated and paired fractionally with each pixel in the galaxy reconstruction.

 - **Reconstruction Interpolation**: Using a RectangularUniform mesh with bilinear interpolation to pair each 
 image (sub-)pixel to multiple reconstruction pixels with interpolation weights.

 - **Galaxy Morphology Pixelization Adaption**: Adapting the pixelization such that it congregates pixels around the
   brightest regions of the galaxy, rather than using a uniform pixelization.

 - **Luminosity-Weighted Regularization**: Using an adaptive regularization coefficient which adjusts the level of
   regularization applied based on the reconstructed galaxy luminosity.
"""
