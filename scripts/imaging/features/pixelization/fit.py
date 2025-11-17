"""
Modeling: Light Inversion
=========================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is modeled using an `Inversion` with a rectangular pixelization and constant regularization
 scheme.

An `Inversion` reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. Due to the simplicity of this example the inversion effectively just find a model
galaxy image that is denoised and deconvolved.

More complicated and useful inversion fits are given elsewhere in the workspace (e.g. the `chaining` package), where
they are combined with light profiles to fit irregular galaxies in a efficient way.

Inversions are covered in detail in chapter 4 of the **HowToGalaxy** lectures.

__Positive Only Solver__

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these
unphysical solutions, which can degrade the results of lens model in generag.

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

Load and plot the galaxy dataset `complex` via .fits files, where:
 
  -The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - The galaxy's has four star forming clumps which are `Sersic` profiles.
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

Apply adaptive over sampling to ensure the calculation is accurate, you can read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.

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

The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
versions.
"""
image_mesh = None
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = ag.Preloads(
    mapper_indices=ag.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__Pixelization__

We create a `Pixelization` object to perform the pixelized source reconstruction, which is made up of three
components:

- `image_mesh:`The coordinates of the mesh used for the pixelization need to be defined. The way this is performed
depends on pixelization used. In this example, we define the source pixel centers by overlaying a uniform regular grid
in the image-plane and ray-tracing these coordinates to the source-plane. Where they land then make up the coordinates
used by the mesh.

- `mesh:` Different types of mesh can be used to perform the source reconstruction, where the mesh changes the
details of how the source is reconstructed (e.g. interpolation weights). In this exmaple, we use a `Voronoi` mesh,
where the centres computed via the `image_mesh` are the vertexes of every `Voronoi` triangle.

- `regularization:` A pixelization uses many pixels to reconstructed the source, which will often lead to over fitting
of the noise in the data and an unrealistically complex and strucutred source. Regularization smooths the source
reconstruction solution by penalizing solutions where neighboring pixels (Voronoi triangles in this example) have
large flux differences.
"""
mesh = ag.mesh.RectangularMagnification(shape=mesh_shape)
regularization = ag.reg.Constant(coefficient=1.0)

pixelization = ag.Pixelization(
    image_mesh=image_mesh, mesh=mesh, regularization=regularization
)

"""
__Fit__

This is to illustrate the API for performing a fit via a pixelization using standard autolens objects like 
the `Galaxy`, `Tracer` and `FitImaging` 

We simply create a `Pixelization` and pass it to the source galaxy, which then gets input into the tracer.
"""
lens = ag.Galaxy(
    redshift=0.5,
    mass=ag.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = ag.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = ag.Tracer(galaxies=[lens, source])

fit = ag.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

"""
By plotting the fit, we see that the pixelized source does a good job at capturing the appearance of the source galaxy
and fitting the data to roughly the noise level.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Pixelizations have bespoke visualizations which show more details about the source-reconstruction, image-mesh
and other quantities.

These plots use an `InversionPlotter`, which gets its name from the internals of how pixelizations are performed in
the source code, where the linear algebra process which computes the source pixel fluxes is called an inversion.

The `subplot_mappings` overlays colored circles in the image and source planes that map to one another, thereby
allowing one to assess how the mass model ray-traces image-pixels and therefore to assess how the source reconstruction
maps to the image.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.subplot_of_mapper(mapper_index=0)
inversion_plotter.subplot_mappings(pixelization_index=0)

"""
The inversion can be extracted directly from the fit the perform these plots, which we also use below
for various calculations
"""
inversion = fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
__Mask Extra Galaxies__

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source.

If their emission is significant, and close enough to the lens and source, we may simply remove it from the data
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

__Pixelization / Mapper Calculations__

The pixelized source reconstruction output by an `Inversion` is often on an irregular grid (e.g. a
Voronoi triangulation or Voronoi mesh), making it difficult to manipulate and inspect after the lens modeling has
completed.

Internally, the inversion stores a `Mapper` object to perform these calculations, which effectively maps pixels
between the image-plane and source-plane.

After an inversion is complete, it has computed values which can be paired with the `Mapper` to perform calculations,
most notably the `reconstruction`, which is the reconstructed source pixel values.

By inputting the inversions's mapper and a set of values (e.g. the `reconstruction`) into a `MapperValued` object, we
are provided with all the functionality we need to perform calculations on the source reconstruction.

We set up the `MapperValued` object below, and illustrate how we can use it to interpolate the source reconstruction
to a uniform grid of values, perform magnification calculations and other tasks.
"""
mapper = inversion.cls_list_from(cls=ag.AbstractMapper)[
    0
]  # Only one source-plane so only one mapper, would be a list if multiple source planes

mapper_valued = ag.MapperValued(
    mapper=mapper, values=inversion.reconstruction_dict[mapper]
)

"""
__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).
 
- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `RectangularMagnification` mesh).

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
print(f"RectangularMagnification Mapper = {inversion.linear_obj_list[0]}")

"""
__Pixelization / Mapper Calculations__

The pixelized galaxy reconstruction output by an `Inversion` is often on an irregular grid (e.g. a 
Voronoi triangulation or Voronoi mesh), making it difficult to manipulate and inspect after the lens modeling has 
completed.

Internally, the inversion stores a `Mapper` object to perform these calculations, which effectively maps pixels
between the image-plane and source-plane. 

After an inversion is complete, it has computed values which can be paired with the `Mapper` to perform calculations,
most notably the `reconstruction`, which is the reconstructed source pixel values.

By inputting the inversions's mapper and a set of values (e.g. the `reconstruction`) into a `MapperValued` object, we
are provided with all the functionality we need to perform calculations on the reconstruction.

We set up the `MapperValued` object below, and illustrate how we can use it to interpolate the source reconstruction
to a uniform grid of values, perform magnification calculations and other tasks.
"""
mapper = inversion.cls_list_from(cls=ag.AbstractMapper)[
    0
]  # Only one source-plane so only one mapper, would be a list if multiple source planes

mapper_valued = ag.MapperValued(
    mapper=mapper, values=inversion.reconstruction_dict[mapper]
)

"""
__Interpolated Source__

A simple way to inspect the reconstruction is to interpolate its values from the irregular
pixelization o a uniform 2D grid of pixels.

(if you do not know what the `slim` and `native` properties below refer too, it 
is described in the `results/examples/data_structures.py` example.)

We interpolate the Voronoi triangulation this source is reconstructed on to a 2D grid of 401 x 401 square pixels. 
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401)
)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(interpolated_reconstruction.slim)

plotter = aplt.Array2DPlotter(
    array=interpolated_reconstruction,
)
plotter.figure_2d()

"""
By inputting the arc-second `extent` of the reconstruction, the interpolated array will zoom in on only these regions 
of the reconstruction. The extent is input via the notation (xmin, xmax, ymin, ymax), therefore  unlike the standard 
API it does not follow the (y,x) convention. 

Note that the output interpolated array will likely therefore be rectangular, with rectangular pixels, unless 
symmetric y and x arc-second extents are input.
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_reconstruction.slim)

"""
The interpolated errors on the reconstruction can also be computed, in case you are planning to perform 
model-fitting of the reconstruction.
"""
mapper_valued_errors = ag.MapperValued(
    mapper=mapper, values=inversion.reconstruction_noise_map_dict[mapper]
)

interpolated_errors = mapper_valued_errors.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_errors.slim)

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
