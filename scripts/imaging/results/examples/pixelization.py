"""
Results: Inversion
==================

This tutorial illustrates how to analyse the results of modeling where the galaxy is modeled using an
`Inversion` and therefore has a pixelized reconstruction we may be interested in inspecting.

This includes examples of how to output the reconstruction to .fits files, so that a more detailed analysis
can be performed.

This tutorial focuses on explaining how to use the inferred inversion to compute results as numpy arrays.

__Plot Module__

This example uses the **PyAutoGalaxy** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/imaging/results/examples/data_structure.ipynb` example.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

pixelization = af.Model(
    ag.Pixelization, mesh=ag.mesh.Rectangular, regularization=ag.reg.Constant
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[pixelization]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
This `Inversion` can be used to plot the reconstructed image of specifically all linear light profiles and the
reconstruction of the `Pixelization`.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).
 
- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Rectangular` mesh).

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
print(f"Rectangular Mapper = {inversion.linear_obj_list[0]}")

"""
__Interpolated Galaxy__

The pixelized reconstruction used by an `Inversion` may be a different resolution to the data, making it difficult to 
manipulate and inspect after the modeling has completed.

A simpler way to inspect the reconstruction is to interpolate the reconstruction values from rectangular grid
resolution to a uniform 2D grid of pixels.

(if you do not know what the `slim` and `native` properties below refer too, check back to tutorial 2 of the results
for a description).

Inversions can have multiple reconstructions (e.g. if separate pixelizations are used for each galaxy) thus the 
majority of quantities are returned as a list. It is likely you are only using one `Inversion` to one galaxy,
so these lists will likely contain only one entry

We interpolate the pixelization this galaxy is reconstructed on to a 2D grid of 401 x 401 square pixels. 
"""
interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
    shape_native=(401, 401)
)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(interpolated_reconstruction_list[0].slim)

"""
We can alternatively input the arc-second `extent` of the reconstruction we want, which will not use square 
pixels unless symmetric y and x arc-second extents are input.

The extent is input via the notation (xmin, xmax, ymin, ymax), therefore unlike most of the **PyAutoGalaxy** API it
does not follow the (y,x) convention. This will be updated in a future version.
"""
interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_reconstruction_list[0].slim)

"""
The interpolated errors on the reconstruction can also be computed, in case you are planning to perform 
model-fitting of the source reconstruction.
"""
interpolated_errors_list = inversion.interpolated_errors_list_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_errors_list[0].slim)

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
