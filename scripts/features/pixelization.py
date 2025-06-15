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
from os import path
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
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
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
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a model where:

 - The galaxy's light uses a `Rectangular` meshwhose resolution is free to vary (2 parameters). 
 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (3 parameters) than 
fitting this complex galaxy using parametric light profiles would (20+ parameters). 
"""
pixelization = af.Model(
    ag.Pixelization, mesh=ag.mesh.Rectangular, regularization=ag.reg.Constant
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
    path_prefix=path.join("imaging", "modeling"),
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data. 
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
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
__Result (Advanced)__

The code belows shows all additional results that can be computed from a `Result` object following a fit with a
pixelization.

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
inversion = result.max_log_likelihood_fit.inversion
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
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file is provides all information on the source reconstruciton in a format that does not depend autolens
and therefore be easily loaded to create images of the source or shared collaobrations who do not have PyAutoLens
installed.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.
"""
import csv

with open(
    search.paths.image_path / "source_plane_reconstruction_0.csv", mode="r"
) as file:
    reader = csv.reader(file)
    header_list = next(reader)  # ['y', 'x', 'reconstruction', 'noise_map']

    reconstruction_dict = {header: [] for header in header_list}

    for row in reader:
        for key, value in zip(header_list, row):
            reconstruction_dict[key].append(float(value))

    # Convert lists to NumPy arrays
    for key in reconstruction_dict:
        reconstruction_dict[key] = np.array(reconstruction_dict[key])

print(reconstruction_dict["y"])
print(reconstruction_dict["x"])
print(reconstruction_dict["reconstruction"])
print(reconstruction_dict["noise_map"])

"""
You can now use standard libraries to performed calculations with the reconstruction on the mesh, again avoiding
the need to use autolens.

For example, we can create a Delaunay mesh using the scipy.spatial library, which is a triangulation
of the y and x coordinates of the pixelization mesh. This is useful for visualizing the pixelization
and performing calculations on the mesh.
"""
import scipy

points = np.stack(arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1)

delaunay = scipy.spatial.Delaunay(points)


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
