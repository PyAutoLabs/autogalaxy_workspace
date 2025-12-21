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

We compose our model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a model where:

 - The galaxy's light uses a 20 x 20 `RectangularMagnification` mesh [0 parameters]. 
 
 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (3 parameters) than 
fitting this complex galaxy using parametric light profiles would (20+ parameters). 
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
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
    path_prefix=Path("imaging") / "features",
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data. 
"""
analysis = ag.AnalysisImaging(dataset=dataset, preloads=preloads, use_jax=True)


"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

pixelizations use a lot more VRAM than light profile-only models, with the amount required depending on the size of
dataset and the number of source pixels in the pixelization's mesh. For 400 source pixels, around 0.05 GB per batched
likelihood of VRAM is used. 

This is why the `batch_size` above is 20, lower than other examples, because reducing the batch size ensures a more 
modest amount of VRAM is used. If you have a GPU with more VRAM, increasing the batch size will lead to faster run times.

Given VRAM use is an important consideration, we print out the estimated VRAM required for this 
model-fit and advise you do this for your own pixelization model-fits.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

The run time of a pixelization are fast provided that the GPU VRAM exceeds the amount of memory required to perform
a likelihood evaluation.

Assuming the use of a 20 x 20 mesh grid above means this is the case, the run times of this model-fit on a GPU
should take under 10 minutes. If VRAM is exceeded, the run time will be significantly longer (3+ hours). CPU run
times are also of order hours, but can be sped up using the `numba` library (see the `pixelization/cpu` example).

The run times of pixelizations slow down as the data becomes higher resolution. In this example, data with a pixel
scale of 0.1" gives of order 10 minute run times (when VRAM is under control), for a pixel scale of 0.05" this
becomes around 30 minutes, and an hour for 0.03".

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
__Reconstruction CSV__

In the results `image` folder there is a .csv file called `source_plane_reconstruction_0.csv` which contains the
y and x coordinates of the pixelization mesh, the reconstruct values and the noise map of these values.

This file is provides all information on the source reconstruciton in a format that does not depend autolens
and therefore be easily loaded to create images of the source or shared collaobrations who do not have PyAutoLens
installed.

First, lets load `source_plane_reconstruction_0.csv` as a dictionary, using basic `csv` functionality in Python.
"""
try:
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
except FileNotFoundError:
    print(
        "CSV file not found. Please ensure the model-fit has been run and the file exists."
    )

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More 
- Source gradient calculations.
"""
