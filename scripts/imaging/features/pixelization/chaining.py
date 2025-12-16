"""
Chaining: Parametric To Pixelization
====================================

This script chains two searches to fit `Imaging` data of a galaxy with a model where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - The galaxy's has four star forming clumps which are `Sersic` profiles.

The three searches break down as follows:

 1) Model the galaxy's bulge and disk components using a linear parametric `Sersic` and `Exponential` profiles that
 are fixed to the centre of the image.
 2) Fix these components to the maximum likelihood solution and add an `Inversion` which fits the clumps.
 3) Fit the bulge and disk light profiles simultaneously with the inversion that fits the clumps.

__Why Chain?__

There are a number of benefits of chaining a linear parametric galaxy model and `Inversion`, as opposed to fitting the
`Inversion` in one search:

 - The bulge and disk can be estimate somewhat accurately before we attempt to model the clumps. Thus, we can get
 a quick estimate of their parameters.

 - Parametric sources are computationally faster to fit. Therefore, even though the `Sersic` has more
 parameters for the search to fit than an `Inversion`, the model-fit is faster overall.

__Preloading__

When certain components of a model are fixed its associated quantities do not change during a model-fit. For
example, for a model where all light profiles are fixed, the PSF blurred model-image of those light profiles
is also fixed.

**PyAutoGalaxy** uses _implicit preloading_ to inspect the model and determine what quantities are fixed. It then stores
these in memory before the non-linear search begins such that they are not recomputed for every likelihood evaluation.

In this example no preloading occurs.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "complex"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = Path("imaging") / "chaining" / "parametric_to_pixelization"

"""
__Model__

Search 1 we fit a model where:

 - The galaxy's bulge is an `Sersic` with fixed centre [5 parameters].
 
 - The galaxy's disk is an `Exponential` with fixed centre [4 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

bulge.centre_0 = 0.0
bulge.centre_1 = 0.0
disk.centre_0 = 0.0
disk.centre_1 = 0.0

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model_1 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate model.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)

analysis_1 = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarised in the `info` attribute.
"""
print(result_1.info)

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

total_linear_light_profiles = 2

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
__Model (Search 2)__

We use the results of search 1 to create the model fitted in search 2, where:

 - The galaxy's bulge is an `Sersic` [0 parameters: parameters fixed from search 1].
 
 - The galaxy's disk is an `Exponential` [0 parameters: parameters fixed from search 1].

 - The galaxy's clumps are reconstructed `RectangularMagnification` mesh with resolution 50 x 50 [0 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=1.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the galaxy mass model.

The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.galaxy.bulge,
    disk=result_1.instance.galaxies.galaxy.disk,
    pixelization=pixelization,
)

model_2 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
__Analysis + Search + Model-Fit (Search 2)__

We now create the non-linear search and perform the model-fit using this model.
"""
analysis_2 = ag.AnalysisImaging(
    dataset=dataset,
    settings_inversion=ag.SettingsInversion(use_border_relocator=True),
    preloads=preloads,
    use_jax=True,
)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__pixelization_fixed_parametric",
    unique_tag=dataset_name,
    n_live=80,
    n_batch=20,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The final results can be summarised via printing `info`.
"""
print(result_2.info)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the model fitted in search 3, where:

 - The galaxy's bulge is an `Sersic` [7 parameters: priors initialized from search 1].

 - The galaxy's disk is an `Exponential` [6 parameters: priors initialized from search 1].

 - The galaxy's light uses a `RectangularMagnification` mesh[parameters fixed to results of search 2].

 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.

This search allows us to refit the bulge and disk components with an inversion that takes care of the clumps.
"""
bulge = af.Model(ag.lp_linear.Sersic)
bulge.ell_comps = result_1.model.galaxies.galaxy.bulge.ell_comps
bulge.effective_radius = result_1.model.galaxies.galaxy.bulge.effective_radius
bulge.sersic_index = result_1.model.galaxies.galaxy.bulge.sersic_index

disk = af.Model(ag.lp_linear.Sersic)
disk.ell_comps = result_1.model.galaxies.galaxy.disk.ell_comps
disk.effective_radius = result_1.model.galaxies.galaxy.disk.effective_radius

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk, pixelization=pixelization
)

model_3 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_3.info)

"""
__Analysis + Search + Model-Fit (Search 3)__

We now create the non-linear search and perform the model-fit using this model.
"""
search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]__parametric_and_pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
)

analysis_3 = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
The final results can be summarised via printing `info`.
"""
print(result_3.info)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a linear parametric light profile fit passed this model to a 
second search which modeled the galaxy's star forming clumps using an `Inversion`. We finished with a third 
search which fitted everything simultaneously, ensuring an accurate estimate of the galaxy's bulge and disk.

This was more computationally efficient than just fitting the light profiles and inversion from the offset.
"""
