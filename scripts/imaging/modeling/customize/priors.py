"""
Customize: Priors
=================

This example demonstrates how to customize the priors of a model-fit, for example if you are modeling a galaxy where
certain parameters are known beforehand.

**Benefits:**: This will result in a faster more robust model-fit.

__Disadvantages__

The priors on your model determine the errors you infer. Overly tight priors may lead to over
confidence in the inferred parameters.

The `autogalaxy_workspace/*/imaging/modeling/customize/start_point.ipynb` shows an alternative API, which
customizes where the non-linear search starts its search of parameter space.

This cannot be used for a nested sampling method like `nautilus` (whose parameter space search is dictated by priors)
but can be used for the maximum likelihood estimator / MCMC methods PyAutoGalaxy supports.

The benefit of the starting point API is that one can tell the non-linear search where to look in parameter space
(ensuring a fast and robust fit) but retain uninformative priors which will not lead to over-confident errors.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Dataset__

Load and plot the galaxy dataset `simple` via .fits files, which we will fit with 
the model.
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
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose our model where in this example:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 
__Prior Customization__
 
We customize the parameter of every prior to values near the true valus, using the following priors:

- UniformPrior: The values of a parameter are randomly drawn between a `lower_limit` and `upper_limit`. For example,
the effective radius of ellipitical Sersic profiles typically assumes a uniform prior between 0.0" and 30.0".

- LogUniformPrior: Like a `UniformPrior` this randomly draws values between a `limit_limit` and `upper_limit`, but the
values are drawn from a distribution with base 10. This is used for the `intensity` of a light profile, as the
luminosity of galaxies follows a log10 distribution.

- GaussianPrior: The values of a parameter are randomly drawn from a Gaussian distribution with a `mean` and width
 `sigma`. For example, the $y$ and $x$ centre values in a light profile typically assume a mean of 0.0" and a
 sigma of 0.3", indicating that we expect the profile centre to be located near the centre of the image.
 
The API below can easily be adapted to customize the priors on a `disk` component, for example by simply making it
a `Model`. 
"""
bulge = af.Model(ag.lp_linear.Sersic)

bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.intensity = af.LogUniformPrior(lower_limit=0.5, upper_limit=1.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=0.5)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(galaxy.info)

"""
__Alternative API__

The priors can also be customized after the `galaxy` model object is created instead.
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

galaxy.bulge.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
galaxy.bulge.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
galaxy.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
galaxy.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
galaxy.bulge.intensity = af.LogUniformPrior(lower_limit=0.5, upper_limit=1.5)
galaxy.bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
galaxy.bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=0.5)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(galaxy.info)

"""
We could also customize the priors after the creation of the whole model.

Note that you can mix and match any of the API's above, and different styles will lead to concise and readable
code in different circumstances.
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

model.galaxies.galaxy.bulge.centre_0 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
model.galaxies.galaxy.bulge.centre_1 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
model.galaxies.galaxy.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.galaxy.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
model.galaxies.galaxy.bulge.intensity = af.LogUniformPrior(
    lower_limit=0.5, upper_limit=1.5
)
model.galaxies.galaxy.bulge.effective_radius = af.UniformPrior(
    lower_limit=0.5, upper_limit=1.5
)
model.galaxies.galaxy.bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=0.5)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(model.info)

"""
__Search + Analysis + Model-Fit__

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "customize"),
    name="priors",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)


analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
By inspecting the `model.info` file of this fit we can confirm the above priors were used. 
"""
