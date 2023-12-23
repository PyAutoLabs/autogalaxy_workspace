"""
Results: Linear
===============

This tutorial inspects an inferred model using linear light profiles which solve for the intensity via linear
algebra, in the form of both linear light profiles (via `lp_linear`) and a `Basis` of linear light profiles.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality illustrated in this tutorial.

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

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
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
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

Note that the model that is fitted has both a linear light profile `Sersic` and a `Basis` with 10 `Gaussian` 
profiles which is regularized.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussians_bulge = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussians_bulge):
    gaussian.centre = gaussians_bulge[0].centre
    gaussian.ell_comps = gaussians_bulge[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))

bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_bulge,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=ag.lp_linear.Sersic)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk_linear]",
    unique_tag=dataset_name,
    n_live=100,
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
This `Inversion` can be used to plot the reconstructed image of specifically all linear light profiles.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
# inversion_plotter.figures_2d(reconstructed_image=True)

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_plane`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
plane = result.max_log_likelihood_plane

print(plane.galaxies[0].intensity)

"""
The `Plane` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result.max_log_likelihood_fit

plane = fit.plane

print(plane.galaxies[0].intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a plane) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
plane = result.max_log_likelihood_plane

plane_plotter = aplt.PlanePlotter(plane=plane, grid=dataset.grid)
plane_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=plane.galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Linear Objects (Internal Source Code)__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Rectangular` mesh).

In this example, two linear objects were used to fit the data:
 
 - An `Sersic` linear light profile.
 ` A `Basis` containing 5 Gaussians. 
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"LightProfileLinearObjFuncList (Sersic) = {inversion.linear_obj_list[0]}")
print(f"LightProfileLinearObjFuncList (Basis) = {inversion.linear_obj_list[1]}")

"""
Each of these `LightProfileLinearObjFuncList` objects contains its list of light profiles, which for the
`Sersic` is a single entry whereas for the `Basis` is 5 Gaussians.
"""
print(
    f"Linear Light Profile list (Sersic) = {inversion.linear_obj_list[0].light_profile_list}"
)
print(
    f"Linear Light Profile list (Basis -> x5 Gaussians) = {inversion.linear_obj_list[1].light_profile_list}"
)

"""
__Intensities (Internal Source Code)__

The intensities of linear light profiles are not a part of the model parameterization and therefore cannot be
accessed in the resulting galaxies, as seen in previous tutorials, for example:

plane = result.max_log_likelihood_plane
intensity = plane.galaxies[0].bulge.intensity

The intensities are also only computed once a fit is performed, as they must first be solved for via linear algebra. 
They are therefore accessible via the `Fit` and `Inversion` objects, for example as a dictionary mapping every
linear light profile (defined above) to the intensity values:
"""
fit = result.max_log_likelihood_fit

print(fit.linear_light_profile_intensity_dict)

"""
To extract the `intensity` values of a specific component in the model, we use that component as defined in the
`max_log_likelihood_plane`.
"""
plane = fit.plane

bulge = plane.galaxies[0].bulge
disk = plane.galaxies[0].disk

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of bulge (lp_linear.Sersic) = {fit.linear_light_profile_intensity_dict[bulge.light_profile_list[0]]}"
)
print(
    f"\n Intensity of first Gaussian in disk = {fit.linear_light_profile_intensity_dict[disk]}"
)

"""
A `Plane` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible.

For example, the linear light profile `Sersic` of the `bulge` component above has a solved for `intensity` of ~0.75. 

The `Plane` created below instead has an ordinary light profile with an `intensity` of ~0.75.
"""
plane = fit.model_obj_linear_light_profiles_to_light_profiles

print(
    f"Intensity via Plane With Ordinary Light Profiles = {plane.galaxies[0].bulge.intensity}"
)

"""
Finish.
"""
