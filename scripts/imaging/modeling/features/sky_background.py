"""
Modeling Features: Sky Background
=================================

The background of an image is the light that is not associated with the galaxy we are interested in. This is due to
light from the sky, zodiacal light, and light from other galaxies in the field of view.

The background sky is often subtracted from image data during the data reduction procedure. If this subtraction is
perfect, there is then no need to include the sky in the model-fitting. However, it is difficult to achieve a perfect
subtraction.

The residuals of an imperfect back sky subtraction can leave a signal in the image which is degenerate with the
light profile of the galaxy. This is especially true for low surface brightness features, such as the faint outskirts
of a galaxy.

This example script illustrate how to include the sky background in the model-fitting of an `Imaging` dataset.

The code shows how to fit the sky background as a non-linear free parameter (e.g. an extra dimension in the non-linear
parameter space), and how to include the sky background as a linear light profile that is solved for (see the
feature `linear_light_profiles.py`). The latter is recommended for all model-fits, as it is faster and more reliable.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.
 - The sky background is included as a `Sky` light profile.

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

Load and plot the galaxy dataset `simple` via .fits files, which we will fit with the model.
"""
dataset_name = "simple"
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
Manually added a background sky of 10.0 electrons per second to the dataset, which is large compared to the signal
found in most real Astronomy images but will help us illustrate the sky background in this example.
"""
dataset.data += 10.0

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
 - The galaxy's disk is a parametric `Exponential` disk, whose centre is aligned with the bulge [4 parameters].
 - The sky background is included as a `Sky` light profile [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

The sky is not included in the `galaxies` collection, but is its own separate component in the overall model.

We update the prior on the `intensity` of the sky manually, such that it surrounds the true value of 10.0 electrons
per second. It is recommend you always update the prior on the sky's intensity manually, because the appropriate
prior depends on the dataset being fitted. However, when we use linear light profiles below, this is not necessary.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

sky = af.Model(ag.lp.Sky)
sky.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy), sky=sky)

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the sky is a model component that is not part of the `galaxies` collection.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="sky_background",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

Adding the background sky model to the analysis has a negligible impact on the run time, as it requires simply adding
a constant value to the data. The run time is therefore still of order ~0.01 seconds.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that a sky `intensity` value of approximately 10.0 electrons per second was inferred, as expected.
"""
print(result.info)

"""
To print the exact value, the `sky` attribute of the result contains the `intensity` of the sky.
"""
print(result.instance.sky.intensity)

"""
__Model (Linear)__

We now repeat the model-fit using a `Sky` as a linear light profile, which is recommended for all model-fits. 
Linear light profiles are described in more detail in the `linear_light_profiles.py` example feature script.

In short, they use linear algebra to solve for the `intensity` of a light profile, meaning that it is not a free
parameter fitted for by the non-linear search. This makes the model-fit faster and more reliable, and for a background
sky means we do not need to manually update the prior on the sky's intensity.

We compose our model where in this example:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 
 - The galaxy's disk is a linear parametric `Exponential` disk, whose centre is aligned with the bulge [3 parameters].
 - The sky background is included as a linear light profile [0 parameter].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.

Note how both light profiles use linear light profiles, meaning that the `intensity` parameter of both is also no 
longer a free parameter in the fit, like the sky.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy), sky=ag.lp_linear.Sky())

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the sky does not include an `intensity` parameter.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="sky_background_linear",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

For linear light profiles, the log likelihood evaluation increases to around ~0.05 seconds per likelihood evaluation.
This is still fast, but it does mean that the fit may take around five times longer to run. The run time is
~0.05 seconds when any linear light profiles are used, irrespective of whether its just the sky or also the
other light profiles.

Because three free parameters have been removed from the model (the `intensity` of the bulge, disk and sky), the 
total number of likelihood evaluations will reduce. Furthermore, the simpler parameter space likely means that the 
fit will take less than 10000 per free parameter to converge. 

Fits using standard light profiles and linear light profiles therefore take roughly the same time to run. However,
the simpler parameter space of linear light profiles means that the model-fit is more reliable, less susceptible to
converging to an incorrect solution and scales better if even more light profiles are included in the model.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)


"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that `intensity` parameters are not inferred by the model-fit.
"""
print(result.info)

"""
We plot the maximum likelihood fit, galaxy images and posteriors inferred via Nautilus.

The galaxy bulge and disk appear similar to those in the data, confirming that the `intensity` values inferred by
the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grid
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()


"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of the sky, we use the `max_log_likelihood_fit`, which has already performed the 
inversion and therefore the sky has the solved for `intensity`'s associated with it.

The implementation of a linear light profile sky is a bit strange, it actually has two `intensity`'s, one for the
positive component and one for the negative component. This is because the linear solver has a positivity constraint,
meaning that two separate sky's, positive and negative, are used to represent the sky background. One of these
values will be zero, the other the true sky background intensity.
"""
fit = result.max_log_likelihood_fit

sky = fit.sky_linear_light_profiles_to_light_profiles

print(sky.light_profile_list[0].intensity)
print(sky.light_profile_list[1].intensity)


"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.

In particular, checkout the results example `linear.py` which details how to extract all information about linear
light profiles from a fit.
"""
