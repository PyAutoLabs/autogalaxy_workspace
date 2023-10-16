"""
Modeling Features: Linear Light Profiles
========================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example by 2 dimensions).

This also removes the degeneracies that occur between the `intensity` and other light profile parameters
(e.g. `effective_radius`, `sersic_index`), which are difficult degeneracies for the non-linear search to map out
accurately. This produces more reliable lens model results and converge in fewer iterations, speeding up the overall
analysis.

The inversion has a relatively small computational cost, thus we reduce the model complexity with much expensive and
can therefore fit models more reliably and faster!

It is therefore recommended you always use linear light profiles to fit models over standard light profiles!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical.

**PyAutoGalaxy** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.
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
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 
 - The galaxy's disk is a linear parametric `Exponential` disk, whose centre is aligned with the bulge [3 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.

Note how both the lens and source galaxies use linear light profiles, meaning that the `intensity` parameter of both
is not longer a free parameter in the fit.

__Coordinates__

**PyAutoGalaxy** assumes that the galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the galaxy is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autogalaxy_workspace/*/preprocess`). 
 - Manually override the model priors (`autogalaxy_workspace/*/imaging/modeling/customize/priors.py`).
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/generag.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/). We make the following changes to the Nautilus settings:

 - Increase the number of live points, `n_live`, from the default value of 50 to 100. `n_live`
 
These changes are motivated by the higher dimensionality non-linear parameter space that including the galaxy light 
creates, which requires more thorough sampling by the non-linear search.

The folders: 

 - `autogalaxy_workspace/*/imaging/modeling/searches`.
 - `autogalaxy_workspace/*/imaging/modeling/customize`
  
Give overviews of the non-linear searches **PyAutoGalaxy** supports and more details on how to customize the
model-fit, including the priors on the model. 

If you are unclear of what a non-linear search is, checkout chapter 2 of the **HowToGalaxy** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autogalaxy_workspace/output/imaging/simple__sersic/mass[sie]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk_linear]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Plane` and `FitImaging` objects.
 - Information on the posterior as estimated by the `Nautilus` non-linear search. 
"""
print(result.max_log_likelihood_instance)

plane_plotter = aplt.PlanePlotter(
    plane=result.max_log_likelihood_plane, grid=result.grid
)
plane_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

search_plotter = aplt.NautilusPlotter(samples=result.samples)
search_plotter.cornerplot()

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
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
