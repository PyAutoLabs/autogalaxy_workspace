"""
Modeling: Light Shapelets
=========================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's bulge is a super position of `ShapeletCartesian`` profiles.

__Basis / Shapelet Fitting__

Fits using symmetric light profiles such as elliptical Sersics often leave significant residuals, because they do not
capture irregular and asymmetric features within a galaxy, for example isophotal twists, varying radial ellipticity or
disruption due to mergers.

Basis fitting uses a super position of light profiles to represent the different structural components within a
galaxy. The `intensity` value of every basis function is solved for via linear
algebra (see `light_parametric_linear.py`), meaning that the super position can adapt so as to capture these
irregular and asymmetric features.

This example fits a galaxy with asymmetric features using shapelet basis functions. Shapelets are basis functions
with analytic properties that are appropriate for capturing exponential / disk-like features in a galaxy. They
do so over a wide range of scales, and can often represent features in these galaxies that a single Sersic function
cannot.

Shapelets are described in full in the following papers:

 https://arxiv.org/abs/astro-ph/0105178

 __Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For shapelets, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.
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

Load and plot the galaxy dataset `light_basis` via .fits files, which we will fit with 
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

 - The galaxy's bulge is a superposition of 10 parametric linear `ShapeletCartesian` profiles [3 parameters]. 
 - The centres of the Shapelets are all linked together.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all Shapelet basis 
   functions.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3. 

Note how this Shapelet model can capture features more complex than a Sersic, but has fewer non-linear parameters
(N=3 compared to N=7 for a `Sersic`).

__Coordinates__

**PyAutoGalaxy** assumes that the galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the galaxy is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autogalaxy_workspace/*/preprocess`). 
 - Manually override the model priors (`autogalaxy_workspace/*/imaging/modeling/customize/priors.py`).
"""
# shapelets_bulge_list = af.Collection(
#     af.Model(ag.lp_shapelets.ShapeletCartesian) for _ in range(20)
# )

# for i, shapelet in enumerate(shapelets_bulge_list):
#
#     shapelet.n_y = i
#     shapelet.n_x = i
#     shapelet.centre = shapelets_bulge_list[0].centre
#     shapelet.beta = shapelets_bulge_list[0].beta

# total_shapelets = 10
#
# shapelets_bulge_list = af.Collection(
#     af.Model(ag.lp_shapelets.ShapeletCartesian) for _ in range(total_shapelets**2)
# )
#
# y_count = 1
#
# for i, shapelet in enumerate(shapelets_bulge_list):
#
#     shapelet.n_y = y_count
#     shapelet.n_x = (i % total_shapelets) + 1
#
#     if shapelet.n_x == total_shapelets:
#         y_count += 1
#
#     shapelet.centre = shapelets_bulge_list[0].centre
#     shapelet.beta = shapelets_bulge_list[0].beta

total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(ag.lp_shapelets.ShapeletPolarEll) for _ in range(total_n + total_m)
)

# shapelets_bulge_list = af.Collection(
#     af.Model(ag.lp_shapelets.ShapeletExponential) for _ in range(total_n+total_m)
# )

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    shapelet.n = n_count
    shapelet.m = m_count

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.beta = shapelets_bulge_list[0].beta

bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format, which has a lot more parameters than other examples
as it shows the parameters of every individual Shapelet.

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
    name="light[shapelets]_polar_5_ell",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
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
Checkout `autogalaxy_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**, which 
includes a dedicated tutorial for linear objects like basis functions.

__Regularization__

There is one downside to `Basis` functions, we may compose a model with too much freedom. The `Basis` (e.g. our 20
Shapelets) may overfit noise in the data, or possible the galaxyed source galaxy emission -- neither of which we 
want to happen! 

To circumvent this issue, we have the option of adding regularization to a `Basis`. Regularization penalizes
solutions which are not smooth -- it is essentially a prior that says we expect the component the `Basis` represents
(e.g. a bulge or disk) to be smooth, in that its light changes smoothly as a function of radius.

Below, we compose and fit a model using Basis functions which includes regularization, which adds one addition 
parameter to the fit, the `coefficient`, which controls the degree of smoothing applied.
"""
bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=shapelets_bulge_list,
    regularization=ag.reg.Constant,
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, which has addition priors now associated with regularization.
"""
print(model.info)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[basis_regularized]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
To learn more about Basis functions, regularization and when you should use them, checkout the 
following **HowToGalaxy** tutorials:

 - `howtogalaxy/chapter_2_lens_modeling/tutorial_5_linear_profiles.ipynb`.
 - `howtogalaxy/chapter_4_pixelizations/tutorial_4_bayesian_regularization.ipynb.
"""
