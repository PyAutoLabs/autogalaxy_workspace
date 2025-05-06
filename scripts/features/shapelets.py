"""
Modeling Features: Shapelets
============================

A shapelet is a basis function that is appropriate for capturing the exponential / disk-like features of a galaxy. It
has been employed in galaxy structure studies to model the light of the galaxy, because it can represent
features of disky star forming galaxies that a single Sersic function cannot.

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T
- https://iopscience.iop.org/article/10.1088/0004-637X/813/2/102 t

Shapelets are described in full in the following paper:

 https://arxiv.org/abs/astro-ph/0105178

This script performs a model-fit using shapelet, where it decomposes the galaxy light into ~20
Shapelets. The `intensity` of every Shapelet is solved for via linear algebra (see the `light_parametric_linear.py`
feature).

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
Shapelets can capture some of these features and can therefore better represent the emission of complex galaxies.

The shapelet model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this
example, the ~20 shapelets which represent the `bulge` of that are composed in a model corresponding to just
N=3 non-linear parameters (a `bulge` comprising a linear Sersic would give N=6).

Therefore, shapelet fit more complex galaxy morphologies using fewer non-linear parameters than the standard
light profile models!

__Disadvantages__

- There are many types of galaxy structure which shapelets may struggle to represent, such as a bar or assymetric
knots of star formation. They also rely on the galaxy have a distinct central over which the shapelets can be
centered, which is not the case of the galaxy is multiple merging systems or has bright companion galaxies.

- The linear algebra used to solve for the `intensity` of each shapelet has to allow for negative values of intensity
in order for shapelets to work. Negative surface brightnesses are unphysical, and are often inferred in a shapelet
decomposition, for example if the true galaxy has structure that cannot be captured by the shapelet basis.

- Computationally slower than standard light profiles like the Sersic.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's bulge is a super position of `ShapeletCartesianSph`` profiles.

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

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Apply adaptive over sampling to ensure the calculation is accurate, you can read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
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
__Fit__

We first show how to compose a basis of multiple shapelets and use them to fit the galaxy's light in data.

This is to illustrate the API for fitting shapelets using standard autogalaxy objects like the `Galaxy`
and `FitImaging`.

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the shapelets. However, shapelets can do a reasonable even if we just guess sensible 
parameter values.

__Basis__

We first build a `Basis`, which is built from multiple linear light profiles (in this case, shapelets). 

Below, we make a `Basis` out of 10 elliptical shapelet linear light profiles which: 

 - All share the same centre and elliptical components.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all shapelet basis 
   functions.
 
Note that any linear light profile can be used to compose a Basis. This includes Gaussians, which are often used to 
represent the light of elliptical galaxies (see `modeling/features/multi_gaussian_expansion.py`).
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = []

n_count = 1
m_count = -1

for i in range(total_n + total_m):
    shapelet = ag.lp_linear.ShapeletPolarSph(
        n=n_count, m=m_count, centre=(0.01, 0.01), beta=0.1
    )

    shapelets_bulge_list.append(shapelet)

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
)

"""
Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Tracer` and 
use it to fit data.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
By plotting the fit, we see that the `Basis` does a reasonable job at capturing the appearance of the galaxy.

There are few residuals, except for perhaps some central regions where the light profile is not perfectly fitted.

Given that there was no non-linear search to determine the optimal values of the Gaussians, this is a pretty good fit!
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Nevertheless, there are still residuals, which we now rectify by fitting the shapelets in a non-linear search.

__Model__

We compose our model where in this example:

 - The galaxy's bulge is a superposition of 10 parametric linear `ShapeletCartesianSph` profiles [3 parameters]. 
 - The centres of the Shapelets are all linked together.
 - The size of the Shapelet basis is controlled by a `beta` parameter, which is the same for all Shapelet basis 
   functions.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3. 

Note how this Shapelet model can capture features more complex than a Sersic, but has fewer non-linear parameters
(N=3 compared to N=7 for a `Sersic`).
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(ag.lp_linear.ShapeletPolar) for _ in range(total_n + total_m)
)

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
    profile_list=shapelets_bulge_list,
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
 - `autogalaxy_workspace/*/modeling/imaging/customize`
  
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

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
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
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

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
    profile_list=shapelets_bulge_list,
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
