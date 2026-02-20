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
Shapelets. The `intensity` of every Shapelet is solved for via linear algebra (see the `linear_light_profiles.py`
feature).

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using shapelets.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Basis:** How to create a basis of multiple light profiles, in this example shapelets.
**Coefficients:** A visualization of the real and imaginary shapelet coefficients in the Basis.
**Linear Light Profiles:** How to create a basis of linear light profiles to perform the shapelet decomposition.
**Fit:** Perform a fit to a dataset using linear light profile MGE.
**Intensities:** Access the solved for intensities of linear light profiles from the fit.
**Model:** Composing a model using shapelets and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of shapelet run times and discussion of how they compare to standard light profiles.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Shaeplet results, including accessing light profiles with solved for intensity values.
**Cartesian Shapelets:** Using shapelets definedon a Cartesian coordinate system instead of polar coordinates.
**Lens Shapelets:** Using shapelets to decompose the lens galaxy instead of the source galaxy.
**Regularization:** API for applying regularization to shapelets, which is not recommend but included for illustration.

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
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load and plot the galaxy dataset `light_basis` via .fits files, which we will fit with 
the model.
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
__Positive Negative Solver__

In other examples which use linear algebra to fit the data, for example linear light profiles, the Multi Gaussian
Expansion (MGE) and pixelization, we use a `positive_only` solver, which forces all solved for intensities to be
positive. This is a physical and sensible approach, because the surface brightnesses of a galaxy cannot be negative.

Shapelets cannot be solved for using a `positive_only` solver, because the shapelets ability to decompose the
light of a galaxy relies on the ability to use negative intensities. This is because the shapelets are not
physically motivated light profiles, but instead a mathematical basis that can represent any light profile.

This means shapelets may include negative flux in the reconstructed source galaxy, which is unphysical, and
a disadvantage of using shapelets.

The `Settings` object below uses a `use_positive_only_solver=False` to allow for negative intensities.

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
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the source galaxy is made of many `ShapeletPolar` profiles.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="shapelets",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to print the estimated VRAM 
required by a model.

For each shapelet, extra VRAM is used. For around 60 shapelets this typically requires  a modest amount of 
VRAM (e.g. 10–50 MB per batched likelihood). Models that use hundreds of shapelets, especially in  combination with a 
large batch size, may therefore exceed GBs of VRAM and require you to adjust the batch size to fit within your GPU's VRAM.

__Run Time__

The likelihood evaluation time for a shapelets is significantly slower than standard light profiles.
This is because the image of every shapelets must be computed and evaluated, and each must be blurred with the PSF.
In this example, the evaluation time is ~0.37s, compared to ~0.01 seconds for standard light profiles.

Gains in the overall run-time however are made thanks to the models reduced complexity and lower
number of free parameters. The source is modeled with 3 free parameters, compared to 6+ for a linear light profile 
Sersic.

However, the multi-gaussian expansion (MGE) approach is even faster than shapelets. It uses fewer Gaussian basis
functions (speed up the likelihood evaluation) and has fewer free parameters (speeding up the non-linear search).
Furthermore, non of the free parameters scale the size of the source galaxy, which means the non-linear search
can converge faster.

I recommend you try using an MGE approach alongside shapelets. For many science cases, the MGE approach will be
faster and give higher quality results. Shapelets may perform better for irregular sources, but this is not
guaranteed.

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
Checkout `autogalaxy_workspace/*/results` for a full description of analysing results in **PyAutoGalaxy**, which 
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
    path_prefix=Path("imaging") / "features",
    name="light[basis_regularized]",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

result = search.fit(model=model, analysis=analysis)


"""
__Shapelet Cartesian__

The shapelets above were defined on a polar grid, which is suitable for modeling radially symmetric sources like
most galaxies.

An alternative approach is to define the shapelets on a Cartesian grid, which we plot the basis of below
and show an example fit.

These are generally not recommended for modeling galaxies, but may be better in certain situations.
"""
total_xy = 5

shapelets_bulge_list = []

for x in range(total_xy):
    for y in range(total_xy):
        shapelet = ag.lp.ShapeletCartesian(
            n_y=y,
            n_x=x,
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            intensity=1.0,
            beta=1.0,
        )

        shapelets_bulge_list.append(shapelet)

bulge = ag.lp_basis.Basis(profile_list=shapelets_bulge_list)

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Cartesian Shapelets__
"""
total_xy = 5

shapelets_bulge_list = []

for x in range(total_xy):
    for y in range(total_xy):
        shapelet = ag.lp_linear.ShapeletCartesian(
            n_y=y, n_x=x, centre=(0.0, 0.0), ell_comps=(0.0, 0.0), beta=1.0
        )

        shapelets_bulge_list.append(shapelet)

bulge = ag.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(
    dataset=dataset,
    galaxies=galaxies,
    settings=ag.Settings(use_positive_only_solver=False),
)
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

galaxies = fit.model_obj_linear_light_profiles_to_light_profiles

basis_plotter = aplt.BasisPlotter(basis=galaxies[0].bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Model__

Here is how we compose a model using Cartesian shapelets.
"""
total_xy = 5

shapelets_bulge_list = af.Collection(
    af.Model(ag.lp_linear.ShapeletCartesian) for _ in range(total_xy**2)
)

for x in range(total_xy):
    for y in range(total_xy):
        shapelet.n_y = y
        shapelet.n_x = x

        shapelet.centre = shapelets_bulge_list[0].centre
        shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
        shapelet.beta = shapelets_bulge_list[0].beta

bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
)

galaxy = af.Model(ag.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="shapelets_cartesian",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
)

result = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

This script has illustrated how to use shapelets to model the light of galaxies.

Shapelets are a powerful basis function for capturing complex morphological features of galaxies that standard
light profiles struggle to represent. However, they do have drawbacks, such as the need to allow for negative
intensities in the solution, which is unphysical. 

As a rule of thumb, modeling is generally better if a pixelization is used to reconstruct the source galaxy's light,
but shapelets can be a useful middle-ground between standard light profiles and a pixelization.
"""
