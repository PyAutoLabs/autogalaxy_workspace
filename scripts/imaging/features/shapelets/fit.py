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
To learn more about Basis functions, regularization and when you should use them, checkout the 
following **HowToGalaxy** tutorials:

 - `howtogalaxy/chapter_2_lens_modeling/tutorial_5_linear_profiles.ipynb`.
 - `howtogalaxy/chapter_4_pixelizations/tutorial_4_bayesian_regularization.ipynb.
"""
