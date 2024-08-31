"""
Modeling: Light Parametric Operated
===================================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a parametric `Sersic` bulge.
 - The galaxy includes a parametric `Gaussian` psf.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.

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

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
"""
dataset_name = "operated"
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
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

The prior on the operated `Gaussian`'s `sigma` value is very important, as it often the case that this is a
very small value (e.g. ~0.1). 

By default, **PyAutoGalaxy** assumes a `UniformPrior` from 0.0 to 5.0, but the scale of this value depends on 
resolution of the data. I therefore recommend you set it manually below, using your knowledge of the PSF size.
"""
bulge = af.Model(ag.lp.Sersic)
psf = af.Model(ag.lp_operated.Gaussian)

psf.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

bulge.centre = psf.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, psf=psf)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
There is also a linear variant of every operated light profile (see `light_parametric_linear.py`).

We will use this, as it simplifies parameter space, which is particularly important for operated light profiles 
which can prove quite difficult to sample robustly.

The number of free parameters and therefore the dimensionality of non-linear parameter space for this model is N=9.
"""
bulge = af.Model(ag.lp_linear.Sersic)
psf = af.Model(ag.lp_linear_operated.Gaussian)

psf.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

bulge.centre = psf.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, psf=psf)

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
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_psf]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data. 
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
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
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.uniform
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
