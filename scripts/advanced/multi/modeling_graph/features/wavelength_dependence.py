"""
Modeling: Light Parametric
========================================

This script fits a multi-wavelength `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

Three images are fitted, corresponding to a green ('g' band), red (`r` band) and near infrared ('I' band) images.

This script assumes previous knowledge of the `multi` modeling API found in other scripts in the `multi/modeling`
package. If anything is unclear check those scripts out.

__Effective Radius vs Wavelength__

Unlike other `multi` modeling scripts, the effective radius of the galaxy's bulge and disk are modeled as a user defined
function of wavelength, for example following a relation `y = (m * x) + c` -> `effective_radius = (m * wavelength) + c`.

By using a linear relation `y = mx + c` the free parameters are `m` and `c`, which does not scale with the number
of datasets. For datasets with multi-wavelength images (e.g. 5 or more) this allows us to parameterize the variation
of parameters across the datasets in a way that does not lead to a very complex parameter space.

If a free `effective radius` is created for every dataset, this would add 5+ free parameters to the model for 5+ datasets.
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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band), red (r-band) and 
near infrared (I-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r", "I"]

"""
__Wavelengths__

The effective radius of each source galaxy is parameterized as a function of wavelength.

Therefore we define a list of wavelengths of each color above.
"""
wavelength_list = [464, 658, 806]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12, 0.012]

"""
__Dataset__

Load and plot each multi-wavelength galaxy dataset, using a list of their waveband colors.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "wavelength_dependence"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    ag.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the galaxy model to the data, which we define
and use to set up the `Imaging` object that the galaxy model fits.

For multi-wavelength galaxy modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_list = [
    ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Model__

We compose our galaxy model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a galaxy model where:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [7 parameters]. 
 
 - The galaxy's disk is a linear parametric `Exponential` disk [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Model + Analysis__

We now make the galaxy bulge and disk `effective_radius` a free parameter across every analysis object.

We will assume that the `effective_radius` of the galaxy linearly varies as a function of wavelength, and therefore 
compute  the `effective_radius` value for each color image using a linear relation `y = mx + c`.

The function below is not used to compose the model, but illustrates how the `effective_radius` values were computed
in the corresponding `wavelength_dependence` simulator script.
"""


def bulge_effective_radius_from(wavelength):
    m = 1.0 / 100.0  # bulge appears brighter with increasing wavelength
    c = 3

    return m * wavelength + c


def disk_effective_radius_from(wavelength):
    m = -(1.2 / 100.0)  # disk appears fainter with increasing wavelength
    c = 10

    return m * wavelength + c


"""
To parameterize the above relation as a model, we compose `m` and `c` as priors and use PyAutoFit's prior arithmatic
to compose a model as a linear relation.
"""
bulge_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

disk_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
disk_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

"""
The free parameters of our model there are no longer `effective_radius` values, but the parameters `m` and `c` in the 
relation above. 

The model complexity therefore does not increase as we add more parameters to the model.

__Analysis__

We create an `Analysis` object for every dataset and sum it to combine the analysis of all images.
"""
analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis_factor_list = []

for wavelength, analysis in zip(wavelength_list, analysis_list):

    analysis_model = model.copy()
    galaxy = analysis_model.galaxies.galaxy

    galaxy.bulge.effective_radius = bulge_effective_radius = (wavelength * bulge_m) + bulge_c
    galaxy.disk.effective_radius = disk_effective_radius = (wavelength * disk_m) + disk_c

    analysis_factor = af.AnalysisFactor(prior_model=analysis_model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list)

#
# analysis_list = []
#
# for wavelength, dataset in zip(wavelength_list, dataset_list):
#     bulge_effective_radius = (wavelength * bulge_m) + bulge_c
#     disk_effective_radius = (wavelength * disk_m) + disk_c
#
#     analysis_list.append(
#         ag.AnalysisImaging(dataset=dataset).with_model(
#             model.replacing(
#                 {
#                     model.galaxies.galaxy.bulge.effective_radius: bulge_effective_radius,
#                     model.galaxies.galaxy.disk.effective_radius: disk_effective_radius,
#                 }
#             )
#         )
#     )
#
# analysis = sum(analysis_list)
# analysis.n_cores = 1

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling_graph"),
    name="wavelength_dependence",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.

For example, close inspection of the `max_log_likelihood_instance` of the two results shows that all parameters,
except the `effective_radius` of the source galaxy's `bulge`, are identical.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's galaxies shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    galaxies_plotter = aplt.GalaxiesPlotter(
        galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
    )
    galaxies_plotter.subplot_galaxies()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
"""
for result in result_list:
    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**.
"""
