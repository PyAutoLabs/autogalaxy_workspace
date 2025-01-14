"""
Modeling: Light Parametric
==========================

This script fits multiple multi-wavelength `Imaging` datasets of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

Two images are fitted, corresponding to a greener ('g' band) and redder image (`r` band).

This is an advanced script and assumes previous knowledge of the core **PyAutoGalaxy** API for galaxy modeling. Thus,
certain parts of code are not documented to ensure the script is concise.

__Linear Light Profiles__

The example `multi/light_parametric_linear.py` shows an example scripts which use linear light profiles,
where the `intensity` parameters of each light profile components is solved via linear algebra.

These can straight forwardly be used for multi-wavelength datasets, by simply changing the light profiles
in the model below from `ag.lp_linear.Sersic` to `ag.lp_linear.Sersic`.

In this script, we make the `intensity` parameter of each component a free parameter in every waveband of imaging,
increasing the number of free parameters and dimensionality of non-linear parameter space for every waveband of
imaging we fit. By using linear light profiles, each component can effectively have a free `intensity` in a way that
does not make parameter space more complex.
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

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Load and plot each multi-wavelength galaxy dataset, using a list of their waveband colors.

Note how the disk appears brighter in the g-band image, whereas the bulge is clearer in the r-band image.
Multi-wavelength image can therefore better decompose the structure of galaxies.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

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
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

For a new user, the details of over-sampling are not important, therefore just be aware that below we make it so that 
all calculations use an adaptive over sampling scheme which ensures high accuracy and precision.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
for dataset in dataset_list:
    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Analysis__

We create an `Analysis` object for every dataset.
"""
analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
By summing this list of analysis objects, we create an overall `CombinedAnalysis` which we can use to fit the 
multi-wavelength imaging data, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis objects (e.g. the fit to each separate waveband).

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis and therefore each dataset.
 
 - Next, we will use this combined analysis to parameterize a model where certain galaxy parameters vary across
 the dataset.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 1

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
Galaxies change appearance across wavelength, with the most significant change in their brightness.

Models applied to summed analyses can be extended to include free parameters specific to each dataset. In this example,
we want the galaxy's effective radii to vary across the g and r-band datasets.

The API for doing this is shown below, where by inputting the `effective_radius` model parameters to the `with_free_parameter` 
method the effective_radius of the lens's bulge and source's bulge become free parameters across every analysis object.

NOTE: Other aspects of galaxies may vary across wavelength, none of which are included in this example. The API below 
can easily be extended to include these additional parameters, and the `features` package explains other tools for 
extending the model across datasets.
"""
analysis = analysis.with_free_parameters(
    model.galaxies.galaxy.bulge.effective_radius,
    model.galaxies.galaxy.disk.effective_radius,
)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="light[bulge_disk]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=model, analysis=analysis)

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
Plotting each result's galaxies shows that the galaxy appears different, owning to its different intensities.
"""
for result in result_list:
    galaxies_plotter = aplt.GalaxiesPlotter(
        galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
    )
    galaxies_plotter.subplot_galaxies()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=16). 

Therefore, the samples is identical in every result object.
"""
plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_cornerpy()

"""
Checkout `autogalaxy_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**.
"""
