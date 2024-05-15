"""
Results: Start Here
===================

This script is the starting point for investigating the results of modeling and it provides
an overview of the modeling API.

The majority of results are dataset independent, meaning that the same API can be used to inspect the results of any
model. Therefore, for the majority of results we refer you to the `autogalaxy_workspace/imaging/results` package,
which details the API which can be copy and pasted for interferometer fits.

The `examples` folder here does provide specific examples of how to inspects the results of fits using
interferometer datasets.

__Model__

We begin by fitting a quick model to a simple dataset, which we will use to illustrate the modeling
results API.

If you are not familiar with the modeling API and process, checkout the `autogalaxy_workspace/examples/modeling`
folder for examples.
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
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerNUFFT
)

bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("interferometer", "modeling"),
    name="light[bulge_disk]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = ag.AnalysisInterferometer(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
__Loading From Hard-disk__

When performing fits which output results to hard-disk, a `files` folder is created containing .json / .csv files of 
the model, samples, search, etc. You should check it out now for a completed fit on your hard-disk if you have
not already!

These files can be loaded from hard-disk to Python variables via the aggregator, making them accessible in a 
Python script or Jupyter notebook. They are loaded as the internal **PyAutoFit** objects we are familiar with,
for example the `model` is loaded as the `Model` object we passed to the search above.

Below, we will access these results using the aggregator's `values` method. A full list of what can be loaded is
as follows:

 - `model`: The `model` defined above and used in the model-fit (`model.json`).
 - `search`: The non-linear search settings (`search.json`).
 - `samples`: The non-linear search samples (`samples.csv`).
 - `samples_info`: Additional information about the samples (`samples_info.json`).
 - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
 - `info`: The info dictionary passed to the search (`info.json`).
 - `covariance`: The inferred covariance matrix (`covariance.csv`).
 - `cosmology`: The cosmology used by the fit (`cosmology.json`).
 - `settings_inversion`: The settings associated with a inversion if used (`settings_inversion.json`).
 - `dataset/data`: The data that is fitted (`data.fits`).
 - `dataset/noise_map`: The noise-map (`noise_map.fits`).
 - `dataset/psf`: The Point Spread Function (`psf.fits`).
 - `dataset/mask`: The mask applied to the data (`mask.fits`).
 - `dataset/settings`: The settings associated with the dataset (`settings.json`).

The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does not reperform calculations using the full 
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the `samples.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder"),
)

"""
__Generators__

Before using the aggregator to inspect results, lets discuss Python generators. 

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects 
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory 
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once. 
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the 
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the 
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the `values` method. This takes the `name` of the
object we want to create a generator of, for example inputting `name=samples` will return the results `Samples`
object (which is illustrated in detail below).
"""
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

"""
__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is benefitial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

It is recommended you use hard-disk loading to begin, as it is simpler and easier to use.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.

__Result__

From here on we will use attributes contained in the `result` passed from the `search.fit` method above, as opposed
to using the aggregator. This is because things will run faster, but all of the results we use can be loaded using
the aggregator as shown above.

__Samples__

The result's `Samples` object contains the complete set of non-linear search nautilus samples, where each sample 
corresponds to a set of model parameters that were evaluated and accepted. 

The examples script `autogalaxy_workspace/*/imaging/results/examples/samples.py` provides a detailed description of 
this object, including:

 - Extracting the maximum likelihood model.
 - Using marginalized PDFs to estimate errors on the model parameters.
 - Deriving errors on derived quantities, such as the Einstein radius.

Below, is an example of how to use the `Samples` object to estimate the mass model parameters which are 
the median of the probability distribution function and its errors at 3 sigma confidence intervals.
"""
samples = result.samples

median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance.galaxies.galaxy.bulge)
print()

ue3_instance = samples.values_at_upper_sigma(sigma=3.0)
le3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance.galaxies.galaxy.bulge, "\n")
print(le3_instance.galaxies.galaxy.bulge, "\n")

"""
__Galaxies__

The result's maximum likelihood `Galaxies` object contains everything necessary to perform calculations with the model
like retrieving the images of each galaxy.

The guide `autogalaxy_workspace/*/guides/galaxies.py` provides a detailed description of this object, including:

 - Producing individual images of the galaxies.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

The example result script `autogalaxy_workspace/*/imaging/results/examples/galaxies_fits.py` show how to use 
model-fitting results specific functionality of galaxies, including:

 - Drawing galaxies from the samples and plotting their images.
 - Producing 1D plots of the galaxy's light and mass profiles with error bars.

Below, is an example of how to use the result's `Galaxies` object to calculate the image of the galaxies.
"""
galaxies = result.max_log_likelihood_galaxies

image = galaxies.image_2d_from(grid=dataset.grid)

"""
__Fits__

The result's maximum likelihood `FitImaging` object contains everything necessary to inspect the model fit to the 
data.

The guide `autogalaxy_workspace/*/guides/fits.py` provides a detailed description of this object, including:

 - Performing a fit to data with galaxies.
 - Inspecting the model data, residual-map, chi-squared, noise-map of the fit.
 - Other properties of the fit that inspect how good it is.

The example result script `autogalaxy_workspace/*/imaging/results/examples/galaxies_fits.py` show how to use 
model-fitting results specific functionality of galaxies, including:

 - Repeating fits using the results contained in the samples.

This script uses a `FitImaging` object, but the API for the majority of quantities are identical for an 
interferometer fit.

Below, is an example of how to use the `FitImaging` object to output the source reconstruction to print the 
chi-squared and log likelihood values.
"""
fit = result.max_log_likelihood_fit

print(fit.chi_squared)
print(fit.log_likelihood)

"""
__Units and Cosmological Quantities__

The maximum likelihood model includes cosmological quantities, which can be computed via the result.

The guide `autogalaxy_workspace/*/guides/units_and_cosmology.py` provides a detailed description of this object, 
including:

 - Calculating the Einstein radius of the galaxy.
 - Converting quantities like the Einstein radius or effective radius from arcseconds to kiloparsecs.
 - Computing the Einstein mass of the galaxy in solar masses.

This guide is not in the `results` package but the `guides` package, as it is a general guide to the
**PyAutoGalaxy** API. However, it may be useful when inspecting results.

Below, is an example of how to convert the effective radius of the galaxy from arcseconds to kiloparsecs.
"""
galaxies = result.max_log_likelihood_galaxies

cosmology = ag.cosmo.Planck15()

galaxy = galaxies[0]
galaxy_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=galaxy.redshift)
galaxy_effective_radius_kpc = galaxy.bulge.effective_radius * galaxy_kpc_per_arcsec

"""
__Linear Light Profiles / Basis Objects__

A model can be fitted using a linear light profile, which is a light profile whose `intensity` parameter is 
sovled for via linear algebra.

This includes Basis objects such as a Multi-Gaussian expansion of Shapelets.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality.

The example script `autogalaxy_workspace/*/imaging/modeling/linear_light_profiles.py` provides a detailed description of 
using linear light profile results including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.

__Pixelization__

The model can reconstruct the galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autogalaxy_workspace/*/imaging/results/examples/pixelizations.py` describes using pixelization 
results including:

 - Producing galaxy reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the galaxy's image using the pixelization.
"""
