"""
Results: Start Here
===================

This script is the starting point for investigating the results of modeling and it provides
an overview of the modeling API.

After reading this script, the `examples` folder provides more detailed examples for analysing the different aspects of
performing modeling results outlined here.

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

from pathlib import Path
from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_batch=50,
    n_live=100,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

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
    directory=Path("output") / "results_folder",
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

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

It is recommended you use hard-disk loading to begin, as it is simpler and easier to use.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.

__Workflow Examples__

The `results/workflow` folder contains examples describing how to build a scientific workflow using the results
of model-fits, in order to quickly and easily inspect and interpret results.

These examples use functionality designed for modeling large dataset samples, with the following examples:

- `csv_maker.py`: Make .csv files from the modeling results which summarize the results of a large samples of fits.
- `png_maker.py`: Make .png files of every fit, to quickly check the quality of the fit and interpret the results.
- `fits_maker.py`: Make .fits files of every fit, to quickly check the quality of the fit and interpret the results.

The above examples work on the raw outputs of the model-fits that are stored in the `output` folder, for example
the visualization .png files, the .fits files containing results and parameter inferences which make the .csv files.

They are therefore often quick to run and allow you to make a large number of checks on the results of your model-fits
in a short period of time.

Below is a quick example, where we use code from the `csv_maker.py` scripts to create a .csv file from the fit above,
containing the inferred bulge sersic index, in a folder you can inspect quickly.

The `workflow_path` specifies where these files are output, in this case the .csv files which summarise the results,
and the code below can easily be adapted to output the .png and .fits files.
"""
workflow_path = Path("output") / "results_folder_csv_png_fits" / "workflow_make_example"

agg_csv = af.AggregateCSV(aggregator=agg)
agg_csv.add_variable(
    argument="galaxies.galaxy.bulge.sersic_index"
)  # Example of adding a column
agg_csv.save(path=workflow_path / "csv_very_simple.csv")

"""
__Result__

From here on we will use attributes contained in the `result` passed from the `search.fit` method above, as opposed
to using the aggregator. This is because things will run faster, but all of the results we use can be loaded using
the aggregator as shown above.

__Samples__

The result's `Samples` object contains the complete set of non-linear search nautilus samples, where each sample 
corresponds to a set of model parameters that were evaluated and accepted. 

The examples script `autogalaxy_workspace/*/results/examples/samples.py` provides a detailed description of 
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
__Linear Light Profiles__

In the model fit, linear light profiles are used, solving for the `intensity` of each profile through linear algebra.

The `intensity` value is not a free parameter of the linear light profiles in the model, meaning that in the `Samples`
object the `intensity` are always defaulted to values of 1.0 in the `Samples` object. 

You can observe this by comparing the `intensity` values in the `Samples` object to those in 
the `result.max_log_likelihood_galaxies` instance and `result.max_log_likelihood_fit` instance.
"""
samples = result.samples
ml_instance = samples.max_log_likelihood()

print(
    "Intensity of first galaxy's bulge in the Samples object (before solving linear algebra):"
)
print(ml_instance.galaxies.galaxy.bulge.intensity)

print(
    "Intensity of first galaxy's bulge in the max log likelihood galaxy (after solving linear algebra):"
)
print(result.max_log_likelihood_galaxies[0].bulge.intensity)
print(
    result.max_log_likelihood_fit.galaxies_linear_light_profiles_to_light_profiles[
        0
    ].bulge.intensity
)

"""
To interpret results associated with the linear light profiles, you must input the `Samples` object into a `FitImaging`,
which converts the linear light profiles to standard light profiles with `intensity` values solved for using the linear 
algebra.
"""
ml_instance = samples.max_log_likelihood()

fit = ag.FitImaging(dataset=dataset, galaxies=ml_instance.galaxies)
galaxies = fit.galaxies_linear_light_profiles_to_light_profiles

print("Intensity of first galaxy's bulge after conversion using FitImaging:")
print(galaxies[0].bulge.intensity)

"""
Whenever possible, the result already containing the solved `intensity` values is used, for example
the `Result` object returned by a search.

However, when manually loading results from the `Samples` object, you must use the `FitImaging` object to convert
the linear light profiles to their correct `intensity` values.

__Galaxies__

The result's maximum likelihood `Galaxies` object contains everything necessary to perform calculations with the model
like retrieving the images of each galaxy.

Following the discussion above, this object contains the correct `intensity` values for the light profiles which
are already solved via linear algebra.

The guide `autogalaxy_workspace/*/guides/galaxies.py` provides a detailed description of this object, including:

 - Producing individual images of the galaxies.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

The example result script `autogalaxy_workspace/*/results/examples/galaxies_fits.py` show how to use 
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

Following the discussion above, this object contains the correct `intensity` values for the light profiles which
are already solved via linear algebra.

The guide `autogalaxy_workspace/*/guides/fits.py` provides a detailed description of this object, including:

 - Performing a fit to data with galaxies.
 - Inspecting the model data, residual-map, chi-squared, noise-map of the fit.
 - Other properties of the fit that inspect how good it is.

The example result script `autogalaxy_workspace/*/results/examples/galaxies_fits.py` show how to use 
model-fitting results specific functionality of galaxies, including:

 - Repeating fits using the results contained in the samples.

Below, is an example of how to use the `FitImaging` object to print the maximum likelihood chi-squared and 
log likelihood values.
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

The example script `autogalaxy_workspace/*/modeling/imaging/linear_light_profiles.py` provides a detailed description of 
using linear light profile results including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.

__Pixelization__

The model can reconstruct the galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autogalaxy_workspace/*/results/examples/pixelizations.py` describes using pixelization 
results including:

 - Producing galaxy reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the galaxy's image using the pixelization.
"""
