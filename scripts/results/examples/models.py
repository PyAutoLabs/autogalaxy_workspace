"""
Results: Models
===============

Suppose we have the results of many fits and we only wanted to load and inspect a specific set
of model-fits (e.g. the results of `start_here.py`). We can use querying tools to only load the results we are
interested in.

This includes support for advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset)
can be loaded.

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""

import os

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


"""
__Aggregator__

Set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=Path("output") / "results_folder",
)

"""   
__Galaxies via Aggregator__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we want to inspect
the `Galaxies` objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Galaxies` object. For large datasets, this would require us to use generators to ensure it is memory-light,
which are cumbersome to write.

This example therefore uses the `GalaxiesAgg` object, which conveniently loads the `Galaxies` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a galaxies generator via the `ag.agg.GalaxiesAgg` object, where this `galaxies_gen` contains the maximum log
likelihood `Galaxies `object of every model-fit.
"""
galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_gen = galaxies_agg.max_log_likelihood_gen_from()

"""
We can now iterate over our galaxies generator to make the plots we desire.

The `galaxies_gen` returns a list of `Galaxies` objects, as opposed to just a single `Galaxies` object. This is because
only a single `Analysis` class was used in the model-fit, meaning there was only one imaging dataset that was
fit. 

The `multi` package of the workspace illustrates model-fits which fit multiple datasets 
simultaneously, (e.g. multi-wavelength imaging)  by summing `Analysis` objects together, where the `galaxies_list` 
would contain multiple `Galaxies` objects.

The parameters of galaxies in the `Galaxies` may vary across the datasets (e.g. different light profile intensities 
for different wavelengths), which would be reflected in the galaxies list.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

for dataset_list, galaxies_list in zip(dataset_gen, galaxies_gen):
    # Only one `Analysis` so take first and only dataset.
    dataset = dataset_list[0]

    # Only one `Analysis` so take first and only galaxies.
    galaxies = galaxies_list[0]

    # Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
    fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)
    galaxies = fit.galaxies_linear_light_profiles_to_light_profiles

    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(convergence=True, potential=True)

"""
__Luminosity Example__

Each galaxies has the information we need to compute the luminosity of that model. Therefore, lets print 
the luminosity of each of our most-likely galaxies.

The model instance uses the model defined by a pipeline. In this pipeline, we called the galaxy `galaxy`.
"""
dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_gen = galaxies_agg.max_log_likelihood_gen_from()

print("Maximum Log Likelihood Luminosities:")

for dataset_list, galaxies_list in zip(dataset_gen, galaxies_gen):
    # Only one `Analysis` so take first and only dataset.
    dataset = dataset_list[0]

    # Only one `Analysis` so take first and only tracer.
    galaxies = galaxies_list[0]

    # Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
    fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)
    galaxies = fit.galaxies_linear_light_profiles_to_light_profiles

    luminosity = galaxies[0].luminosity_within_circle_from(radius=10.0)

    print("Luminosity (electrons per second) = ", luminosity)


"""
__Errors (PDF from samples)__

In this example, we will compute the errors on the axis ratio of a model. Computing the errors on a quantity 
like the trap `density` is simple, because it is sampled by the non-linear search. The errors are therefore accessible
via the `Samples`, by marginalizing over all over parameters via the 1D Probability Density Function (PDF).

Computing the errors on the axis ratio is more tricky, because it is a derived quantity. It is a parameter or 
measurement that we want to calculate but was not sampled directly by the non-linear search. The `GalaxiesAgg` object 
object has everything we need to compute the errors of derived quantities.

Below, we compute the axis ratio of every model sampled by the non-linear search and use this determine the PDF 
of the axis ratio. When combining each axis ratio we weight each value by its `weight`. For Nautilus, 
the nested sampler used by the fit, this ensures models which gave a bad fit (and thus have a low weight) do not 
contribute significantly to the axis ratio error estimate.

We set `minimum_weight=`1e-4`, such that any sample with a weight below this value is discarded when computing the 
error. This speeds up the error computation by only using a small fraction of the total number of samples. Computing
a axis ratio is cheap, and this is probably not necessary. However, certain quantities have a non-negligible
computational overhead is being calculated and setting a minimum weight can speed up the calculation without 
significantly changing the inferred errors.

Below, we use the `GalaxiesAgg` to get the `Plane` of every Nautilus sample in each model-fit. We extract from each 
galaxies the model's axis-ratio, store them in a list and find the value via the PDF and quantile method. This again
uses generators, ensuring minimal memory use. 

In order to use these samples in the function `quantile`, we also need the weight list of the sample weights. We 
compute this using the `GalaxiesAgg`'s function `weights_above_gen_from`, which computes generators of the weights of all 
points above this minimum value. This again ensures memory use in minimag.
"""
galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_list_gen = galaxies_agg.all_above_weight_gen_from(minimum_weight=1e-4)
weight_list_gen = galaxies_agg.weights_above_gen_from(minimum_weight=1e-4)

for galaxies_gen, weight_gen in zip(galaxies_list_gen, weight_list_gen):
    axis_ratio_list = []

    for galaxies_list in galaxies_gen:
        # Only one `Analysis` so take first and only tracer.
        galaxies = galaxies_list[0]

        axis_ratio = ag.convert.axis_ratio_from(ell_comps=galaxies[0].bulge.ell_comps)

        axis_ratio_list.append(axis_ratio)

    weight_list = [weight for weight in weight_gen]

    try:
        median_axis_ratio, lower_axis_ratio, upper_axis_ratio = af.marginalize(
            parameter_list=axis_ratio_list, sigma=3.0, weight_list=weight_list
        )

        print(
            f"Axis-Ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}"
        )
    except IndexError:
        pass

"""
__Errors (Random draws from PDF)__

An alternative approach to estimating the errors on a derived quantity is to randomly draw samples from the PDF 
of the non-linear search. For a sufficiently high number of random draws, this should be as accurate and precise
as the method above. However, it can be difficult to be certain how many random draws are necessary.

The weights of each sample are used to make every random draw. Therefore, when we compute the axis-ratio and its errors
we no longer need to pass the `weight_list` to the `quantile` function.
"""
galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_list_gen = galaxies_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

for galaxies_gen in galaxies_list_gen:
    axis_ratio_list = []

    for galaxies_list in galaxies_gen:
        # Only one `Analysis` so take first and only tracer.
        galaxies = galaxies_list[0]

        axis_ratio = ag.convert.axis_ratio_from(ell_comps=galaxies[0].bulge.ell_comps)

        axis_ratio_list.append(axis_ratio)

    median_axis_ratio, lower_axis_ratio, upper_axis_ratio = af.marginalize(
        parameter_list=axis_ratio_list, sigma=3.0
    )

    print(f"Axis-Ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")


"""
Finish.
"""
