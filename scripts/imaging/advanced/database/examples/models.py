"""
Database: Models
================

In this tutorial, we use the database to load models and `Plane`'s from a non-linear search. This allows us to
visualize and interpret its results.

We then show how the database also allows us to load many `Plane`'s correspond to many samples of the non-linear
search. This allows us to compute the errors on quantities that the `Plane` contains, but were not sampled directly
by the non-linear search.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database("database.sqlite")

"""
__Plane via Database__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we want to inspect
the `Plane` object objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Plane` object. For large datasets, this would require us to use generators to ensure it is memory-light,
which are cumbersome to write.

This example therefore uses the `PlaneAgg` object, which conveniently loads the `Plane` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a plane generator via the `ac.agg.PlaneAgg` object, where this `plane_gen` contains the maximum log
likelihood `Plane `object of every model-fit.
"""
plane_agg = ag.agg.PlaneAgg(aggregator=agg)
plane_gen = plane_agg.max_log_likelihood_gen_from()

"""
We can now iterate over our plane generator to make the plots we desire.

The `plane_gen` returns a list of `Plane` objects, as opposed to just a single `Plane`object. This is because
only a single `Analysis` class was used in the model-fit, meaning there was only one `Plane` dataset that was
fit. 

The `multi` package of the workspace illustrates model-fits which fit multiple datasets 
simultaneously, (e.g. multi-wavelength imaging)  by summing `Analysis` objects together, where the `plane_list` 
would contain multiple `Plane` objects.

The parameters of galaxies in the `Plane` may vary across the datasets (e.g. different light profile intensities 
for different wavelengths), which would be reflected in the plane list.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for plane_list in plane_gen:

    # Only one `Analysis` so take first and only plane.
    plane = plane_list[0]

    plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
    plane_plotter.figures_2d(convergence=True, potential=True)

"""
__Luminosity Example__

Each plane has the information we need to compute the luminosity of that model. Therefore, lets print 
the luminosity of each of our most-likely galaxies.

The model instance uses the model defined by a pipeline. In this pipeline, we called the galaxy `galaxy`.
"""
plane_agg = ag.agg.PlaneAgg(aggregator=agg)
plane_gen = plane_agg.max_log_likelihood_gen_from()

print("Maximum Log Likelihood Luminosities:")

for plane_list in plane_gen:
    # Only one `Analysis` so take first and only tracer.
    plane = plane_list[0]

    luminosity = plane.galaxies[0].luminosity_within_circle_from(radius=10.0)

    print("Luminosity (electrons per second) = ", luminosity)


"""
__Errors (PDF from samples)__

In this example, we will compute the errors on the axis ratio of a model. Computing the errors on a quantity 
like the trap `density` is simple, because it is sampled by the non-linear search. The errors are therefore accessible
via the `Samples`, by marginalizing over all over parameters via the 1D Probability Density Function (PDF).

Computing the errors on the axis ratio is more tricky, because it is a derived quantity. It is a parameter or 
measurement that we want to calculate but was not sampled directly by the non-linear search. The `PlaneAgg` object 
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

Below, we use the `PlaneAgg` to get the `Plane` of every Nautilus sample in each model-fit. We extract from each 
plane the model's axis-ratio, store them in a list and find the value via the PDF and quantile method. This again
uses generators, ensuring minimal memory use. 

In order to use these samples in the function `quantile`, we also need the weight list of the sample weights. We 
compute this using the `PlaneAgg`'s function `weights_above_gen_from`, which computes generators of the weights of all 
points above this minimum value. This again ensures memory use in minimag.
"""
plane_agg = ag.agg.PlaneAgg(aggregator=agg)
plane_list_gen = plane_agg.all_above_weight_gen_from(minimum_weight=1e-4)
weight_list_gen = plane_agg.weights_above_gen_from(minimum_weight=1e-4)

for plane_gen, weight_gen in zip(plane_list_gen, weight_list_gen):

    axis_ratio_list = []

    for plane_list in plane_gen:
        # Only one `Analysis` so take first and only tracer.
        plane = plane_list[0]

        axis_ratio = ag.convert.axis_ratio_from(
            ell_comps=plane.galaxies[0].bulge.ell_comps
        )

        axis_ratio_list.append(axis_ratio)

    weight_list = [weight for weight in weight_gen]

    try:
        median_axis_ratio, upper_axis_ratio, lower_axis_ratio = af.marginalize(
            parameter_list=axis_ratio_list, sigma=3.0, weight_list=weight_list
        )

        print(f"Axis-Ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")
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
plane_agg = ag.agg.PlaneAgg(aggregator=agg)
plane_list_gen = plane_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

for plane_gen in plane_list_gen:
    axis_ratio_list = []

    for plane_list in plane_gen:
        # Only one `Analysis` so take first and only tracer.
        plane = plane_list[0]

        axis_ratio = ag.convert.axis_ratio_from(
            ell_comps=plane.galaxies[0].bulge.ell_comps
        )

        axis_ratio_list.append(axis_ratio)

    median_axis_ratio, upper_axis_ratio, lower_axis_ratio = af.marginalize(
        parameter_list=axis_ratio_list, sigma=3.0
    )

    print(f"Axis-Ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")


"""
Finish.
"""
