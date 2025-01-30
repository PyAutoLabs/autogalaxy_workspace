"""
Results: Database
=================

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to perform
ellipse fits to the data.

We show how to use these tools to inspect the maximum log likelihood model of a fit to the data, customize things
like its visualization and also inspect fits randomly drawm from the PDF.

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is benefitial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Aggregator__

Aggregator use is different to other `results` examples, creating an .sqlite database file which is then used
throughout the example, with API that mirrors the normal aggregator.
"""
database_name = "ellipse"

if path.exists(path.join("output", f"{database_name}.sqlite")):
    os.remove(path.join("output", f"{database_name}.sqlite"))

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
The masks we used to fit the imaging data is accessible via the aggregator.
"""
mask_gen = agg.values("dataset.mask")
print([mask for mask in mask_gen])

"""   
__Ellipses via Aggregator__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we want to inspect
the `Ellipse` objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Ellipses` object. For large datasets, this would require us to use generators to ensure it is 
memory-light, which are cumbersome to write.

This example therefore uses the `EllipsesAgg` object, which conveniently loads the `Ellipses` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a ellipses generator via the `ag.agg.EllipsesAgg` object, where this `ellipses_gen` contains the maximum log
likelihood `Galaxies `object of every model-fit.
"""
ellipses_agg = ag.agg.EllipsesAgg(aggregator=agg)
ellipses_gen = ellipses_agg.max_log_likelihood_gen_from()

"""
We can now iterate over our ellipses generator to extract the information we desire.

The `ellipses_gen` returns a list of `Ellipses` objects, as opposed to just a single `Ellipses` object. This is because
only a single `Analysis` class was used in the model-fit, meaning there was only one imaging dataset that was
fit. 

The `multi` package of the workspace illustrates model-fits which fit multiple datasets 
simultaneously, (e.g. multi-wavelength imaging)  by summing `Analysis` objects together, where the `ellipses_list` 
would contain multiple `Ellipses` objects.

The parameters of ellipses in the `Ellipses` may vary across the datasets (e.g. different light profile intensities 
for different wavelengths), which would be reflected in the ellipses list.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for ellipses_lists_list in ellipses_gen:
    # Only one `Analysis` so take first and only ellipses.
    ellipses = ellipses_lists_list[0]

    for ellipse in ellipses:
        print(ellipse.major_axis)

"""
__Fits via Aggregator__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we inspect 
the `Imaging` objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Imaging` object. For large datasets, this would require us to use generators to ensure it is 
memory-light, which are cumbersome to write.

This example therefore uses the `ImagingAgg` object, which conveniently loads the `Imaging` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a dataset generator via the `ag.agg.ImagingAgg` object, where this `dataset_gen` contains the maximum log
likelihood `Imaging `object of every model-fit.

The `dataset_gen` returns a list of `Imaging` objects, as opposed to just a single `Imaging` object. This is because
only a single `Analysis` class was used in the model-fit, meaning there was only one `Imaging` dataset that was
fit. 

The `multi` package of the workspace illustrates model-fits which fit multiple datasets 
simultaneously, (e.g. multi-wavelength imaging)  by summing `Analysis` objects together, where the `dataset_list` 
would contain multiple `Imaging` objects.
"""
dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

for dataset_list in dataset_gen:
    # Only one `Analysis` so take first and only dataset.
    dataset = dataset_list[0]

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
We now use the aggregator to load a generator containing the fit of the maximum log likelihood model (and therefore 
galaxies) to each dataset.

Analogous to the `dataset_gen` above returning a list with one `Imaging` object, the `fit_gen` returns a list of
`FitEllipse` objects, because only one `Analysis` was used to perform the model-fit.
"""
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_lists_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit_list = fit_lists_list[0]

    fit_plotter = aplt.FitEllipsePlotter(fit_list=fit_list)
    fit_plotter.figures_2d(data=True)

"""
__Visualization Customization__

The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the PyAutoGalaxy `mat_plot`.

Below, we create a new function to apply as a generator to do this. However, we use a convenience method available 
in the aggregator package to set up the fit.
"""
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_lists_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit_list = fit_lists_list[0]

    mat_plot = aplt.MatPlot2D(
        figure=aplt.Figure(figsize=(12, 12)),
        title=aplt.Title(label="Custom Image", fontsize=24),
        yticks=aplt.YTicks(fontsize=24),
        xticks=aplt.XTicks(fontsize=24),
        cmap=aplt.Cmap(norm="log", vmax=1.0, vmin=1.0),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=20),
        units=aplt.Units(in_kpc=True),
    )

    fit_plotter = aplt.FitEllipsePlotter(fit_list=fit_list, mat_plot_2d=mat_plot)
    fit_plotter.figures_2d(data=True)

"""
Making this plot for a paper? You can output it to hard disk.
"""
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_lists_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit_list = fit_lists_list[0]

    mat_plot = aplt.MatPlot2D(
        title=aplt.Title(label="Hey"),
        output=aplt.Output(
            path=path.join("output", "path", "of", "file"),
            filename="publication",
            format="png",
        ),
    )

"""
__Errors (Random draws from PDF)__

In the `examples/models.py` example we showed how `Galaxies` objects could be randomly drawn form the Probability 
Distribution Function, in order to quantity things such as errors.

The same approach can be used with `FitEllipse` objects, to investigate how the properties of the fit vary within
the errors (e.g. showing how the model galaxy appearances changes for different fits).
"""
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)


for fit_list_gen in fit_gen:  # Total samples 2 so fit_list_gen contains 2 fits.
    for fit_lists_list in fit_gen:  # Iterate over each fit of total_samples=2
        # Only one `Analysis` so take first and only dataset.
        fit_list = fit_lists_list[0]

        fit_plotter = aplt.FitEllipsePlotter(fit_list=fit_list)
        fit_plotter.figures_2d(data=True)


"""
__Multipoles__

If you have performed a model-fit using multipoles, the database fully supports loading these results and has
dedicated tools for this.

First, lets build a database of a model-fit using multipoles.
"""
database_name = "ellipse_multipole"

if path.exists(path.join("output", f"{database_name}.sqlite")):
    os.remove(path.join("output", f"{database_name}.sqlite"))

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""   
__Multipoles via Aggregator__

Multipoles are included in the model as a separate component to the ellipses and therefore use their own separate
aggregator object.
"""
multipoles_agg = ag.agg.MultipolesAgg(aggregator=agg)
multipoles_gen = multipoles_agg.max_log_likelihood_gen_from()

for multipoles_lists_list in multipoles_gen:
    # Only one `Analysis` so take first and only multipoles.
    multipoles = multipoles_lists_list[0]

    for multipole_list in multipoles:
        print(multipole_list[0].m)
        print(multipole_list[1].m)

"""
The `FitEllipseAgg` automatically accounts for the multipoles in the model-fit if they are present.
"""
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_lists_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit_list = fit_lists_list[0]

    print(fit_list)

    print(fit_list[0].multipole_list)

    fit_plotter = aplt.FitEllipsePlotter(fit_list=fit_list)
    fit_plotter.figures_2d(data=True)

"""
Finished.
"""
