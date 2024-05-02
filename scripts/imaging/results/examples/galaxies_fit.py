"""
Results: Galaxies and Fits
===========================

This tutorial inspects an inferred model using galaxies inferred by the non-linear search.
This allows us to visualize and interpret its results.

The galaxies and fit API is described fully in the guides:

 - `autogalaxy_workspace/*/guides/galaxies.ipynb`
 - `autogalaxy_workspace/*/guides/fit.ipynb`

This result example only explains specific functionality for using a `Result` object to inspect galaxies or a fit
and therefore you should read these guides in detail first.

__Plot Module__

This example uses the **PyAutoGalaxy** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autogalaxy_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/guides/data_structure.ipynb` guide.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
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

Note that the model that is fitted has two galaxies, as opposed to just one like usual!
"""
dataset_name = "sersic_x2"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

bulge_0 = af.Model(ag.lp.Sersic)
bulge_0.centre = (0.0, -1.0)

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_0)

bulge_1 = af.Model(ag.lp.Sersic)
bulge_1.centre = (0.0, 1.0)

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_1)

model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]__x2",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Galaxies__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_galaxies` which we can visualize.
"""
galaxies = result.max_log_likelihood_galaxies

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=mask.derive_grid.all_false
)
galaxies_plotter.subplot_galaxies()

"""
__Samples__

In the first results tutorial, we used `Samples` objects to inspect the results of a model.

We saw how these samples created instances, which include a `galaxies` property that mains the API of the `Model`
creates above (e.g. `galaxies.galaxy.bulge`). 

We can also use this instance to extract individual components of the model.
"""
samples = result.samples

ml_instance = samples.max_log_likelihood()

bulge = ml_instance.galaxies.galaxy_0.bulge

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
print(bulge_image_2d.slim[0])

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=grid)
bulge_plotter.figures_2d(image=True)

"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light models estimated via a model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of the model-fit. 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `effective_radius` with errors, which are plotted as vertical lines.

Below, we manually input one hundred realisations of the galaxy with light profiles that clearly show 
these errors on the figure.
"""
galaxy_pdf_list = [samples.draw_randomly_via_pdf().galaxies.galaxy_0 for i in range(10)]

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=galaxy_pdf_list, grid=dataset.grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(image=True)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(image=True)

"""
__Refitting__

Using the API introduced in the `samples.py` tutorial, we can also refit the data locally. 

This allows us to inspect how the galaxies changes for models with similar log likelihoods. Below, we create and plot
the galaxies of the 100th last accepted model by nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

galaxies = ag.Galaxies(galaxies=instance.galaxies)

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.subplot_galaxies()

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

galaxies = ag.Galaxies(galaxies=instance.galaxies)

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
Fin.
"""
