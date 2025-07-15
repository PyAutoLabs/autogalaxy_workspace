"""
Multipoles
==========

This guide shows how to perform ellipse fitting and modeling with multipoles modeling.

__Previous Examples__

Ellipse fitting with multipoles is the most advanced form of ellipse fitting, therefore it is recommended that you
are familiar with regular ellipse fitting before reading this example.

To ensure this is the case, make sure you've complted the examples `fit.py` and `modeling.py`.

For brevity, this example does not repeat the description of the API used to perform the fit and how the model is
composed. It only discusses new aspects of the API that are used to perform multipoles modeling.
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
__Loading Data__

We we begin by loading the galaxy dataset `simple` from .fits files, which is the same dataset we fitted in the
previous examples.
"""
dataset_name = "ellipse"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True, noise_map=True)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Mask the data and retain its radius to set up the ellipses in the model fitting.
"""
mask_radius = 5.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.set_title("Image Data With Mask Applied")
dataset_plotter.figures_2d(data=True)

"""
__Multipole Fit__

We have seen that we can create and fit an ellipse to the data as follows:
"""
ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse)

fit_plotter = aplt.FitEllipsePlotter(
    fit_list=[fit], mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
To perform ellipse fitting with multipoles, we simply create an `EllipseMultipole` object and pass it to 
the `FitEllipse` object along with the dataset and ellipse.

We create a fourth order multipole, which quadrupole perturbations to the ellipse. This makes the ellipse
appear more boxy and is a common feature of real galaxies.
"""
multipole_order_4 = ag.EllipseMultipole(m=4, multipole_comps=(0.05, 0.05))

fit_multipole = ag.FitEllipse(
    dataset=dataset, ellipse=ellipse, multipole_list=[multipole_order_4]
)

"""
Up to now, the ellipses plotted over the data in white have always been ellipses.

When a multipole is included in the fit, it perturbs the ellipse to create a more complex shape that departs
from an ellipse. 

This is shown by the white lines in the figure below, which because the multipole is a quadrupole, show a
boxy shape.
"""
fit_plotter = aplt.FitEllipsePlotter(
    fit_list=[fit_multipole], mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
__Multipole Order__

Multipoles of different order can be combined to create even more complex shapes, for example:

 - An `m=1` multipole creates a monopole which represents lopsidedness in the galaxy.
 - An `m=3` multipole creates a tripole which represents a galaxy with a 3 fold symmetry.

We include both these multipoles below, in addition to the `m=4` quadrupole, create a complex perturbation to the
ellipse.
"""
multipole_order_1 = ag.EllipseMultipole(m=1, multipole_comps=(0.05, 0.05))
multipole_order_3 = ag.EllipseMultipole(m=3, multipole_comps=(0.05, 0.05))

fit_multipole = ag.FitEllipse(
    dataset=dataset,
    ellipse=ellipse,
    multipole_list=[multipole_order_1, multipole_order_3, multipole_order_4],
)

fit_plotter = aplt.FitEllipsePlotter(
    fit_list=[fit_multipole], mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
__Multiple Perturbed Ellipses__

The API above can be combined with lists to fit many ellipses with many multipoles, allowing for the most complex
shapes to be fitted to the data.
"""
number_of_ellipses = 10

major_axis_list = np.linspace(0.3, mask_radius * 0.9, number_of_ellipses)

fit_list = []

for i in range(len(major_axis_list)):
    ellipse = ag.Ellipse(
        centre=(0.0, 0.0), ell_comps=(0.3, 0.5), major_axis=major_axis_list[i]
    )

    fit = ag.FitEllipse(
        dataset=dataset,
        ellipse=ellipse,
        multipole_list=[multipole_order_1, multipole_order_3, multipole_order_4],
    )

    fit_list.append(fit)

fit_plotter = aplt.FitEllipsePlotter(
    fit_list=fit_list, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
fit_plotter.figures_2d(data=True)

"""
__Modeling__

We now perform model-fitting via a non-linear search to perform ellipse fitting with multipoles.

First, we set up the `ellipses` using identical code to the `modeling.py` example.

This begins by performing a model fit with one ellipse to the centrral regions of the data, in order to determine
the centre of all ellipses.
"""
ellipse = af.Model(ag.Ellipse)

ellipse.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

ellipse.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
ellipse.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

ellipse.major_axis = 0.3

model = af.Collection(ellipses=[ellipse])

"""
We now set up a third and fourth order multipole component and add it as a model component to all 10 ellipses.

The model is composed such that only N=2 free parameters are fitted for each multipole, as the same multipole amplitudes
are used for every ellipse. 

This is a common assumption when fitting multipoles, although there are also studies showing that multipoles can
vary radially over galaxies, which would require a more complex model.
"""
multipole_list = []

multipole_3_a = af.GaussianPrior(mean=0.0, sigma=0.1)
multipole_3_b = af.GaussianPrior(mean=0.0, sigma=0.1)

multipole_4_a = af.GaussianPrior(mean=0.0, sigma=0.1)
multipole_4_b = af.GaussianPrior(mean=0.0, sigma=0.1)

multipole_3 = af.Model(ag.EllipseMultipole)
multipole_3.m = 3
multipole_3.multipole_comps.multipole_comps_0 = multipole_3_a
multipole_3.multipole_comps.multipole_comps_1 = multipole_3_b

multipole_4 = af.Model(ag.EllipseMultipole)
multipole_4.m = 4
multipole_4.multipole_comps.multipole_comps_0 = multipole_4_a
multipole_4.multipole_comps.multipole_comps_1 = multipole_4_b

multipole_list.append([multipole_3, multipole_4])

"""
Create the model, which is a `Collection` of `Ellipses` and `Multipole` components.
"""
model = af.Collection(ellipses=[ellipse], multipoles=multipole_list)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. 

Everything below uses the same API introduced in the `modeling.py` example.
"""
search = af.DynestyStatic(
    path_prefix=Path("ellipse_multipole"),
    name=f"fit_start",
    unique_tag=dataset_name,
    sample="rwalk",
    n_live=50,
    number_of_cores=4,
    iterations_per_update=10000,
)

"""
__Analysis__

Create the `AnalysisEllipse` object.
"""
analysis = ag.AnalysisEllipse(dataset=dataset)

"""
__Run Times__

When only ellipses are fitted, the run time of the likelihood function was ~ 0.04 seconds.

The inclusion of a multipole component slightly increases the run time of the likelihood function, but it
is almost negligible.

This is because perturbing the ellipse with a multipole is a simple operation that does not require significant
computation time.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
The biggest increase in run time when fitting multipoles is because the number of free parameters in the model
increases, as well as the complexity of the model and parameter space.

We estimate the overall run time of the model-fit below, noting that it generally still stays well below an hour
and is therefore feasible to perform on a laptop.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

The simulated galaxy image contained in the data fitted in this example did not include multipoles, therefore
the multipole components go to values close to zero.
"""
print(result.info)

"""
The `Result` object also the maximum log likelihood instance which can be inspected to check the 
inferred multipole parameters.
"""
instance = result.max_log_likelihood_instance

print("Max Log Likelihood Model:")
print(instance)

print(
    f"First Ellipse Multipole Components: {instance.multipoles[0][0].multipole_comps}"
)

"""
The maximum log likelihood fit is also available via the result, which can visualize the fit.
"""
fit_plotter = aplt.FitEllipsePlotter(
    fit_list=result.max_log_likelihood_fit_list,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
fit_plotter.figures_2d(data=True)

"""
The result contains the full posterior information of our non-linear search, including all parameter samples, 
log likelihood values and tools to compute the errors on the model. 

When multipoles are included in the model, the parameter space complexity increases, producing more
significant degeneracies between the model parameters.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Multiple Ellipses__
"""
number_of_ellipses = 10

major_axis_list = np.linspace(0.3, mask_radius * 0.9, number_of_ellipses)

total_ellipses = len(major_axis_list)

result_list = []

for i in range(len(major_axis_list)):
    ellipse = af.Model(ag.Ellipse)

    ellipse.centre.centre_0 = result.instance.ellipses[0].centre[0]
    ellipse.centre.centre_1 = result.instance.ellipses[0].centre[1]

    ellipse.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
    ellipse.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

    ellipse.major_axis = major_axis_list[i]

    multipole_list = []

    multipole_3_a = af.GaussianPrior(mean=0.0, sigma=0.1)
    multipole_3_b = af.GaussianPrior(mean=0.0, sigma=0.1)

    multipole_4_a = af.GaussianPrior(mean=0.0, sigma=0.1)
    multipole_4_b = af.GaussianPrior(mean=0.0, sigma=0.1)

    multipole_3 = af.Model(ag.EllipseMultipole)
    multipole_3.m = 3
    multipole_3.multipole_comps.multipole_comps_0 = multipole_3_a
    multipole_3.multipole_comps.multipole_comps_1 = multipole_3_b

    multipole_4 = af.Model(ag.EllipseMultipole)
    multipole_4.m = 4
    multipole_4.multipole_comps.multipole_comps_0 = multipole_4_a
    multipole_4.multipole_comps.multipole_comps_1 = multipole_4_b

    multipole_list.append([multipole_3, multipole_4])

    model = af.Collection(ellipses=[ellipse], multipoles=multipole_list)

    search = af.DynestyStatic(
        path_prefix=Path("ellipse_multipole"),
        name=f"fit_{i}",
        unique_tag=dataset_name,
        sample="rwalk",
        n_live=50,
        number_of_cores=4,
        iterations_per_update=10000,
    )

    analysis = ag.AnalysisEllipse(dataset=dataset)

    result = search.fit(model=model, analysis=analysis)

    result_list.append(result)

"""
__Final Fit__

A final fit is performed combining all ellipses.
"""
ellipses = [result.instance.ellipses[0] for result in result_list]
multipole_list = [result.instance.multipoles[0] for result in result_list]

model = af.Collection(ellipses=ellipses, multipoles=multipole_list)

model.dummy_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

search = af.Drawer(
    path_prefix=Path("ellipse_multipole"),
    name=f"fit_all",
    unique_tag=dataset_name,
    total_draws=1,
)

result = search.fit(model=model, analysis=analysis)


"""
__Masking__
"""
mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
)

dataset = dataset.apply_mask(mask=mask + mask_extra_galaxies)


number_of_ellipses = 10

major_axis_list = np.linspace(0.3, mask_radius * 0.9, number_of_ellipses)

total_ellipses = len(major_axis_list)

result_list = []

for i in range(len(major_axis_list)):
    ellipse = af.Model(ag.Ellipse)

    ellipse.centre.centre_0 = result.instance.ellipses[0].centre[0]
    ellipse.centre.centre_1 = result.instance.ellipses[0].centre[1]

    ellipse.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
    ellipse.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

    ellipse.major_axis = major_axis_list[i]

    multipole_list = []

    multipole_3_a = af.GaussianPrior(mean=0.0, sigma=0.1)
    multipole_3_b = af.GaussianPrior(mean=0.0, sigma=0.1)

    multipole_4_a = af.GaussianPrior(mean=0.0, sigma=0.1)
    multipole_4_b = af.GaussianPrior(mean=0.0, sigma=0.1)

    multipole_3 = af.Model(ag.EllipseMultipole)
    multipole_3.m = 3
    multipole_3.multipole_comps.multipole_comps_0 = multipole_3_a
    multipole_3.multipole_comps.multipole_comps_1 = multipole_3_b

    multipole_4 = af.Model(ag.EllipseMultipole)
    multipole_4.m = 4
    multipole_4.multipole_comps.multipole_comps_0 = multipole_4_a
    multipole_4.multipole_comps.multipole_comps_1 = multipole_4_b

    multipole_list.append([multipole_3, multipole_4])

    model = af.Collection(ellipses=[ellipse], multipoles=multipole_list)

    search = af.DynestyStatic(
        path_prefix=Path("ellipse_multipole_mask"),
        name=f"fit_{i}",
        unique_tag=dataset_name,
        sample="rwalk",
        n_live=50,
        number_of_cores=4,
        iterations_per_update=10000,
    )

    analysis = ag.AnalysisEllipse(dataset=dataset)

    result = search.fit(model=model, analysis=analysis)

    result_list.append(result)

ellipses = [result.instance.ellipses[0] for result in result_list]
multipole_list = [result.instance.multipoles[0] for result in result_list]

model = af.Collection(ellipses=ellipses, multipoles=multipole_list)

model.dummy_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

search = af.Drawer(
    path_prefix=Path("ellipse_multipole_mask"),
    name=f"fit_all",
    unique_tag=dataset_name,
    total_draws=1,
)

result = search.fit(model=model, analysis=analysis)

"""
This script gives a concise overview of the ellipse fitting modeling API with multipole components.

You should now be able to perform complex ellipse fitting with multipoles, which are a powerful tool to model
the shapes of real galaxies.
"""
