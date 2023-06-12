"""
Tutorial 6: Science Case
========================

This tutorial shows a realistic science case. 

We have a dataset containing 10 galaxies, each of which are made of an `Sersic` bulge and `Sersic` disk, where:

 - The `sersic_index` of each bulge is drawn from a parent hierarchical Gaussian distribution with `mean=4.0`
 and `sigma=2.0`,

 - The `sersic_index` parameters of the disks are drawn from an independent parent Gaussian distribution with
 `mean=1.0` and `sigma=1.0`. 

This tutorial fits this dataset using expectation propagation (EP) in order to infer the parameters of both parent
hierarchical distributions.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autofit as af
from os import path

"""
__Initialization__

The following steps repeat all the initial steps performed in the previous tutorials.
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "sersic_sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 5

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        ag.Imaging.from_fits(
            data_path=path.join(dataset_sample_path, "data.fits"),
            psf_path=path.join(dataset_sample_path, "psf.fits"),
            noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    masked_imaging_list.append(dataset.apply_mask(mask=mask))

"""
__Paths__
"""
path_prefix = path.join("imaging", "graphical")

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data.

This graphical model creates a non-linear parameter space that has parameters for every galaxy in our sample. In this
example, there are 10 galaxies each with their own galaxy model, therefore:

 - Each galaxy's bulge is a parametric linear  `Sersic` with its centre fixed to the input 
 value of (0.0, 0.0) [4 parameters]. 

 - Each galaxy's disk is a parametric linear `Sersic` with its centre fixed to the input 
 value of (0.0, 0.0) [4 parameters]. 

 - There are ten galaxies in our graphical model [10 x 8 parameters]. 

The overall dimensionality of each parameter space fitted separately via EP is therefore N=8.

In total, the graph has N = 10 x 8 = 80 free parameters, albeit EP knows the `sersic_index` parameters are drawn from
hierarchical distributions and uses this information in the model fit.
"""
model_list = []

for model_index in range(total_datasets):
    bulge = af.Model(ag.lp_linear.Sersic)
    bulge.centre = (0.0, 0.0)

    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    bulge.effective_radius = af.GaussianPrior(
        mean=3.0, sigma=3.0, lower_limit=0.0, upper_limit=10.0
    )
    bulge.sersic_index = af.GaussianPrior(
        mean=4.0, sigma=3.0, lower_limit=0.5, upper_limit=10.0
    )

    disk = af.Model(ag.lp_linear.Sersic)
    disk.centre = (0.0, 0.0)
    disk.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
    )
    disk.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
    )
    disk.effective_radius = af.GaussianPrior(
        mean=3.0, sigma=3.0, lower_limit=0.0, upper_limit=10.0
    )
    disk.sersic_index = af.GaussianPrior(
        mean=1.0, sigma=3.0, lower_limit=0.5, upper_limit=10.0
    )

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    model_list.append(model)

"""
__Analysis__
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = ag.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s, just like we did in the
previous tutorial.
"""
dynesty = af.DynestyStatic(
    path_prefix=path.join("imaging", "graphical"),
    name="tutorial_6_science_case",
    nlive=100,
    sample="rwalk",
)

analysis_factor_list = []
dataset_index = 0

for model, analysis in zip(model_list, analysis_list):
    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=dynesty, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
__Model__

We now compose the hierarchical model components that we fit.
"""
hierarchical_factor_bulge = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.5, upper_limit=10.0),
    sigma=af.GaussianPrior(mean=5.0, sigma=5.0, lower_limit=0.0, upper_limit=10.0),
)

hierarchical_factor_disk = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.5, upper_limit=10.0),
    sigma=af.GaussianPrior(mean=5.0, sigma=5.0, lower_limit=0.0, upper_limit=10.0),
)

for model in model_list:
    hierarchical_factor_bulge.add_drawn_variable(
        model.galaxies.galaxy.bulge.sersic_index
    )
    hierarchical_factor_disk.add_drawn_variable(model.galaxies.galaxy.disk.sersic_index)

"""
We again combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = af.FactorGraphModel(
    *analysis_factor_list, hierarchical_factor_bulge, hierarchical_factor_disk
)

"""
The factor graph model `info` attribute shows the complex model we are fitting, including both hierarchical
factors.
"""
print(factor_graph.global_prior_model.info)

"""
__Expectation Propagation__

We perform the fit using EP as we did in tutorial 5.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(name=path.join(path_prefix, "tutorial_6_science_case"))

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
)

"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/graphical/imaging/tutorial_6_science_case`. 
"""
print(factor_graph_result)

print(factor_graph_result.updated_ep_mean_field.mean_field)

"""
__Output__

The MeanField object representing the posterior.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field)
print()

print(factor_graph_result.updated_ep_mean_field.mean_field.variables)
print()

"""
The logpdf of the posterior at the point specified by the dictionary values
"""
# factor_graph_result.updated_ep_mean_field.mean_field(values=None)
print()

"""
A dictionary of the mean with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.mean)
print()

"""
A dictionary of the variance with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.variance)
print()

"""
A dictionary of the s.d./variance**0.5 with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.scale)
print()

"""
self.updated_ep_mean_field.mean_field[v: Variable] gives the Message/approximation of the posterior for an 
individual variable of the model.
"""
# factor_graph_result.updated_ep_mean_field.mean_field["help"]

"""
Finish.
"""
