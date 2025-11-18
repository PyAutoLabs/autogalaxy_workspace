"""
Tutorial 3: Graphical Benefits
==============================

In the previous tutorials, we fitted a dataset containing 3 galaxies which had a shared `sersic_index` value.

We used different approaches to estimate the shared `sersic_index`, for example a simple approach of fitting each
dataset one-by-one and estimating the Sersic index via a weighted average or posterior multiplication and a more
complicated approach using a graphical model.

The estimates were consistent with one another, making it hard to justify the use of the more complicated graphical
model. However, the model fitted in the previous tutorial was extremely simple, and by making it slightly more complex
in this tutorial we will be able to show the benefits of using the graphical modeling approach.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autogalaxy_workspace/scripts/simulators/imaging/samples/dev_exp.py`.

__The Model__

The more complex datasets and model fitted in this tutorial is an extension of those fitted in the previous tutorial.

Previously, there was only a bulge in each galaxy dataset which all had the same Sersic index.

In this tutorial, each dataset now contains a bulge and disk, where all bulges have `sersic_index=4` and all disks
`sersic_index=1.0`
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
__Dataset__

For each galaxy dataset in our sample we set up the correct path and load it by iterating over a for loop. 

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autogalaxy_workspace/scripts/simulators/imaging/samples/dev_exp.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "dev_exp"

dataset_path = Path("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = Path(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        ag.Imaging.from_fits(
            data_path=Path(dataset_sample_path, "data.fits"),
            psf_path=Path(dataset_sample_path, "psf.fits"),
            noise_map_path=Path(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__

We now mask each galaxy in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every galaxy in the dataset is appropriate.
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    masked_imaging_list.append(dataset)

"""
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = Path("imaging") / "graphical"

"""
__Model (one-by-one)__

We are first going to fit each dataset one by one.

We therefore fit a model where

 - The galaxy's bulge is a linear parametric `Sersic` bulge with its centre fixed to the input 
 value of (0.0, 0.0) [4 parameters]. 

 - The galaxy's disk is a linear parametric `Sersic` disk with its centre fixed to the input 
 value of (0.0, 0.0) [4 parameters]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

We require that the bulge Sersic index is between 3.0 and 6.0 and disk Sersic index 0.5 to 3.0 -- this ensures that
the model does not swap the two components and fit the bulge with the lower Sersic index component and visa versa.
"""
bulge = af.Model(ag.lp_linear.Sersic)
bulge.centre = (0.0, 0.0)
bulge.sersic_index = af.UniformPrior(lower_limit=3.0, upper_limit=6.0)

disk = af.Model(ag.lp_linear.Sersic)
disk.centre = (0.0, 0.0)
disk.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class. 
"""
analysis_list = []

for dataset_index, masked_dataset in enumerate(masked_imaging_list):
    #
    analysis = ag.AnalysisImaging(dataset=masked_dataset, use_jax=True)

    analysis_list.append(analysis)

"""
__Model Fits (one-by-one)__

For each dataset we now create a non-linear search, analysis and perform the model-fit using this model.

The `Result` is stored in the list `result_list` and they are output to a unique folder named using the `dataset_index`..
"""
result_list = []

for dataset_index, analysis in enumerate(analysis_list):
    dataset_name_with_index = f"dataset_{dataset_index}"
    path_prefix_with_index = Path(path_prefix, "tutorial_3_graphical_benefits")

    search = af.Nautilus(
        path_prefix=path_prefix_with_index, name=dataset_name_with_index, n_live=100
    )

    result = search.fit(model=model, analysis=analysis)
    result_list.append(result)

"""
__Sersic Index Estimates (Weighted Average)__

We can now compute the Sersic index estimate of both light profiles, including their errors, from the individual 
model fits performed above.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
mp_bulge_sersic_indexes = [
    instance.galaxies.galaxy.bulge.sersic_index for instance in mp_instances
]

ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_bulge_sersic_indexes = [
    instance.galaxies.galaxy.bulge.sersic_index for instance in ue1_instances
]
le1_bulge_sersic_indexes = [
    instance.galaxies.galaxy.bulge.sersic_index for instance in le1_instances
]

error_list = [
    ue1 - le1 for ue1, le1 in zip(ue1_bulge_sersic_indexes, le1_bulge_sersic_indexes)
]

values = np.asarray(mp_bulge_sersic_indexes)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

bulge_weighted_sersic_index = np.sum(values * weights) / np.sum(weights, axis=0)
bulge_weighted_error = 1.0 / np.sqrt(weight_averaged)


mp_instances = [samps.median_pdf() for samps in samples_list]
mp_disk_sersic_indexes = [
    instance.galaxies.galaxy.disk.sersic_index for instance in mp_instances
]

ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_disk_sersic_indexes = [
    instance.galaxies.galaxy.disk.sersic_index for instance in ue1_instances
]
le1_disk_sersic_indexes = [
    instance.galaxies.galaxy.disk.sersic_index for instance in le1_instances
]

error_list = [
    ue1 - le1 for ue1, le1 in zip(ue1_disk_sersic_indexes, le1_disk_sersic_indexes)
]

values = np.asarray(mp_disk_sersic_indexes)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

disk_weighted_sersic_index = np.sum(values * weights) / np.sum(weights, axis=0)
disk_weighted_error = 1.0 / np.sqrt(weight_averaged)


print(
    f"Weighted Average Bulge Sersic Index Estimate = {bulge_weighted_sersic_index} ({bulge_weighted_error}) [1.0 sigma confidence intervals]"
)
print(
    f"Weighted Average Disk Sersic Index Estimate = {disk_weighted_sersic_index} ({disk_weighted_error}) [1.0 sigma confidence intervals]"
)

"""
The estimate of the Sersic indexes are not accurate, with both estimates well offset from the input values 
of 4.0 and 1.0.

We will next show that the graphical model offers a notable improvement, but first lets consider why this
approach is suboptimag.

The most important difference between this model and the model fitted in the previous tutorial is that there are now
two shared parameters we are trying to estimate, *and they are degenerate with one another*.

We can see this by inspecting the probability distribution function (PDF) of the fit, placing particular focus on the 
2D degeneracy between the Sersic index of the bulge and disk.
"""
plotter = aplt.NestPlotter(samples=result_list[0].samples)
plotter.corner_cornerpy()

"""
The problem is that the simple approach of taking a weighted average does not capture the curved banana-like shape
of the PDF between the two Sersic indexes. This leads to significant error over estimation and biased inferences on the 
estimates.

__Discussion__

Let us now consider other downsides of fitting each dataset one-by-one, from a more statistical perspective. We 
will contrast these to the graphical model later in the tutorial.

1) By fitting each dataset one-by-one this means that each model-fit fails to fully exploit the information we know 
about the global model. We *know* that there are only two single shared values of `sersic_index` across the full dataset 
that we want to estimate. However, each individual fit has its own `sersic_index` value which is able to assume 
different values than the `sersic_index` values used to fit the other datasets. This means that the large degeneracies 
between the two Sersic indexes emerge for each model-fit.

By not fitting our model as a global model, we do not maximize the amount of information that we can extract from the 
dataset as a whole. If a model fits dataset 1 particularly bad, this *should* be reflected in how we interpret how 
well the model fits datasets 2 and 3. Our non-linear search should have a global view of how well the model fits the 
whole dataset. This is the *crucial aspect of fitting each dataset individually that we miss*, and what a graphical 
model addresses.

2) When we combined the result to estimate the global `sersic_index` value via a weighted average, we marginalized over 
the samples in 1D. As showed above, when there are strong degeneracies between models parameters the information on 
the covariance between these parameters is lost when computing the global `sersic_index`. This increases the inferred 
uncertainties. A graphical model performs no such 1D marginalization and therefore fully samples the
parameter covariances.

3) In Bayesian inference it is important that we define priors on all of the model parameters. By estimating the 
global `sersic_index` after the model-fits are completed it is unclear what prior the global `sersic_index` a
ctually has! We actually defined the prior five times -- once for each fit -- which is not a well defined prior.

In a graphical model the prior is clearly defined.

What would have happened if we had estimate the shared Sersic indexes via 2D posterior multiplication using a KDE? We
will discuss this at the end of the tutorial after fitting a graphical model.

__Model (Graphical)__

We now compose a graphical model and fit it.

Our model now consists of a galaxy with a bulge and disk, which each have a `sersic_index_shared_prior` variable, 
such that the same `sersic_index` parameters are used for the bulge and disks of all galaxies fitted to all datasets. 

We require that the bulge Sersic index is between 3.0 and 6.0 and disk Sersic index 0.5 to 3.0 -- this ensures that
the model does not swap the two components and fit the bulge with the lower Sersic index component and visa versa.
"""
bulge_sersic_index_shared_prior = af.UniformPrior(lower_limit=3.0, upper_limit=6.0)
disk_sersic_index_shared_prior = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

"""
We now set up a list of `Model`'s, each of which contain a bulge and disk.

All of these `Model`'s use the `sersic_index_shared_prior`'s above. This means all model-components use the same value 
of `sersic_index` for the bulge and same `sersic_index` values for the disk.

For a fit to three datasets (each using an `Sersic` bulge and disk), this produces a parameter space with
dimnensionality N=20 (8 parameters per pair of `Sersic` and 2 shared Sersic indexes).
"""
model_list = []

for model_index in range(total_datasets):
    bulge = af.Model(ag.lp_linear.Sersic)
    bulge.centre = (0.0, 0.0)

    # This makes every Galaxy bulge share the same `sersic_index`.
    bulge.sersic_index = bulge_sersic_index_shared_prior

    disk = af.Model(ag.lp_linear.Sersic)
    disk.centre = (0.0, 0.0)

    # This makes every Galaxy disk share the same `sersic_index`.
    disk.sersic_index = disk_sersic_index_shared_prior

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    model_list.append(model)

"""
__Analysis Factors__

We again create the graphical model using `AnalysisFactor` objects.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

The analysis factors are then used to create the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The factor graph model can again be printed via the `info` attribute, which shows that there are two shared
parameters across the datasets.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and use it to the fit the factor graph, again using its `global_prior_model` 
property.
"""
search = af.Nautilus(
    path_prefix=path_prefix,
    name="tutorial_3_graphical_benefits",
    n_live=250,
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same structure of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
We can now inspect the inferred `sersic_index` values and compare this to the values estimated above via a weighted 
average.  

(The errors of the weighted average is what was estimated for a run on my PC, yours may be slightly different!)
"""
bulge_sersic_index = result.samples.median_pdf()[0].galaxies.galaxy.bulge.sersic_index

u1_error_0 = result.samples.values_at_upper_sigma(sigma=1.0)[
    0
].galaxies.galaxy.bulge.sersic_index
l1_error_0 = result.samples.values_at_lower_sigma(sigma=1.0)[
    0
].galaxies.galaxy.bulge.sersic_index

u3_error_0 = result.samples.values_at_upper_sigma(sigma=3.0)[
    0
].galaxies.galaxy.bulge.sersic_index
l3_error_0 = result.samples.values_at_lower_sigma(sigma=3.0)[
    0
].galaxies.galaxy.bulge.sersic_index

disk_sersic_index = result.samples.median_pdf()[0].galaxies.galaxy.disk.sersic_index

u1_error_1 = result.samples.values_at_upper_sigma(sigma=1.0)[
    0
].galaxies.galaxy.disk.sersic_index
l1_error_1 = result.samples.values_at_lower_sigma(sigma=1.0)[
    0
].galaxies.galaxy.disk.sersic_index

u3_error_1 = result.samples.values_at_upper_sigma(sigma=3.0)[
    0
].galaxies.galaxy.disk.sersic_index
l3_error_1 = result.samples.values_at_lower_sigma(sigma=3.0)[
    0
].galaxies.galaxy.disk.sersic_index

print(
    f"Weighted Average Bulge Sersic Index Estimate = 3.035967168057999 (0.020862051618561108) [1.0 sigma confidence intervals]\n"
)
print(
    f"Weighted Average Disk Sersic Index Estimate = 1.0034699385233146 (0.011400000233187503) [1.0 sigma confidence intervals]"
)

print(
    f"Inferred value of the bulge Sersic index via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{bulge_sersic_index} ({l1_error_0} {u1_error_0}) ({u1_error_0 - l1_error_0}) [1.0 sigma confidence intervals]"
)
print(
    f"{bulge_sersic_index} ({l3_error_0} {u3_error_0}) ({u3_error_0 - l3_error_0}) [3.0 sigma confidence intervals]"
)

print(
    f"Inferred value of the disk Sersic index via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{disk_sersic_index} ({l1_error_1} {u1_error_1}) ({u1_error_1 - l1_error_1}) [1.0 sigma confidence intervals]"
)
print(
    f"{disk_sersic_index} ({l3_error_1} {u3_error_1}) ({u3_error_1 - l3_error_1}) [3.0 sigma confidence intervals]"
)

"""
As expected, using a graphical model allows us to infer a more precise and accurate model.

You may already have an idea of why this is, but lets go over it in detail:

__Discussion__

Unlike a fit to each dataset one-by-one, the graphical model:

1) Infers a PDF on the global Sersic index that fully accounts for the degeneracies between the models fitted to 
different datasets. This reduces significantly the large 2D degeneracies between the two Sersic indexes we saw when 
inspecting the PDFs of each individual fit.

2) Fully exploits the information we know about the global model, for example that the Sersic index of every galaxy 
in every dataset is aligned. Now, the fit of the galaxy in dataset 1 informs the fits in datasets 2 and 3, and visa 
versa.

3) Has a well defined prior on the global Sersic index, instead of independent priors on the Sersic index of each 
dataset.

__Posterior Multiplication__

What if we had combined the results of the individual model fits using 2D posterior multiplication via a KDE?

This would produce an inaccurate estimate of the error, because each posterior contains the prior on the Sersic index 
multiple times which given the properties of this model should not be repeated.

However, it is possible to convert each posterior to a likelihood (by dividing by its prior), combining these
likelihoods to form a joint likelihood via 2D KDE multiplication and then insert just one prior back (agian using a 2D
KDE) at the end to get a posterior which does not have repeated priors. 

This posterior, in theory, should be equivalent to the graphical model, giving the same accurate estimates of the
Sersic indexes with precise errors. The process extracts the same information, fully accounting for the 2D structure 
of the PDF between the two Sersic indexes for each fit.

However, in practise, this will likely not work that well. Every time we use a KDE to represent and multiply a 
posterior, we make an approximation which will impact our inferred errors. The removal of the prior before combining 
the likelihood and reinserting it after also introduces approximations, especially because the fit performed by the 
non-linear search is informed by the prior. 

Crucially, whilst posterior multiplication maybe sort-of-works-ok in two dimensions, for models with many more 
dimensions and degeneracies between parameters that are in 3D, 4D of more dimensions it simply does not work.

In contrast, a graphical model fully samples all of the information a large dataset contains about the model, without
making an approximations. In this sense, irrespective of how complex the model gets, it will fully extract the 
information contained in the dataset.

__Wrap Up__

In this tutorial, we demonstrated the strengths of a graphical model over fitting each dataset one-by-one. 

We argued that irrespective of how one may try to combine the results of many individual fits, the approximations that 
are made will always lead to a suboptimal estimation of the model parameters and fail to fully extract all information
from the dataset. 

Furthermore, we argued that for high dimensional complex models a graphical model is the only way to fully extract
all of the information contained in the dataset.

In the next tutorial, we will consider a natural extension of a graphical model called a hierarchical model.
"""
