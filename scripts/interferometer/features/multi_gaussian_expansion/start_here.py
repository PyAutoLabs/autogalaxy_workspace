"""
Features: Multi Gaussian Expansion
==================================

Describe a galaxy's central light and extended envelope with a flexible sum of Gaussian profiles. An MGE is useful when one simple profile leaves structured residuals, or when you need a smooth light model from which to measure the combined luminosity and radial structure.

__Scientific Use Cases__

- Model light over a wide range of radii without assigning every Gaussian to a separate physical component.
- Allow more flexible radial structure than a single elliptical Sersic, within the chosen component geometry.
- Measure integrated properties from the combined light distribution and examine what remains in the residuals.

__This Example__

The fitted data are complex visibilities. The galaxy-light model is transformed into the sampled
Fourier quantities, with the dataset's visibility noise retained. A dirty image helps with inspection,
but it is not the image-plane dataset used by the likelihood.

![Multi Gaussian Expansion](../../../../images/features/mge_illustration.png)

Conceptual illustration: the displayed Gaussians are components of one light model. The inset is a decomposition, not a spatial magnification of the galaxy core. The image illustrates sky emission; the likelihood fits complex visibilities.

__Why Use Many Gaussians?__

A simple light profile, such as an elliptical Sersic, describes a galaxy using a small number of parameters.
This makes it useful and relatively easy to interpret, but its restricted shape may leave structured residuals
when fitted to a more complex galaxy.

For example, a galaxy's ellipticity or orientation may change with radius. A single elliptical profile cannot
describe all of those changes.

An MGE combines Gaussians with different widths and, depending on the model, different shapes. Narrow
Gaussians describe central structure, while broader Gaussians describe more extended emission. Their
combination can represent radial changes that a single profile misses.

The flexibility depends on how the components are arranged. Giving every Gaussian the same centre and shape
still imposes restrictions; adding more Gaussians does not automatically describe every possible galaxy
morphology.

__How Are the Intensities Fitted?__

For fixed Gaussian widths, centres and shapes, the model image depends linearly on the Gaussian intensities.
We can therefore solve for those intensities during each likelihood evaluation, instead of treating them as
separate parameters in the non-linear search.

The fitting process is:

1. The non-linear search proposes the model's remaining parameters.
2. The corresponding Gaussian images are calculated, including the instrumental response.
3. A linear solve determines the Gaussian intensities.
4. The resulting model is compared with the data.

An MGE containing 60 Gaussians therefore does not require the search to sample 60 additional intensity
parameters.

The number of non-linear parameters depends on the choices you make: which widths are fixed, which centres are
shared, and which shapes are allowed to vary. The modeling example explains its choices alongside the code.

__Positive Intensities__

The examples constrain Gaussian intensities to be non-negative.

Without this constraint, overlapping Gaussians can acquire large positive and negative intensities that cancel
one another. Such a combination may fit the data, but its individual components no longer represent positive
contributions to the galaxy's light.

Non-negative intensities avoid this cancellation. They do not, however, guarantee that the model is adequate:
you should still inspect the residuals and consider whether the chosen components can describe the observed
structure.

__Computational and Scientific Trade-offs__

Evaluating many Gaussian components and solving their intensities adds work to each likelihood calculation. On
the other hand, reducing the number of sampled parameters can make the non-linear search easier. The balance
depends on the dataset and model; an MGE is not guaranteed to make every fit faster.

An MGE is also less directly interpretable than a single Sersic profile. An individual Gaussian is usually a
component of the mathematical description, rather than a separate physical structure in the galaxy. Quantities
such as total luminosity or a half-light radius should be derived from the combined light distribution.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/likelihood_function.ipynb) — Follow the likelihood calculation.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
