"""
Features: Shapelets
===================

Describe departures from a smooth galaxy profile with a flexible set of spatial basis functions. Shapelets are useful for studying asymmetric morphology and structured residuals while retaining a compact parameterization of the basis geometry.

__Scientific Use Cases__

- Represent light that a single symmetric profile cannot describe adequately.
- Explore how basis order and scale affect the recovered morphology.
- Compare flexible light models while inspecting the summed brightness and residuals.

__This Example__

The fitted data are complex visibilities. The galaxy-light model is transformed into the sampled
Fourier quantities, with the dataset's visibility noise retained. A dirty image helps with inspection,
but it is not the image-plane dataset used by the likelihood.

![Shapelets](../../../../images/features/shapelets_illustration.png)

Reused conceptual illustration. “Source model” here means the emitting galaxy; there is no gravitational-lensing source plane. The summed basis represents light, not separate physical structures. The image illustrates sky emission; the likelihood fits complex visibilities.

__Basis Functions and Their Sum__

An individual basis function is a mathematical building block, not a separate physical component of the
galaxy. Its appearance should therefore not be interpreted as the appearance of a star-forming region, disk or
other structure by itself. The combined light distribution is the quantity to compare with the observations.

The basis order and scale determine the range of structures that can be represented. Increasing the number of
functions adds flexibility and computational work; it does not by itself establish that the additional
structure is constrained by the data.

__What the Examples Teach__

The direct-fit example introduces the basis and its coefficients. The modeling example shows how the basis is
embedded in the wider model and which parameters are varied. Keep the basis definition, coefficient
constraints and any regularization in view when interpreting the result.

As with other flexible light models, inspect the residuals and the reconstructed image together. A useful
basis should describe the signal without introducing structure solely to follow noise.

__Basis Teaching Figure__

![Shapelet basis grid](../../../../images/features/shapelets_basis.png)

Retain this basis grid for the detailed lesson: individual patterns are mathematical functions; the galaxy model is their weighted sum.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/shapelets/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/shapelets/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
