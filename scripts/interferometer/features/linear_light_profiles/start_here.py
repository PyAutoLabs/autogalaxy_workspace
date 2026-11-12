"""
Features: Linear Light Profiles
===============================

Fit galaxy structure with fewer sampled parameters by solving component brightnesses at every likelihood evaluation. This is especially useful for bulge–disk decomposition and models containing many overlapping light components: the search explores their geometry while their intensities are solved together.

__Scientific Use Cases__

- Measure bulge and disk structure without sampling a separate intensity for each component.
- Build richer light models while controlling the dimensionality of the non-linear search.
- Re-solve component fluxes as sizes, shapes and centres change; solved brightness is not fixed brightness.

__This Example__

The fitted data are complex visibilities. The galaxy-light model is transformed into the sampled
Fourier quantities, with the dataset's visibility noise retained. A dirty image helps with inspection,
but it is not the image-plane dataset used by the likelihood.

![Linear Light Profiles](../../../../images/features/linear_light_profiles.png)

Conceptual illustration: the paths symbolize different sampled parameter spaces, not measured convergence rates or removal of physical degeneracies. The image illustrates sky emission; the likelihood fits complex visibilities.

__What Is Linear About It?__

Once a profile's centre, shape and size are specified, changing its intensity simply scales its image. The
image of several profiles is a weighted sum of their individual images. This allows their intensities to be
determined together by linear algebra, accounting for the data and noise.

The other model parameters still need to be fitted. At each proposed geometry, the profile images are
recalculated and their intensities are solved again. Solving brightness therefore does not mean fixing it, and
the best intensities depend on the rest of the model.

__Why This Changes the Search__

With ordinary profiles, the search explores brightness as well as shape. With linear profiles, brightness is
determined conditionally at every step, reducing the number of sampled parameters. This is especially useful
when a model contains many light components.

The calculation still has a cost, and the total run time depends on both the likelihood calculation and the
search. The examples show the construction and solved intensities so that you can inspect what changed, rather
than treating the linear solve as a black box.

__Reading the Examples__

Start with the direct fit to see how the solved intensities are retrieved. Then read the modeling example to
see which parameters remain in the non-linear model. The likelihood example explains the underlying
calculation in more detail.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/likelihood_function.ipynb) — Follow the likelihood calculation.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
