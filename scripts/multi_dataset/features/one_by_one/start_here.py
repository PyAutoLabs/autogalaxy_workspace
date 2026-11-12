"""
Features: One By One
====================

Use a well-constrained observation to guide modeling of another band. Sequential fitting is useful for organizing a multi-band decomposition and testing which galaxy properties can be held fixed, transferred as priors or fitted again.

__Scientific Use Cases__

- Establish a starting morphology from one observation before fitting another.
- Explore the effect of fixing a component while refitting the rest of the galaxy.
- Fit a relative dataset offset with an earlier light model held fixed.

__This Example__

The example first fits one dataset, then fixes its bulge to the earlier result while carrying the
disk model forward for refitting. A further fit keeps the galaxy model fixed and varies the relative
dataset offset. These are illustrative transfers: fixing an earlier estimate does not propagate its
full uncertainty into the later fit.

![One By One](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__What Is Carried Forward?__

A result can inform initial values, priors or fixed parameters. These choices have different consequences:
fixing a parameter to an earlier estimate does not propagate its uncertainty in the same way as a joint fit.

Read precisely what the example transfers and what it fits again. The order of the datasets can matter when
early decisions constrain the later analysis.

__When to Compare with a Joint Fit__

Sequential fitting can make a complicated analysis easier to organize, particularly when one observation
supplies a useful starting model. If the datasets offer complementary constraints on shared parameters, a
simultaneous analysis may be needed to represent that information together.

The modeling script keeps its transfer and refitting steps explicit so you can follow how the later result
depends on the earlier one.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
