"""
Features: Extra Galaxies
========================

A target rarely occupies an otherwise empty field. Nearby galaxies may contribute light inside the region
being fitted.

__Deciding What to Include__

The central concern is how overlapping emission affects the decomposition of the target.

The examples make the relevant choice for their data type. They retain the positions, light profiles and
parameter constraints beside the model-building code; these should not be transferred to a different field
without inspection.

__Separate the Contributions__

A good fit to the combined image does not guarantee that the light has been assigned correctly to each galaxy.
Inspect individual model components and residuals, especially where their emission overlaps. Fixing or sharing
parameters can help constrain a fit, but also introduces assumptions about the components.

Use the simulator to understand the example field, then follow the modeling example to add the neighbours
explicitly.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
galaxy light is modeled in the observed field; there is no lens-source plane mapping. Retain the particular
light-component choices stated in each recipe.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
