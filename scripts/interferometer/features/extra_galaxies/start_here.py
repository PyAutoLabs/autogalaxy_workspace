"""
Features: Extra Galaxies
========================

Measure a target galaxy's light in a crowded field by modeling the neighbouring galaxies whose emission overlaps the fitted region. Separating those contributions helps protect the target's inferred size, shape and flux from contamination.

__Scientific Use Cases__

- Fit overlapping galaxy light instead of assigning all emission to the target.
- Measure morphology and component fluxes in blended systems.
- Test how the treatment of faint companions changes the target fit.

__This Example__

The fitted data are complex visibilities. The galaxy-light model is transformed into the sampled
Fourier quantities, with the dataset's visibility noise retained. A dirty image helps with inspection,
but it is not the image-plane dataset used by the likelihood.

![Extra Galaxies](../../../../images/new_user_guide/galaxy_systems.png)

Reused PyAutoGalaxy new-user-guide schematic. These are galaxy-light environments, not lensing mass models; the blended-galaxy panel is the relevant comparison here. The image illustrates sky emission; the likelihood fits complex visibilities.

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

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
