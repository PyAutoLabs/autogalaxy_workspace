"""
Features: Sky Background
========================

Measure faint galaxy outskirts while fitting an uncertain sky level alongside the galaxy light. This is useful for retained backgrounds and residual sky-subtraction offsets, where a small background error can otherwise alter the inferred size or outer profile.

__Scientific Use Cases__

- Fit the mean sky level rather than attributing it to extended galaxy emission.
- Assess the sensitivity of galaxy sizes and outskirts to background assumptions.
- Compare masks and priors that constrain the sky–galaxy decomposition.

__This Example__

The example fits a spatially constant background. Its simulated sky is 5.0 electrons per second, while
the modeling prior spans 0.0 to 5.0: the injected value is at the upper boundary, not inside a surrounding
interval. Choose an appropriate prior for your own data. Fitting the mean does not remove sky photon noise.

![Sky Background](../../../../images/features/sky_background.png)

Conceptual illustration: the separated layers are additive light and background components, not physical planes in space.

__Why the Outskirts Are Sensitive__

Bright central light can be easy to distinguish from the background, while faint extended emission may be
comparable to it. A change in the assumed sky level can therefore change the inferred outer profile even when
the central fit appears satisfactory.

The background and the extended light model should be considered together. A flexible galaxy model may absorb
an incorrect background, while a freely fitted background can absorb genuine low-surface-brightness emission.

__Using the Examples__

The simulator establishes the background used in the demonstration. The direct fit and modeling examples show
how it enters the analysis and which parameters are fixed or inferred. Keep the data units, mask and
background parameterization explicit when adapting them.

Inspect residuals across the fitted region and consider how strongly the available background pixels constrain
the sky. The example's chosen prior and field of view are part of its setup, not universal choices for every
image.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
