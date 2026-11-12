"""
Features: Wavelength Dependence
===============================

Measure how galaxy structure changes across bands by linking selected profile parameters through a wavelength-dependent relation. This supports comparisons of bulge and disk morphology while limiting the number of independently sampled parameters.

__Scientific Use Cases__

- Fit a smooth trend of galaxy size with wavelength.
- Connect bands that constrain the same structural components with different precision.
- Test whether the chosen wavelength relation describes the observed morphology in every band.

__This Example__

The recipe links the bulge and disk effective radii through separate straight-line functions of
wavelength, fitting their slopes and intercepts. Other structural choices remain explicit in the model;
linear-profile intensities are solved for each dataset. A radius trend alone does not identify the
stellar populations or dust distribution that caused it.

![Wavelength Dependence](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__A Relationship Is a Model Assumption__

A parameter can be shared across all wavelengths, fitted separately in each band, or expressed through a
function of wavelength. These are different models with different amounts of freedom.

A smooth relationship may reduce the number of free parameters and connect weakly constrained bands, but it
also restricts the changes the model can describe. The observations must still support the adopted form.

__Follow the Linked Parameters__

The example explains which properties vary, how their wavelength dependence is expressed and which
coefficients are fitted. Retain these details beside the code; “multi-wavelength” does not by itself specify a
unique linking scheme.

Inspect the predicted morphology and residuals in every band. Differences between bands may reflect the light
distribution, instrumental response or an inadequate relationship, and the fit should be interpreted
accordingly.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
