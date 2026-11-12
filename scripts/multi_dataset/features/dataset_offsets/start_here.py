"""
Features: Dataset Offsets
=========================

Align observations within a joint galaxy fit so that small pointing or registration differences are not mistaken for changes in morphology. The relative coordinate shifts are fitted alongside the shared physical model.

__Scientific Use Cases__

- Combine observations with residual astrometric misalignment.
- Separate a shift of the whole image from a change in the galaxy's structure.
- Inspect the uncertainty in relative registration when comparing bands.

__This Example__

Choose one dataset as the positional reference and fit the other dataset's relative translations.
Retain the recipe's coordinate order, angular units and priors. A translation does not correct a
rotation, a PSF mismatch or genuine wavelength-dependent morphology.

![Dataset Offsets](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__Relative Alignment__

An offset describes how one dataset is placed relative to a reference. The reference choice is part of the
model; otherwise a common shift can be confused with moving the object itself.

Offsets should be distinguished from differences in PSF, pixel scale or morphology between observations.
Correcting one translation does not account for every possible discrepancy between datasets.

__Following the Example__

The simulator establishes a known offset and the modeling example shows which dataset provides the reference
and which offsets are fitted. Retain the coordinate order, units and priors alongside the code. After fitting,
inspect each dataset in its own coordinates as well as considering the joint result.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
