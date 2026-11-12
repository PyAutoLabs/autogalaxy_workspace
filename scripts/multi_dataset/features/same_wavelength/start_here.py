"""
Features: Same Wavelength
=========================

Combine repeated observations of a galaxy to constrain one common light model without discarding the PSF and noise information of each exposure. This is useful when several images observe the same emission with different instrumental conditions.

__Scientific Use Cases__

- Use multiple observations to constrain shared galaxy structure.
- Retain each exposure's PSF and noise in its own forward calculation.
- Check for exposure-specific residuals that a combined fit might conceal.

__This Example__

The common galaxy model is evaluated separately through each dataset's response. Check exactly which
parameters are shared, including flux calibration and positional choices. Repeated exposures are not
identical copies, and the same band does not by itself establish identical calibration.

![Same Wavelength](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__One Model, Several Forward Calculations__

The observations need not have the same PSF, sampling or signal-to-noise. Each analysis predicts what the
common model would look like through its own measurement process, rather than treating all images as identical
copies.

Whether additional quantities such as normalizations or offsets should be shared depends on the dataset and
the model. The example's construction shows its actual choices.

__Inspect Every Observation__

The combined likelihood brings the measurements together, but residuals should still be examined for each
observation. An instrumental mismatch in one exposure may otherwise be hidden by a strong fit to another.

Use the simulator to see the differences introduced between the observations, then follow how those
differences are handled in the modeling script.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
