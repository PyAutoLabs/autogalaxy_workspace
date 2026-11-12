"""
Features: Operated Light Profile
================================

Fit compact radio or submillimeter emission together with an extended galaxy directly in visibility space. The operated-profile class can be used for the compact component, but its imaging PSF-bypass interpretation does not apply to this likelihood.

__Scientific Use Cases__

- Separate compact and extended sky emission through their predicted visibilities.
- Keep the compact-component model explicit when comparing imaging and interferometer observations.
- Avoid confusing an imaging PSF template with intrinsic sky brightness or a synthesized dirty-image beam.

__This Example__

Here the compact profile is a sky-brightness model that is Fourier transformed into visibilities like
other light profiles. There is no imaging PSF convolution to skip. The synthesized beam of a dirty image
is not an additional blur to apply before that transform. The imaging operated-profile interpretation
therefore cannot be carried over unchanged.

![Operated Light Profile](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__From Compact Sky Light to Visibilities__

For this example, the profile image represents compact sky emission and is Fourier transformed at the measured
visibility coordinates. The fitted compact width is not automatically an imaging PSF width. The data must
support any claim that the compact structure is resolved.

An imaging operated profile instead describes light after a specified PSF operation. Sharing a profile class
between datasets therefore does not justify sharing its fitted size or response interpretation without checking
the physical model.

__Compact Emission and Extended Light__

The compact component and host contribute jointly to the predicted visibilities. Their separation depends on
the visibility coverage, noise and adopted light profiles. Inspect the per-component predictions and the fit,
not just a displayed dirty image. The modeling script also describes a separate advanced visibility-matrix
override; that is not required for the ordinary profile transformation illustrated here.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
