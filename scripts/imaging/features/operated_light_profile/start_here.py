"""
Features: Operated Light Profile
================================

Separate compact nuclear emission from an extended host using a profile whose imaging PSF response is already included. This is useful when the component you want to fit describes the observed compact glow, rather than an intrinsic light distribution that still needs to be blurred.

__Scientific Use Cases__

- Fit PSF-like nuclear light alongside an extended galaxy model.
- Prevent a compact component that already includes the PSF from being convolved twice.
- Assess how the compact-light description affects host-galaxy measurements.

__This Example__

These examples fit image pixels after accounting for the PSF and noise. The inferred components
describe galaxy light in the observed field. Examine their individual contributions and the residuals,
not only whether the sum gives an acceptable fit.

![Operated Light Profile](../../../../images/features/operated_light_profile.png)

Conceptual illustration of the imaging response: the compact component already includes its PSF blur. The extended galaxy is still treated with its appropriate response.

__Avoid Applying the Same Response Twice__

Ordinary intrinsic light profiles are passed through the instrumental response before comparison with the
data. An operated profile is already defined at that stage for the relevant operation. Applying the same PSF
convolution again would broaden it a second time.

This distinction is about the meaning of the profile, not simply whether a source is small. Decide whether
your parameters describe intrinsic emission or its observed appearance before choosing the profile class.

__Compact Emission and Extended Light__

Compact nuclear emission can overlap an extended host galaxy. The examples show how those components are
combined and which instrumental response is already represented. Their decomposition depends on the adopted
compact-profile shape as well as the host model.

Read the simulator and modeling example together. Imaging and interferometer responses are different, so
retain the data-type-specific explanation in the recipe rather than applying an imaging PSF assumption to
every dataset.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
