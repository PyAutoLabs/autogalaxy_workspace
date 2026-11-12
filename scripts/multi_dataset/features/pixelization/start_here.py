"""
Features: Pixelization
======================

Compare galaxy morphology across bands without forcing every observation into the same analytic light profile. A joint pixelized model reconstructs brightness separately for each dataset, with regularization controlling how much structure each observation supports.

__Scientific Use Cases__

- Map irregular light rather than forcing it into a smooth elliptical profile.
- Separate a smooth component from more complex emission when both are included in the model.
- Inspect and export the reconstructed galaxy light, checking which structure is supported by the data.

__This Example__

The worked model fits a pixelized galaxy in each dataset, using a rectangular mesh construction and
a separate regularization coefficient for each analysis. Each dataset solves its own brightnesses. A
common mesh description does not force identical brightness or colour. The single-band imaging example
provides the related smooth-bulge-plus-irregular-light construction.

![Pixelization](../../../../images/features/pixelization_illustration.png)

Conceptual illustration, not a fitted reconstruction. The two views represent the same galaxy brightness; the schematic triangular overlay is not a claim about the mesh used in every example.

__From a Mesh to the Data__

The reconstructed light must be mapped into the observable being fitted. Imaging examples include the PSF;
interferometer examples compare the prediction with visibilities through the Fourier response.

A reconstructed pixel is therefore not simply copied from a data pixel. Its value is inferred through the
forward model, and neighbouring reconstructed pixels may be constrained by overlapping parts of the data.

__Why Regularization Is Needed__

A flexible mesh can describe structure, but it can also follow noise. Regularization expresses a preference
for relationships between reconstructed pixels, such as a smoother distribution of light. The regularization
strength controls the balance between flexibility and that preference.

A reconstruction with a visually smooth appearance is not automatically the right one. Too much regularization
can suppress real structure; too little can introduce features driven by noise. The mesh, regularization and
residuals should be considered together.

__Choosing the Mesh__

The number and placement of mesh pixels affect both the structures the model can describe and the
computational cost. Adaptive meshes concentrate resolution where it is useful, while different mesh
constructions imply different interpolation and regularization choices.

The examples retain these choices alongside the code. Read the fixed-model fit first, then the modeling
example; specialist examples explore other meshes, adaptive calculations or measurements from the
reconstruction.

__Interpreting the Reconstruction__

Check the fit in the space where the data were measured as well as inspecting the reconstructed light. A
plausible reconstruction can still accompany an inadequate PSF, an unsuitable mask or an incomplete physical
model. Fine structure should be interpreted in light of the resolution and assumptions of the full analysis.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/pixelization/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
