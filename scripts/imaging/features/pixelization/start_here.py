"""
Features: Pixelization
======================

Reconstruct galaxy light that is difficult to describe with a small set of analytic profiles, such as asymmetric or clumpy emission. A pixelization solves brightness on a mesh, allowing the morphology of the emission to be studied while accounting for the observation's response and noise.

__Scientific Use Cases__

- Map irregular light rather than forcing it into a smooth elliptical profile.
- Separate a smooth component from more complex emission when both are included in the model.
- Inspect and export the reconstructed galaxy light, checking which structure is supported by the data.

__This Example__

These examples fit image pixels after accounting for the PSF and noise. The inferred components
describe galaxy light in the observed field. Examine their individual contributions and the residuals,
not only whether the sum gives an acceptable fit.

For galaxy science, continue to [galaxy_reconstruction.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/galaxy_reconstruction.ipynb) to inspect and export the inferred light.

The main modeling example combines a smooth bulge with a pixelized component for irregular light.
The decomposition is a model assumption; examine what flux each component absorbs.

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

- [fit.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/simulator.ipynb) — Generate the example dataset.
- [galaxy_reconstruction.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/galaxy_reconstruction.ipynb) — Export the reconstructed galaxy light.
- [plot.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/plot.ipynb) — Inspect the available visualizations.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
