"""
Features: Imaging And Interferometer
====================================

Compare the stellar light and radio or submillimeter emission of the same galaxy by fitting imaging and interferometer observations together. Shared parameters connect the models, while each dataset retains the calculation appropriate to its measured quantities.

__Scientific Use Cases__

- Compare complementary tracers of galaxy structure across instruments.
- Link selected geometric parameters while allowing the emission to differ.
- Inspect whether a shared model describes both image pixels and complex visibilities.

__This Example__

The imaging likelihood uses PSF-convolved image pixels; the interferometer likelihood fits complex
visibilities. Link parameters only where a shared structure is justified: optical and radio/submillimeter
emission need not have identical sizes, shapes or centroids. Inspect each fit separately.

![Imaging And Interferometer](../../../../images/new_user_guide/dataset_types.png)

Reused PyAutoGalaxy new-user-guide schematic, not measured data. The interferometer panel shows visibility sampling locations, not amplitudes; multi-band panels illustrate changing emission.

__Two Measurements of a Shared System__

Imaging compares a PSF-blurred model with image pixels. Interferometry compares a Fourier-domain prediction
with visibilities. A displayed interferometer image is useful for inspection, but does not replace the
visibility likelihood used by these examples.

Shared geometry does not require identical emission. Different wavelengths may trace different components of a
galaxy, so decide which parameters should be linked and which should remain dataset-specific.

__Reading the Model__

Follow the construction of each dataset's analysis and then their combination. The example retains its actual
shared parameters, light components and instrumental settings. Inspect the fit to each dataset separately: an
acceptable joint score can conceal a poor description of one observation.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
