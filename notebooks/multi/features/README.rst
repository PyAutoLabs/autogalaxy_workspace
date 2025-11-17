The ``advanced/modeling`` folder contains example scripts showing how to analyse multiple datasets simultaneously.

This combines all other dataset types, so it could be multple imaging datasets (e.g. multi wavelength) observations,
imaging and interferometry, etc.

Notes
-----

The ``multi`` package extends the ``imaging`` package and readers should refer to the ``imaging`` package for
descriptions of how to customize the non-linear search, the fit settings, etc.

These scripts show how to perform lens modeling but only give a brief overview of how to analyse
and interpret the results a lens model fit. A full guide to result analysis is given at ``autolens_workspace/*/results``.

Files (Advanced)
----------------

The following example illustrate multi-dataset lens modeling with features that are specific to having multiple datasets:

- ``dataset_offsets``: Datasets may have small offsets due to pointing errors, which can be accounted for in the model.
- ``imaging_and_interferometer``: Imaging and interferometer datasets are fitted simultaneously.
- ``one_by_one``: Multiple datasets are fitted one-by-one in a sequence.
- ``same_wavelength``: Multiple datasets that are observed at the same wavelength are fitted simultaneously.
- ``wavelength_dependence``: A model is fitted where parameters depend on wavelength following a functional form.
- ``pixelization``: The source is reconstructed using an adaptive Voronoi mesh.