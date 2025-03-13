The ``advanced/modeling`` folder contains example scripts showing how to fit a galaxy model to multiple imaging datasets:

Notes
-----

The ``multi`` package extends the ``imaging`` package and readers should refer to the ``imaging`` package for
descriptions of how to customize the non-linear search, the fit settings, etc.

These scripts show how to perform galaxy modeling but only give a brief overview of how to analyse
and interpret the results a galaxy model fit. A full guide to result analysis is given at ``autogalaxy_workspace/*/imaging/results``.

Files (Advanced)
----------------

The following example scripts illustrating multi-dataset galaxy modeling where:

- ``no_galaxy_light.py``: The foreground galaxy's light is not present in the data and thus omitted from the model.
- ``linear_light_profiles.py``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``pixelization.py``: The source is reconstructed using an adaptive Voronoi mesh.
- ``imaging_and_interferometer.py``: Imaging and interferometer datasets are fitted simultaneously.
- ``same_wavelength.py`: Multiple datasets that are observed at the same wavelength are fitted simultaneously.
- ``wavelength_dependence.py``: A model is fitted where parameters depend on wavelength following a functional form.
