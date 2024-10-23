The ``modeling/features`` folder contains example scripts showing how to fit a model to imaging data:

Files (Beginner)
----------------

The following example scripts illustrating modeling where:

- ``linear_light_profiles.py``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``multi_gaussian_expansion.py``: The galaxy light is modeled as ~25-100 Gaussian basis functions to capture asymmetric morphological features.
- ``shapelets.py``: Using shapelet basis functions for the galaxy light.
- ``extra_galaxies.py``: Modeling which account for the light and mass of extra nearby galaxies.
- ``operated_light_profiles.py``: There are light profiles which are assumed to already be convolved with the instrumental PSF (e.g. point sources), commonly used for modeling bright AGN in the centre of a galaxy.
- ``pixelization.py``: The galaxy is reconstructed using a rectangular pixel grid.
- ``sky_background.py``: Including the background sky in the model.