The ``simulators/advanced`` folder contains example scripts for simulating imaging datasets (E.g. Hubble Space Telescope).

Folders
-------

- ``samples``: Examples for simulating multiple galaxy imaging datasets.

Files (Beginner)
----------------

- ``sersic.py``: A galaxy which is an elliptical Sersic bulge.
- ``sersic_x2.py``: Two galaxies which are each an elliptical Sersic bulge.
- ``complex.py``: A complex galaxy with a bulge, disk and star forming clumps.
- ``asymmetric.py``: A galaxy which has asymmetric emission using a basis of 14 elliptical Gaussians.
- ``extra_galaxies.py``: A galaxy where the emission of extra galaxies is present in the image and needs removing or modelling.
- ``manual_signal_to_noise_ratio``: Simulate galaxies using light profiles where the signal-to-noise of their lensed images are input.
- ``operated.py``: A galaxy which is an elliptical Sersic bulge, elliptical exponential disk and point source of emission at its centre which appears as the PSF.
- ``sky_background.py``: A galaxy with a sky background which is not subtracted from the image.