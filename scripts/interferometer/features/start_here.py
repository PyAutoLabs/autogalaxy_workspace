"""
Interferometer Features
=======================

These features build on the topic’s main example. Each introduction explains the method and its trade-offs; the worked scripts retain their model choices and detailed calculations.

__Choose a Feature__

- [Extra Galaxies](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/start_here.ipynb): A target rarely occupies an otherwise empty field. Nearby galaxies may contribute light inside the region being fitted, and a lensing model may also need to account for their deflection of the light rays.
- [Linear Light Profiles](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/start_here.ipynb): A linear light profile has the same spatial form as an ordinary light profile, but its intensity is solved during each likelihood evaluation instead of being sampled by the non-linear search.
- [Multi Gaussian Expansion](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/start_here.ipynb): A multi Gaussian expansion (MGE) represents a galaxy's light as the sum of many Gaussian profiles. Each Gaussian describes emission on a particular spatial scale; together, they can provide a flexible model of the galaxy.
- [Operated Light Profile](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/start_here.ipynb): An operated light profile describes emission after an instrumental operation has already been included in the profile. For imaging, the common example is a profile that already describes the PSF-blurred appearance of compact emission.
- [Pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/pixelization/start_here.ipynb): A pixelization represents light using a mesh whose brightness values are solved during the fit. It provides a flexible alternative to describing all of the emission with a small collection of analytic profiles.
- [Shapelets](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/shapelets/start_here.ipynb): Shapelets describe light as a weighted combination of basis functions. Unlike a single smooth light profile, the basis contains functions with different spatial structures, allowing their sum to describe more complicated galaxy morphology.

For the common setup, begin with [interferometer/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/start_here.ipynb). Run Python scripts from the workspace root.
"""
