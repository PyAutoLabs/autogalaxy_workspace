"""
Interferometer: Data Preparation
================================

When an interferometer dataset is analysed, it must conform to certain standards in order for
the analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts
which will help you prepare your dataset to adhere to them if it does not already.

__Contents__

**SLACK:** Contact information for help with interferometer data preparation.
**Pixel Scale:** Choosing the correct pixel scale for interferometer datasets.
**Visibilities:** Loading and inspecting visibility data from FITS files.
**Noise-Map:** Loading and inspecting the noise map for the interferometer dataset.
**UV Wavelengths:** Loading and inspecting the uv-wavelength baselines.
**Real Space Mask:** Setting up the real-space mask for Fourier transform evaluation.
**Data Processing Complete:** Summary of required data standards and overview of optional steps.
**Light Centre (Optional):** Marking the galaxy light centre to fix or constrain model parameters.
**Extra Galaxies (Optional):** Marking centres of nearby extra galaxies for inclusion in the model.
**Mask Extra Galaxies (Optional):** Creating masks to remove signal from nearby extra galaxies.
**Info (Optional):** Storing auxiliary information like redshifts as a JSON file.

__SLACK__

The interferometer data preparation scripts are currently being developed and are not yet complete. If you are
unsure of how to prepare your dataset, please message us on Slack and we will help you directly!

__Pixel Scale__

When fitting an interferometer dataset, the images of the lens  galaxies are first evaluated in real-space
using a grid of pixels, which is then Fourier transformed to the uv-plane.

The "pixel_scale" of an interferometer dataset is this pixel-units to arcsecond-units conversion factor. The value
depends on the instrument used to observe the lens, the wavelength of the light used to observe it and size of
the baselines used (e.g. longer baselines means higher resolution and therefore a smaller pixel scale).

The pixel scale of some common interferometers is as follows:

 - ALMA: 0.02" - 0.1" / pixel
 - JVLA: 0.005" - 0.01" / pixel

It is absolutely vital you use a sufficently small pixel scale that all structure in the data is resolved after the
Fourier transform. If the pixel scale is too large, the Fourier transform will smear out the data and the model.
"""

# from autoconf import setup_notebook; setup_notebook()

