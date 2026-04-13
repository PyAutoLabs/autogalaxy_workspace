"""
Data Preparation: Noise-map
===========================

The noise-map defines the uncertainty in every pixel of your galaxy image, where values are defined as the
RMS standard deviation in every pixel (not the variances, HST WHT-map values, etc.).

You MUST be certain that the noise-map is the RMS standard deviations or else your analysis will be incorrect!

This tutorial describes preprocessing your dataset`s noise-map to adhere to the units and formats required
by **PyAutoGalaxy**.

__Pixel Scale__

The "pixel_scale" of the image (and the data in general) is pixel-units to arcsecond-units conversion factor of
your telescope. You should look up now if you are unsure of the value.

The pixel scale of some common telescopes is as follows:

 - Hubble Space telescope 0.04" - 0.1" (depends on the instrument and wavelength).
 - James Webb Space telescope 0.06" - 0.1" (depends on the instrument and wavelength).
 - Euclid 0.1" (Optical VIS instrument) and 0.2" (NIR NISP instrument).
 - VRO / LSST 0.2" - 0.3" (depends on the instrument and wavelength).
 - Keck Adaptive Optics 0.01" - 0.03" (depends on the instrument and wavelength).

It is absolutely vital you use the correct pixel scale, so double check this value!

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.

__Contents__

**Loading Data From Individual Fits Files:** Loading a noise-map from FITS files and inspecting its standards.
**1) Tools Illustrated In Image:** Overview of unit conversion and resizing tools from the image preparation script.
**Noise Conversions:** Functions for computing noise-maps from various input formats.
"""

# from autoconf import setup_notebook; setup_notebook()

