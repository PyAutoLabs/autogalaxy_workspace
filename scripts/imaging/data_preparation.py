"""
Imaging: Data Preparation
=========================

When a CCD imaging dataset is analysed, it must conform to certain standards in order for the
analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts which
will help you prepare your dataset to adhere to them if it does not already.

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

__Contents__

**Pixel Scale:** Overview of the pixel-to-arcsecond conversion factor for common telescopes.
**Image:** Standards for the galaxy image, including units, centering and stamp size.
**Noise Map:** Standards for the RMS noise-map, including units and values.
**PSF:** Standards for the Point Spread Function, including size, oddness, normalization and centering.
**Data Processing Complete:** Summary of required standards and introduction to optional steps.
**Mask (Optional):** Creating custom masks tailored to the galaxy emission.
**Light Centre (Optional):** Marking the galaxy light centre for use as a fixed model parameter.
**Extra Galaxies (Optional):** Marking centres of nearby extra galaxies for modeling or masking.
**Mask Extra Galaxies (Optional):** Creating a mask to remove extra galaxy emission from the analysis.
**Info (Optional):** Storing auxiliary information about the dataset as a JSON file.
"""

# from autoconf import setup_notebook; setup_notebook()

