"""
Data Preparation: Image
=======================

The image is the image of your galaxy, which comes from a telescope like the Hubble Space telescope (HST).

This tutorial describes preprocessing your dataset`s image to adhere to the units and formats required by PyAutoGalaxy.

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

**Loading Data From Individual Fits Files:** Loading an image from FITS files and inspecting its standards.
**Converting Data To Electrons Per Second:** Converting image flux units between electrons per second, counts and ADUs.
**Resizing Data:** Trimming or padding a large postage stamp to an appropriate size.
**Background Subtraction:** Overview of background sky subtraction tools and modeling approaches.
"""

# from autoconf import setup_notebook; setup_notebook()

