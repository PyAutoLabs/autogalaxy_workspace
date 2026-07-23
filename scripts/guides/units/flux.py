"""
Flux
====

Absolute flux calibration in Astronomy is the process of converting the number of photons detected by a telescope into
a physical unit of luminosity or a magnitude. For example, a luminosity might be given in units of solar luminosities
or the brightness of a galaxy quoted as a magnitude in units of AB magnitudes.

The conversion of a light profile, that has been fitted to data, to physical units can be non-trivial, as careful
consideration must be given to the units that are involved.

The key quantity is the `intensity` of the light profile, the units of which match the units of the data that is fitted.
For example, if the data is in units of electrons per second, the intensity will also be in units of electrons per
second per pixel.

The conversion of this intensity to a physical unit, like solar luminosities, therefore requires us to make a number
of conversion steps that go from electrons per second to the desired physical unit or magnitude.

This guide gives example conversions for units commonly used in astronomy, such as converting the intensity of a
light profile from electrons per second to solar luminosities or AB magnitudes. Once we have values in a more standard
unit, like a solar luminosity or AB magnitude, it becomes a lot more straightforward to follow Astropy tutorials
(or other resources) to convert these values to other units or perform calculations with them.

__Contents__

- **Zero Point:** Explanation of photometric zero points and their role in flux calibration.
- **Mega Janskys / steradian (MJy/sr): James Webb Space Telescope:** Convert JWST NIRCam light profile intensities to AB magnitudes.
- **Latent Variables:** Reading the same total flux directly from the `latent.csv` of a completed fit.

__Zero Point__

In astronomy, a zero point refers to a reference value used in photometry and spectroscopy to calibrate the brightness
of celestial objects. It sets the baseline for a magnitude system, allowing astronomers to compare the brightness of
different stars, galaxies, or other objects.

For example, the zero point in a photometric system corresponds to the magnitude that a standard star (or a theoretical
object) would have if it produced a specific amount of light at a particular wavelength. It provides a way to convert
the raw measurements of light received by a telescope into meaningful values of brightness (magnitudes).

The conversions below all require a zero point, which is typically provided in the documentation of the telescope or
instrument that was used to observe the data.
"""
# ENV: full_datasets
# Guides load committed full-resolution FITS; SMALL_DATASETS would
# mismatch the pre-existing 100x100 data shape.

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Mega Janskys / steradian (MJy/sr): James Webb Space Telescope__

James Webb Space Telescope (JWST) NIRCam data is often provided in units of Mega Janskys per steradian (MJy/sr).
We therefore show how to convert the intensity of a light profile to MJy/sr.

This calculation is well documented in the JWST documentation, and we are following the steps in the following
webpage:

https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints#gsc.tab=0

First, we need a light profile, which we'll assume is a Sersic profilee. If you're analyzing real JWST data, you'll
need to use the light profile that was fitted to the data.
"""
light = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,  # in units of MJy/sr
    effective_radius=0.1,
    sersic_index=3.0,
)

"""
According to the document above, flux density in MJy/sr can be converted to AB magnitude using the following formula:

 mag_AB = -6.10 - 2.5 * log10(flux[MJy/sr]*PIXAR_SR[sr/pix] ) = ZP_AB – 2.5 log10(flux[MJy/sr])

Where ZP_AB is the zeropoint:  

 ZP_AB = –6.10 – 2.5 log10(PIXAR_SR[sr/pix]). 

For example, ZP_AB = 28.0 for PIXAR_SR = 2.29e-14 (corresponding to pixel size 0.0312").

For data in units of MJy/sr, computing the total flux that goes into the log10 term is straightforward, it is
simply the sum of the image of the light profile. 

We compute this using a grid, which must be large enough that all light from the light profile is included. Below,
we use a grid which extends to 10" from the centre of the light profile, which is sufficient for this example,
but you may need to increase this size for your own data.
"""
grid = ag.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = light.image_2d_from(grid=grid)

total_flux = np.sum(image)

"""
We now convert this total flux to an AB magnitude using the zero point of the JWST NIRCam filter we are analyzing.

As stated above, the zero point is given by:

 ZP_AB = –6.10 – 2.5 log10(PIXAR_SR[sr/pix])
 
Where the value of PIXAR_SR is provided in the JWST documentation for the filter you are analyzing. 

The Pixar_SR values for JWST (James Webb Space Telescope) NIRCam filters refer to the pixel scale in steradians (sr) 
for each filter, which is a measure of the solid angle covered by each pixel. These values are important for 
calibrating and understanding how light is captured by the instrument.

For the F444W filter, which we are using in this example, the value is 2.29e-14 (corresponding to a pixel size o
f 0.0312").
"""
pixar_sr = 2.29e-14

zero_point = -6.10 - 2.5 * np.log10(pixar_sr)

magnitude_ab = zero_point - 2.5 * np.log10(total_flux)

"""
__Latent Variables: Total Flux Directly from the Fit__

The example above computed the total flux by hand: build a light profile, sample it on a grid, sum the image, then
apply the zero point. PyAutoGalaxy does exactly this automatically as part of every fit and records the result as
a latent variable in the `latent/samples.csv` file beside the search output. You can skip the manual recipe
entirely and just read the column.

The raw-flux latent ships default-on (it needs no instrument inputs and runs on every fit unless disabled in
`config/latent.yaml`):

- `total_galaxy_0_flux` — total integrated flux of the first galaxy (`fit.galaxies[0]`), in the *raw* image
  units the fit was performed in. For JWST data in MJy/sr, this is MJy/sr; for HST data in e- s^-1, this is
  e- s^-1.

To convert this to AB magnitudes or microjanskies, apply the same zero-point recipe used above. Suppose you
have a JWST F444W fit and want the AB magnitude of the galaxy; reading the column from your result and
converting goes:
"""
from autogalaxy.imaging.model.latent import (
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
)

# Stand-in for what you'd read from `latent.csv` — in a real script this is one column of one row, e.g.
#   total_galaxy_0_flux = pd.read_csv(search.paths.output_path / "latent" / "samples.csv")["total_galaxy_0_flux"].iloc[-1]
total_galaxy_0_flux = 1234.5  # MJy/sr

# JWST F444W zero-point computed exactly as in the MJy/sr section above.
ab_mag_galaxy = ab_mag_via_flux_from(flux=total_galaxy_0_flux, magzero=zero_point)
flux_mujy_galaxy = flux_mujy_via_ab_mag_from(ab_mag=ab_mag_galaxy)

"""
The two helpers used above are the same ones the library uses internally to populate the `_mujy` variant of the
latent (`total_galaxy_0_flux_mujy`). That variant is default-off because it needs a `magzero` you supply per
instrument. If you have a single fixed zero-point you can flip it on by:

1. Setting `total_galaxy_0_flux_mujy: true` in your project's `config/latent.yaml`.
2. Passing `magzero=<value>` when constructing the analysis:
   `analysis = ag.AnalysisImaging(dataset=dataset, magzero=zero_point)`.

The latent dispatcher then writes the converted µJy column into `latent/samples.csv` directly, so you don't
have to run the conversion in post. If you enable the `_mujy` latent but forget the `magzero` keyword, the
column is populated with NaN and a single warning per process notes that the conversion was skipped — the fit
itself is unaffected.

Finish.
"""
