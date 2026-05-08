"""
GUI Preprocessing: Extra Galaxies Mask (Optional)
=================================================

There may be regions of an image that have signal near the galaxy that is from other galaxies not associated with the
main galaxy we are studying. The emission from these images will impact our model fitting and needs to be removed from
the analysis.

The example `data_preparation/imaging/example/optional/extra_galaxies_mask.py` provides a full description of
what the extra galaxies are and how they are used in the model-fit. You should read this script first before
using this script.

This script uses a GUI to mark the regions of the image where these extra galaxies are located, in contrast to the
example above which requires you to input these values manually.

__Contents__

- **Dataset:** Loading the imaging dataset for extra galaxy mask creation.
- **Mask:** Creating a circular mask overlay to guide the spray painting.
- **Scribbler:** Using the interactive GUI to spray paint extra galaxy regions.
- **Output:** Saving the extra galaxies mask as a FITS file and PNG visualization.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import numpy as np

"""
__Dataset__

The path where the extra galaxy mask is output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` data, so that the location of galaxies is clear when scaling the noise-map.
"""
data = ag.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

data = ag.Array2D(
    values=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), mask=data.mask
)

"""
__Mask__

Create a 3.0" mask to plot over the image to guide where extra galaxy light needs its emission removed and noise scaled.
"""
mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

"""
__Scribbler__

Load the Scribbler GUI for spray painting the scaled regions of the dataset.

Push Esc when you are finished spray painting.
"""
scribbler = ag.Scribbler(image=data.native, mask_overlay=mask)
mask = scribbler.show_mask()
mask = ag.Mask2D(mask=mask, pixel_scales=pixel_scales)

"""
The GUI has now closed and the extra galaxies mask has been created.

Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

"""
__Output__

Output to a .png file for easy inspection.
"""
aplt.plot_array(
    array=data,
    title="Data",
    output_path=dataset_path,
    output_filename="data_mask_extra_galaxies",
    output_format="png",
)

"""
The new image is plotted for inspection.
"""
aplt.plot_array(array=data, title="Data")

"""
Plot the data with the new mask, in order to check that the mask removes the regions of the image corresponding to the
extra galaxies.
"""
aplt.plot_array(array=data, title="Data")

"""
__Output__

Output the extra galaxies mask, which will be load and used before a model fit.
"""
aplt.fits_array(
    array=mask,
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"),
    overwrite=True,
)
