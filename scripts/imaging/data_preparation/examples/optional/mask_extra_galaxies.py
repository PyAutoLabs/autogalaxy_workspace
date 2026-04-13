"""
Data Preparation: Extra Galaxies Mask (Optional)
================================================

There may be regions of an image that have signal near the galaxy that is from other galaxies not associated with the
main galaxy we are studying. The emission from these images will impact our model fitting and needs to be removed from
the analysis.

This script creates a mask of these regions of the image, called the `mask_extra_galaxies`, which can be used to
prevent them from impacting a fit. This mask may also include emission from objects which are not technically galaxies,
but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
or emission due to the data reduction process.

The mask can be applied in different ways. For example, it could be applied such that the image pixels are discarded
from the fit entirely, Alternatively the mask could be used to set the image values to (near) zero and increase their
corresponding noise-map to large values.

The exact method used depends on the nature of the model being fitted. For simple fits like a light profile a mask
is appropriate, as removing image pixels does not change how the model is fitted. However, for more complex models
fits, like those using a pixelization, masking regions of the image in a way that removes their image pixels entirely
from the fit can produce discontinuities in the pixelixation. In this case, scaling the data and noise-map values
may be a better approach.

This script outputs a `mask_extra_galaxies.fits` file, which can be loaded and used before a model fit, in whatever
way is appropriate for the model being fitted.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_mask.ipynb` shows how to use a Graphical User Interface (GUI) to create
the extra galaxies mask.

The script `features/extra_galaxies/modeling` shows how to use this mask in different ways in a model-fit.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.

__Contents__

**Links / Resources:** Links to GUI tools and modeling scripts for extra galaxies masks.
**Output:** Saving the extra galaxies mask as a FITS file.
"""

# from autoconf import setup_notebook; setup_notebook()

