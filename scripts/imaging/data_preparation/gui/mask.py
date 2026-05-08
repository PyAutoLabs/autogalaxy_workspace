"""
GUI Preprocessing: Mask
=======================

This tool allows one to mask a bespoke mask for a given image of a galaxy using an interactive GUI. This mask
can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
search (if a mask function is not passed to that search).

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab

__Contents__

- **Dataset:** Loading the imaging dataset for mask creation.
- **Scribbler:** Using the interactive GUI to draw the mask.
- **Output:** Saving the mask as a FITS file and PNG visualization.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import numpy as np

"""
__Dataset__

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` dataset, so that the mask can be plotted over the galaxy image.
"""
data = ag.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

"""
__Scribbler__

Load the Scribbler GUI for drawing the mask. 

Push Esc when you are finished drawing the mask.
"""
scribbler = ag.Scribbler(image=data.native)
mask = scribbler.show_mask()
mask = ag.Mask2D(mask=np.invert(mask), pixel_scales=pixel_scales)

"""
__Output__

Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
aplt.plot_array(array=data, title="Data")

"""
Output this image of the mask to a .png file in the dataset folder for future reference.
"""
aplt.plot_array(
    array=data,
    title="Data",
    output_path=dataset_path,
    output_filename="mask_gui",
    output_format="png",
)

"""
Output it to the dataset folder of the galaxy, so that we can load it from a .fits in our modeling scripts.
"""
aplt.fits_array(
    array=mask, file_path=Path(dataset_path, "mask_gui.fits"), overwrite=True
)
