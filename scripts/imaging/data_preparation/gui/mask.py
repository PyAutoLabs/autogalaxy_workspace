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
- **Refining An Existing Mask:** Reopen a saved mask as a proposal and add to / erase from it.
- **Output:** Saving the mask as a FITS file and PNG visualization.
"""

# from autogalaxy import setup_notebook; setup_notebook()

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

Load the Scribbler GUI for drawing the mask, painting over the region of the image you want to fit.

Two brushes are available: press `1` for the green brush, which ADDS pixels to the painted region, and `2` for the red
brush, which ERASES them. Press `=` / `-` to make the brush bigger / smaller (each press scales it by 1.4x), `z` to
undo the last stroke and Esc when you are finished.

`mask_from()` returns everything painted green that was not painted red.
"""
scribbler = ag.Scribbler(image=data.native)
mask = scribbler.mask_from()
mask = ag.Mask2D(mask=np.invert(mask), pixel_scales=pixel_scales)

"""
__Refining An Existing Mask__

To adjust a mask drawn previously instead of starting from a blank image, load it and pass it to the GUI as a
`proposal`. Its boundary is outlined in white over the image, and `mask_from()` then returns the proposal plus
whatever you paint green, minus whatever you paint red.

The `.fits` written at the end of this script stores the region to *exclude* (`True` = masked), so it is inverted
back to the painted region before being passed as the proposal, and inverted again afterwards. Set
`refine_existing = True` to use this instead of the blank-canvas draw above.
"""
refine_existing = False

mask_path = Path(dataset_path, "mask_gui.fits")

if refine_existing and mask_path.exists():
    previous = ag.Mask2D.from_fits(file_path=mask_path, pixel_scales=pixel_scales)
    scribbler = ag.Scribbler(
        image=data.native, proposal=np.invert(np.asarray(previous))
    )
    mask = ag.Mask2D(mask=np.invert(scribbler.mask_from()), pixel_scales=pixel_scales)

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
