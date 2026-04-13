import numpy as np

"""
Data Preparation: Extra Galaxies (Optional)
===========================================

There may be extra galaxies nearby the main galaxy, whose emission blends with the galaxy.

We can include these extra galaxies in the model as light profiles using the modeling API, where these nearby
objects are denoted `extra_galaxies`.

This script marks the (y,x) arcsecond locations of these extra galaxies, so that when they are included in the model
the centre of these extra galaxies light profiles are fixed to these values (or their priors are initialized
surrounding these centres).

This tutorial closely mirrors tutorial 7, `light_centre`, where the main purpose of this script is to mark the
centres of every object we'll model as an extra galaxy. A GUI is also available to do this.

__Masking Extra Galaxies__

The example `mask_extra_galaxies.py` masks the regions of an image where extra galaxies are present. This mask is used
to remove their signal from the data and increase their noise to make them not impact the fit. This means their
luminous emission does not need to be included in the model, reducing the number of free parameters and speeding up the
analysis. It is still a choice whether their mass is included in the model.

Which approach you use to account for the emission of extra galaxies, modeling or masking, depends on how significant
the blending of their emission with the main galaxy is and how much it impacts the model-fit.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_centres.ipynb` shows how to use a Graphical User Interface (GUI) to mark
the extra galaxy centres in this way.

The script `features/extra_galaxies/modeling` shows how to use extra galaxies in a model-fit, including loading the
extra galaxy centres created by this script.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.

__Contents__

**Masking Extra Galaxies:** Overview of the alternative masking approach for extra galaxy emission.
**Links / Resources:** Links to GUI tools and modeling scripts for extra galaxies.
**Output:** Saving extra galaxy centres as a PNG and JSON file.
"""

# from autoconf import setup_notebook; setup_notebook()

