import numpy as np

"""
Data Preparation: Lens Light Centre (Optional)
==============================================

This script marks the (y,x) arcsecond locations of the galaxy light centre(s) of the galaxy you are
analysing. These can be used as fixed values for the galaxy light and mass models in a model-fit.

This reduces the number of free parameters fitted for in a model and removes inaccurate solutions where
the galaxy mass model centre is unrealistically far from its true centre.

Advanced `chaining` scripts often use these input centres in the early fits to infer an accurate initial model,
amd then make the centres free parameters in later searches to ensure a general and accurate model is inferred.

If you create a `light_centre` for your dataset, you must also update your modeling script to use them.

If your **PyAutoGalaxy** analysis is struggling to converge to a good model, you should consider using a fixed
lens light and / or mass centre to help the non-linear search find a good model.

Links / Resources:

The script `data_preparation/gui/light_centre.ipynb` shows how to use a Graphical User Interface (GUI) to mask the
galaxy light centres.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

