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

import numpy as np
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
The path where the galaxy light centre is output, which is `dataset/imaging/simple`.
"""
dataset_type = "imaging"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` dataset, so that the galaxy light centres can be plotted over the galaxy image.
"""
data = ag.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

"""
Now, create a lens light centre, which is a `Grid2DIrregular` object of (y,x) values.
"""
light_centre = ag.Grid2DIrregular(values=[(0.0, 0.0)])

"""
Now lets plot the image and lens light centre, so we can check that the centre overlaps the galaxy light.
"""
aplt.plot_array(array=data, title="Data", positions=[np.array(light_centre)])

"""
Now we`re happy with the galaxy light centre(s), lets output them to the dataset folder of the galaxy, so that we can
load them from a .json file in our pipelines!
"""
ag.output_to_json(
    obj=light_centre,
    file_path=Path(dataset_path, "light_centre.json"),
)

"""
The workspace also includes a GUI for drawing lens light centres, which can be found at
`autogalaxy_workspace/*/data_preparation/imaging/gui/light_centres.py`.

This tools allows you `click` on the image where the galaxy light centres are, and it uses the brightest
pixel within a 5x5 box of pixels to select the coordinate.
"""
