"""
Plots: ImagingPlotter
=====================

This example illustrates how to plot an `Imaging` dataset using an `ImagingPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a galaxy as an `Imaging` object.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Figures__

We now pass the imaging to an `ImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    psf=True,
)

"""
__Subplots__

The `ImagingPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""
Finish.
"""
