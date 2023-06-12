"""
Simulator: Sersic
=================

This script simulates `Imaging` of a complex galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Sersic`.
 - The galaxy has four star forming clumps which are all `Sersic`'s.

 __Advanced__

This is an advanced simulator script, meaning that detailed explanations of certain code are omitted. Refer to
simulators not in the `advanced` folder for more detailed comments.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "complex"

"""
The path where the dataset will be output, which in this case is:
`/autogalaxy_workspace/dataset/imaging/complex`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

For simulating an image of a galaxy, we use the Grid2DIterate object.
"""
grid = ag.Grid2DIterate.uniform(
    shape_native=(80, 80),
    pixel_scales=0.1,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = ag.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Plane__

Setup the galaxy with a bulge, disk and star forming clumps (all elliptical Sersics) for this simulation.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=0.2,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    disk=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.1,
        effective_radius=1.6,
    ),
    clump_0=ag.lp.Sersic(centre=(1.0, 1.0), intensity=0.5, effective_radius=0.2),
    clump_1=ag.lp.Sersic(centre=(0.5, 0.8), intensity=0.5, effective_radius=0.2),
    clump_2=ag.lp.Sersic(centre=(-1.0, -0.7), intensity=0.5, effective_radius=0.2),
    clump_3=ag.lp.Sersic(centre=(-1.0, 0.4), intensity=0.5, effective_radius=0.2),
)

"""
Use these galaxies to setup a plane, which generates the image for the simulated `Imaging` dataset.
"""
plane = ag.Plane(galaxies=[galaxy])
plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
plane_plotter.figures_2d(image=True)

"""
Pass the simulator a plane, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_plane_from(plane=plane, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the plane's quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid, mat_plot_2d=mat_plot)
plane_plotter.subplot()

"""
__Plane Output__

Save the `Plane` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `Plane.from_json`.
"""
plane.output_to_json(file_path=path.join(dataset_path, "plane.json"))

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/simple__sersic`.
"""
