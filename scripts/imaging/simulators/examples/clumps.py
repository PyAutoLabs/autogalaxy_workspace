"""
Simulator: Clumps
=================

Certain galaxies have small galaxies nearby their emission, which may overlap the galaxy emission.

We may therefore wish to include these additional galaxies in the model, as light profiles which fit and subtract the 
emission of these nearby galaxies.
 
This uses the **PyAutoGalaxy** clump API, which is illustrated in 
the  script `autogalaxy_workspace/*/imaging/modeling/features/clumps.py`.

This script simulates an imaging dataset which includes line-of-sight galaxies / clumps near the main galaxy.
This is used to illustrate the clump API in the script above.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - There are two clump objects whose light obscures that of the main galaxy.

__Other Scripts__

This dataset is used in the following scripts:

 `autogalaxy_workspace/*/imaging/data_preparation/examples/optional/scaled_dataset.ipynb`

To illustrate how to subtract and remove the light of clump objects in real strong lensing data, so that it does
not impact the model.

 `autogalaxy_workspace/*/imaging/data_preparation/examples/optional/clump_centres.ipynb`

To illustrate how mark clump centres on a dataset so they can be used in the model.

 `autogalaxy_workspace/*/imaging/modeling/features/clumps.ipynb`

To illustrate how compose and fit a model which includes the clumps as light and mass profiles.

__Advanced__

This is an advanced simulator script, meaning that detailed explanations of certain code are omitted. Refer to
simulators not in the `advanced` folder for more detailed comments.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
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

The path where the dataset will be output.
"""
dataset_type = "imaging"
dataset_name = "clumps"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using the `Grid2DIterate` object, which is a grid of (y,x) coordinates that is iteratively
where the sub-size of the grid is increased until the input fractional accuracy of 99.99% is met.
"""
grid = ag.Grid2DIterate.uniform(
    shape_native=(100, 100),
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
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.

The galaxy includes two clump objects, which must be modeled / masked / have their noise-map increased in 
preprocessing to ensure they do not impact the fit.
"""
clump_0_centre = (1.0, 3.5)

clump_0 = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=clump_0_centre, intensity=0.8, effective_radius=0.5
    ),
    mass=ag.mp.IsothermalSph(centre=clump_0_centre, einstein_radius=0.1),
)

clump_1_centre = (-2.0, -3.5)

clump_1 = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=clump_1_centre, intensity=0.5, effective_radius=0.8
    ),
    mass=ag.mp.IsothermalSph(centre=clump_1_centre, einstein_radius=0.2),
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy, clump_0, clump_1])
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
Pass the simulator galaxies, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

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

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=grid, mat_plot_2d=mat_plot
)
galaxies_plotter.subplot()

"""
__Plane Output__

Save the `Plane` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=path.join(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/clumps`.
"""
