"""
Simulator: Same Wavelength
==========================

This script simulates multiple `Imaging` datasets of a galaxy where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.

Unlike other `multi` simulators, all datasets are at the same wavelength and therefore the source does not change
its appearance in each dataset.

This dataset demonstrates how PyAutoGalaxy's multi-dataset modeling tools can also simultaneously analyse datasets
observed at the same wavelength.

An example use case might be analysing undithered HST images before they are combined via the multidrizzing process,
to remove correlated noise in the data.

TODO: NEED TO INCLUDE DIFFERENT POINTING / CENTERINGS.
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
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "same_wavelength"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

If observed at the same wavelength, it is likely the datasets have the same pixel-scale.

Nevertheless, we specify this as a list as there could be an exception.
"""
pixel_scales_list = [0.1, 0.1]

grid_list = [
    ag.Grid2D.uniform(
        shape_native=(150, 150),
        pixel_scales=pixel_scales,
        over_sampling=ag.OverSamplingIterate(
            fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
        ),
    )
    for pixel_scales in pixel_scales_list
]

"""
Simulate simple Gaussian PSFs for the images, which we assume slightly vary (e.g. due to different observing conditions
for each image)
"""
sigma_list = [0.09, 0.11]

psf_list = [
    ag.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the images, which we will assume have slightly different exposure times and background
sky levels.
"""
exposure_time_list = [300.0, 350.0]
background_sky_level_list = [0.1, 0.12]

simulator_list = [
    ag.SimulatorImaging(
        exposure_time=exposure_time,
        psf=psf,
        background_sky_level=background_sky_level,
        add_poisson_noise=True,
    )
    for psf, exposure_time, background_sky_level in zip(
        psf_list, exposure_time_list, background_sky_level_list
    )
]


"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) and disk (elliptical exponential) for this simulation.

The galaxy is observed at the same wavelength in each image thus its intensity does not vary across the datasets.
"""
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
Use these to setup galaxies at each waveband, which will generate each image for the simulated `Imaging` 
dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Lets look at the galaxies images, which are the images we'll be simulating.
"""
for grid in grid_list:
    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

"""
We can now pass this simulator galaxies, which creates the image plotted above and simulates it as an
imaging dataset.
"""
dataset_list = [
    simulator.via_galaxies_from(galaxies=galaxies, grid=grid)
    for grid, simulator in zip(grid_list, simulator_list)
]

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Output__

Output each simulated dataset to the dataset path as .fits files, with a tag describing its color.
"""
for i, dataset in enumerate(dataset_list):
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"image_{i}.fits"),
        psf_path=path.join(dataset_path, f"psf_{i}.fits"),
        noise_map_path=path.join(dataset_path, f"noise_map_{i}.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.

For a faster run time, the galaxies visualization uses the binned grid instead of the iterative grid.
"""
for i, dataset in enumerate(dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, suffix=f"_{i}", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

for i, grid in enumerate(grid_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, suffix=f"_{i}", format="png")
    )

    galaxies_plotter = aplt.GalaxiesPlotter(
        galaxies=galaxies, grid=grid, mat_plot_2d=mat_plot
    )
    galaxies_plotter.subplot_galaxies()
    galaxies_plotter.subplot_galaxy_images()

"""
__Galaxies json__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=path.join(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/multi/same_wavelength/simple`.
"""
