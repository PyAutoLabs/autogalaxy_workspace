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

__Contents__

- **Dataset Paths:** Setting the output path for simulated data.
- **Simulate:** Creating grids with matching pixel scales for same-wavelength observations.
- **Galaxies:** Setting up the galaxy with bulge and disk light profiles at one wavelength.
- **Output:** Saving the simulated datasets to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Galaxies json:** Saving the galaxy model as a JSON file for future reference.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt


"""
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "same_wavelength"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Simulate__

If observed at the same wavelength, it is likely the datasets have the same pixel-scale.

Nevertheless, we specify this as a list as there could be an exception.
"""
pixel_scales_list = [0.1, 0.1]

grid_list = []

for pixel_scales in pixel_scales_list:
    grid = ag.Grid2D.uniform(
        shape_native=(150, 150),
        pixel_scales=pixel_scales,
    )

    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    grid_list.append(grid)

"""
Simulate simple Gaussian PSFs for the images, which we assume slightly vary (e.g. due to different observing conditions
for each image)
"""
sigma_list = [0.09, 0.11]

psf_list = [
    ag.Convolver.from_gaussian(
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
        add_poisson_noise_to_data=True,
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
    aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

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
    aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output each simulated dataset to the dataset path as .fits files, with a tag describing its color.
"""
for i, dataset in enumerate(dataset_list):
    aplt.fits_imaging(
        dataset=dataset,
        data_path=Path(dataset_path, f"image_{i}.fits"),
        psf_path=Path(dataset_path, f"psf_{i}.fits"),
        noise_map_path=Path(dataset_path, f"noise_map_{i}.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.

For a faster run time, the galaxies visualization uses the binned grid instead of the iterative grid.
"""
for i, dataset in enumerate(dataset_list):
    aplt.subplot_imaging_dataset(
        dataset=dataset, output_path=dataset_path, output_format="png"
    )
    aplt.plot_array(
        array=dataset.data, title="Data", output_path=dataset_path, output_format="png"
    )

for i, grid in enumerate(grid_list):
    aplt.subplot_galaxies(
        galaxies=galaxies, grid=grid, output_path=dataset_path, output_format="png"
    )

"""
__Galaxies json__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/multi/same_wavelength/simple`.
"""
