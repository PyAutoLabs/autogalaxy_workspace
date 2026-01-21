"""
Simulator: Multi Light Sersic
=============================

This script simulates multi-wavelength `Imaging` of a galaxy where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.

Two images are simulated, corresponding to a greener ('g' band) redder image (`r` band).

This is an advanced script and assumes previous knowledge of the core **PyAutoGalaxy** API for simulating images. Thus,
certain parts of code are not documented to ensure the script is concise.
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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for naming the datasets on output.
"""
waveband_list = ["g", "r"]

"""
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Simulate__

The pixel-scale of each color image is different meaning we make a list of grids for the simulation.
"""
pixel_scales_list = [0.08, 0.12]

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
Simulate simple Gaussian PSFs for the images in the r and g bands.
"""
sigma_list = [0.1, 0.2]

psf_list = [
    ag.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the g and r bands.
"""
background_sky_level_list = [0.1, 0.15]

simulator_list = [
    ag.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=background_sky_level,
        add_poisson_noise_to_data=True,
    )
    for psf, background_sky_level in zip(psf_list, background_sky_level_list)
]

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) and disk (elliptical exponential) for this simulation.

The galaxy at each wavelength has a different intensity for its bulge and disk, thus we create two source galaxies 
for each waveband.
"""
bulge_intensity_list = [0.2, 0.4]
disk_intensity_list = [0.2, 0.5]

galaxy_list = [
    ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=bulge_intensity,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
        disk=ag.lp.Exponential(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
            intensity=disk_intensity,
            effective_radius=1.6,
        ),
    )
    for bulge_intensity, disk_intensity in zip(
        bulge_intensity_list, disk_intensity_list
    )
]

"""
Use these galaxies each waveband, which will generate each image for the simulated `Imaging` dataset.
"""
galaxies_list = [ag.Galaxies(galaxies=[galaxy]) for galaxy in galaxy_list]

"""
Lets look at the galaxies`s images, which are the images we'll be simulating.
"""
for galaxies, grid in zip(galaxies_list, grid_list):
    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

"""
We can now pass this simulator galaxies, which creates the image plotted above and simulates it as an
imaging dataset.
"""
dataset_list = [
    simulator.via_galaxies_from(galaxies=galaxies, grid=grid)
    for grid, simulator, galaxies in zip(grid_list, simulator_list, galaxies_list)
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
for waveband, dataset in zip(waveband_list, dataset_list):
    dataset.output_to_fits(
        data_path=Path(dataset_path) / f"{waveband}_data.fits",
        psf_path=Path(dataset_path) / f"{waveband}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{waveband}_noise_map.fits",
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.

For a faster run time, the galaxies visualization uses the binned grid instead of the iterative grid.
"""
for waveband, dataset in zip(waveband_list, dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{waveband}_", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

for waveband, grid, galaxies in zip(waveband_list, grid_list, galaxies_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{waveband}_", format="png")
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
[
    ag.output_to_json(
        obj=galaxies, file_path=Path(dataset_path, f"{waveband}_galaxies.json")
    )
    for waveband, galaxies in zip(waveband_list, galaxies_list)
]

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/multi/simple`.
"""
