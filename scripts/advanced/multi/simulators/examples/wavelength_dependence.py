"""
Simulator: Wavelength Dependent
===============================

This script simulates multi-wavelength `Imaging` of a galaxy where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.

Unlike other `multi` simulators, the intensity of the source galaxy is a linear function of wavelength following
a relation `y = mx + c`.

This image is used to demonstrate multi-wavelength fitting where a user specified function (e.g. `y = mx+c`) can be
used to parameterize the wavelength variation, as opposed to simply making every `intensity` a free parameter.

Three images are simulated, corresponding green g band (wavelength=464nm), red r-band (wavelength=658nm) and
infrared I-band (wavelength=806nm) observations.

This is an advanced script and assumes previous knowledge of the core **PyAutoGalaxy** API for simulating images. Thus,
certain parts of code are not documented to ensure the script is concise.
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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band), red (r-band) and infrared (I-band).

The strings are used for naming the datasets on output.
"""
color_list = ["g", "r", "I"]

"""
__Wavelengths__

The intensity of each source galaxy is parameterized as a function of wavelength.

Therefore we define a list of wavelengths of each color above.
"""
wavelength_list = [464, 658, 806]

"""
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "wavelength_dependence"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

The pixel-scale of each color image is different meaning we make a list of grids for the simulation.
"""
pixel_scales_list = [0.08, 0.12, 0.012]

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
Simulate simple Gaussian PSFs for the images in the r and g bands.
"""
sigma_list = [0.1, 0.2, 0.25]

psf_list = [
    ag.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the g and r bands.
"""
background_sky_level_list = [0.1, 0.15, 0.1]

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
__Intensity vs Wavelength__

We will assume that the `intensity` of the galaxy bulge and disk linearly varies as a function of wavelength, and 
therefore compute the `intensity` value for each color image using a linear relation.

The relation below is not realistic and has been chosen to make it straight forward to illustrate this functionality.
"""


def bulge_intensity_from(wavelength):
    m = 1.0 / 100.0
    c = 3

    return m * wavelength + c


def disk_intensity_from(wavelength):
    m = -(1.2 / 100.0)
    c = 10

    return m * wavelength + c


"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) and disk (elliptical exponential) for this simulation.

We will assume that the `intensity` of the bulge and disk varies as a function of wavelength, and therefore
compute the `intensity` value for each color image using a linear relation.

The relation below is not realistic and has been chosen to make it straight forward to illustrate this functionality.
"""
bulge_intensity_list = [
    bulge_intensity_from(wavelength=wavelength) for wavelength in wavelength_list
]

disk_intensity_list = [
    disk_intensity_from(wavelength=wavelength) for wavelength in wavelength_list
]

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
Use these to setup galaxies at each waveband, which will generate each image for the simulated `Imaging` 
dataset.
"""
galaxies_list = [ag.Galaxies(galaxies=[galaxy]) for galaxy in galaxy_list]

"""
Lets look at the images, which are the images we'll be simulating.
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
for color, dataset in zip(color_list, dataset_list):
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.

For a faster run time, the galaxies visualization uses the binned grid instead of the iterative grid.
"""
for color, dataset in zip(color_list, dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

for color, grid, galaxies in zip(color_list, grid_list, galaxies_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
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
        obj=galaxies, file_path=path.join(dataset_path, f"{color}_galaxies.json")
    )
    for color, galaxies in zip(color_list, galaxies_list)
]

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/multi/simple`.
"""
