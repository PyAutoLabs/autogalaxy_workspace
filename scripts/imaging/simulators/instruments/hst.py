"""
Simulator: HST
==============

This script simulates `Imaging` of a galaxy where:

 - The resolution, PSF and S/N are representative of Hubble Space Telescope imaging.

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

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "instruments"
dataset_instrument = "hst"

"""
The path where the dataset will be output, which in this case is:
`/autogalaxy_workspace/dataset/imaging/instruments/hst`
"""
dataset_path = path.join("dataset", "imaging", dataset_type, dataset_instrument)

"""
__Simulate__

For simulating an image of a galaxy, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Plane) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = ag.Grid2DIterate.uniform(
    shape_native=(180, 180), pixel_scales=0.05, fractional_accuracy=0.9999
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = ag.Kernel2D.from_gaussian(
    shape_native=(21, 21), sigma=0.05, pixel_scales=grid.pixel_scales, normalize=True
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = ag.SimulatorImaging(
    exposure_time=2000.0, psf=psf, background_sky_level=1.0, add_poisson_noise=True
)

"""
__Plane__

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=45.0),
        intensity=1.0,
        effective_radius=3.0,
        sersic_index=2.0,
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Lets look at the galaxies image, this is the image we'll be simulating.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
We can now pass this simulator galaxies, which creates the image plotted above and simulates it as an
imaging dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

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
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Hubble Space Telescope Image"),
    output=aplt.Output(path=dataset_path, format="png"),
)

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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/instruments/hst`.
"""
