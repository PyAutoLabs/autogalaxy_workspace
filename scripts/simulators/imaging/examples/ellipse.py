"""
Simulator: Ellipse
==================

Ellipse fitting is a common technique in astronomy to determine the shape of a galaxy, which fits a super
position of multiple elliptical isophotes to an image of a galaxy. This allows the ellipticity and position angle of
a galaxy to be determined.

This script simulates a very elliptical galaxy with a `Sersic` light profile, which is used to illustrate ellipse fitting
in the `autogalaxy_workspace/*/notebooks/imaging/modeling/fearure/ellipse_fitting.py` example.

This script simulates `Imaging` of a galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.

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
dataset_name = "ellipse"

dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a `Grid2D` with the adaptive over sampling scheme.
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.

We use a PSF with a sigma value of 0.05, which is smaller than other examples and chosen to ensure the PSF 
does not impact the results of the ellipse fitting example, as ellipse fitting does not account for the PSF convolution.
"""
psf = ag.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])
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

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=path.join(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/simple__sersic`.
"""
