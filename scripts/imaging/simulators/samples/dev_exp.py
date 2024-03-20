"""
Simulator: Sample Power-Law
===========================

This script simulates a sample of `Imaging` datasets of galaxies where:

 - The galaxy's bulge is an `DevVaucouleurs`.
 - The galaxy's disk is an `Exponential`.

To simulate the sample of galaxies, each galaxy is set up as a `Model` such that its parameters are drawn from
distributions defined via priors.

This script uses the signal-to-noise based light profiles described in the
script `imaging/simulators/misc/manual_signal_to_noise_ratio.ipynb`, to make it straight forward to ensure every galaxy
is visible in each image.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The path where the dataset sample will be output.
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "dev_exp"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

"""
__Simulate__

For simulating an image of a galaxy, we use the Grid2DIterate object.
"""
grid = ag.Grid2DIterate.uniform(
    shape_native=(150, 150),
    pixel_scales=0.1,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)

grid = ag.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

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
__Sample Model Distributions__

To simulate a sample, we draw random instances of galaxies where the parameters of their light profiles are drawn from 
distributions. These distributions are defined via priors -- the same objects that are used 
when defining the priors of each parameter for a non-linear search.

Below, we define the distributions the galaxy's bulge light is drawn from.
"""
bulge = af.Model(ag.lp_snr.DevVaucouleurs)

bulge.centre = (0.0, 0.0)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.signal_to_noise_ratio = af.UniformPrior(lower_limit=20.0, upper_limit=60.0)
bulge.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)

disk = af.Model(ag.lp_snr.Exponential)

disk.centre = (0.0, 0.0)
disk.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)
disk.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)
disk.signal_to_noise_ratio = af.UniformPrior(lower_limit=20.0, upper_limit=60.0)
disk.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)

galaxy_model = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

"""
__Sample Instances__

Within a for loop, we will now generate instances of each simulated galaxy using the `Model`'s defined above.
This loop will run for `total_datasets` iterations, which sets the number of galaxies that are simulated.

Each iteration of the for loop creates galaxies to simulate the imaging dataset.
"""
total_datasets = 3

for sample_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{sample_index}")

    galaxy = galaxy_model.random_instance()

    """
    __Galaxies__
    
    Use the sample's galaxies to generate the image for the 
    simulated `Imaging` dataset.
    
    The steps below are expanded on in other `imaging/simulator` scripts, so check them out if anything below is unclear.
    """
    galaxies = ag.Galaxies(galaxies=[galaxy])

    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Output__
    
    Output the simulated dataset to the dataset path as .fits files.
    
    This uses the updated `dataset_path_sample` which outputs this sample lens to a unique folder.
    """
    dataset.output_to_fits(
        data_path=path.join(dataset_sample_path, "data.fits"),
        psf_path=path.join(dataset_sample_path, "psf.fits"),
        noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
        overwrite=True,
    )

    """
    __Visualize__
    
    Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
    """
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_sample_path, format="png")
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
        file_path=path.join(dataset_sample_path, "galaxies.json"),
    )

    """
    The dataset can be viewed in the 
    folder `autogalaxy_workspace/imaging/sample/light_sersic_{sample_index]`.
    """
