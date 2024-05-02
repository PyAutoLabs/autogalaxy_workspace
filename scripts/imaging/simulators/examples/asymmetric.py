"""
Simulator: Light MGE
====================

This script simulates `Imaging` of a galaxy using light profiles where:

 - The galaxy's light is a superposition of 14 `Gaussian` profiles.

The galaxy's light is derived from a Multi-Gaussian Expansion (MGE) fit to a massive elliptical galaxy.

The simulated galaxy has irregular and asymmetric features in the galaxy, including a twist in the isophotes of its
emission.

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
dataset_name = "asymmetric"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = ag.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.05,
    over_sampling=ag.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
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

__Basis of Gaussians__

We use a basis of 14 elliptical Gaussians, which represent a complex galaxy morphology with irregular and
asymmetric features such as an isophotal twist which symmetric profiles like a Sersic cannot capture.

The parameters of these Gaussians are loaded from the file `galaxy_mge.json` and their parameters were inferred by
fitting Hubble Space Telescope imaging of a real galaxy with many Gaussian profiles.
"""
# galaxy = ag.Galaxy.from_json(file_path=path.join(dataset_path, "galaxy_mge.json"))

centre_y_list = [
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
]

centre_x_list = [
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
]

ell_comps_0_list = [
    0.05843285,
    0.0,
    0.05368621,
    0.05090395,
    0.0,
    0.25367341,
    0.01677313,
    0.03626733,
    0.15887384,
    0.02790297,
    0.12368768,
    0.38624915,
    -0.10490247,
    0.0385585,
]

ell_comps_1_list = [
    0.05932136,
    0.0,
    0.04267542,
    -0.06920487,
    -0.0,
    -0.15141799,
    0.01464508,
    0.03084128,
    -0.17983965,
    0.02215257,
    -0.16271084,
    -0.15945967,
    -0.3969543,
    -0.03808391,
]

intensity_list = [
    0.52107394,
    4.2933716,
    2.40608609,
    4.98902608,
    2.72773562,
    1.10429021,
    1.08190372,
    0.30007753,
    0.6462658,
    0.15766566,
    0.24687923,
    0.04815128,
    0.02559108,
    0.06763223,
]

sigma_list = [
    0.01607907,
    0.04039063,
    0.06734373,
    0.08471335,
    0.16048498,
    0.13531624,
    0.25649938,
    0.46096968,
    0.34492195,
    0.92418119,
    0.71803244,
    1.23547346,
    1.2574071,
    2.69979461,
]

gaussian_dict = {}

for gaussian_index in range(len(centre_x_list)):
    gaussian = ag.lp.Gaussian(
        centre=(centre_y_list[gaussian_index], centre_x_list[gaussian_index]),
        ell_comps=(
            ell_comps_0_list[gaussian_index],
            ell_comps_1_list[gaussian_index],
        ),
        intensity=intensity_list[gaussian_index],
        sigma=sigma_list[gaussian_index],
    )

    gaussian_dict[f"gaussian_{gaussian_index}"] = gaussian

galaxy = ag.Galaxy(redshift=0.5, **gaussian_dict)

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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/asymmetric`.
"""
