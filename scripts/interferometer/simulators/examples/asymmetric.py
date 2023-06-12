"""
Simulator: Sersic
=================

This script simulates `Interferometer` data of a galaxy where:

 - The galaxy's light is a superposition of 14 `Gaussian` profiles.

The galaxy's light is derived from a Multi-Gaussian Expansion (MGE) fit to a massive elliptical galaxy.

The simulated galaxy has irregular and asymmetric features in the galaxy, including a twist in the isophotes of its
emission.
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
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "interferometer"
dataset_name = "asymmetric"

"""
The path where the dataset will be output, which in this case is
`/autogalaxy_workspace/dataset/interferometer/asymmetric`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

For simulating interferometer data of a galaxy, we recommend using a Grid2D object with a `sub_size` of 1. This
simplifies the generation of the galaxy image in real space before it is transformed to Fourier space.
"""
grid = ag.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.05, sub_size=1)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.

By default we use baselines from the Square Mile Array (SMA), which produces low resolution interferometer data that
can be fitted extremely efficiently. The `autogalaxy_workspace` includes ALMA uv_wavelengths files for simulating
much high resolution datasets (which can be performed by replacing "sma.fits" below with "alma.fits").
"""
uv_wavelengths_path = path.join("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = ag.util.array_1d.numpy_array_1d_via_fits_from(
    file_path=path.join(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
To simulate the interferometer dataset we first create a simulator, which defines the exposure time, noise levels 
and Fourier transform method used in the simulation.
"""
simulator = ag.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=ag.TransformerDFT,
)

"""
__Plane__

Setup the galaxy with 14 elliptical Gaussians, which represent a complex galaxy morphology with irregular and
asymmetric features such as an isophotal twist which symmetric profiles like a Sersic cannot capture.

The parameters of these Gaussians are loaded from the file `galaxy_mge.json` and their parameters were inferred by
fitting Hubble Space Telescope imaging of a real galaxy with many Gaussian profiles.

For modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a galaxy you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoGalaxy** `convert` module to determine the elliptical components from the axis-ratio and angle.
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
Use these galaxies to setup a plane, which will generate the image for the simulated interferometer dataset.
"""
plane = ag.Plane(galaxies=[galaxy])

"""
Lets look at the plane`s image, this is the image we'll be simulating.
"""
plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
plane_plotter.figures_2d(image=True)

"""
We can now pass this simulator a plane, which creates the image plotted above and simulates it as an
interferometer dataset.
"""
dataset = simulator.via_plane_from(plane=plane, grid=grid)

"""
Lets plot the simulated interferometer dataset before we output it to fits.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(dirty_image=True)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the plane's quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()
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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/asymmetric`.
"""
