"""
Simulator: Manual Signal to Noise Ratio
=======================================

When simulating `Imaging` of a galaxy, one is often not concerned with the actual units of the light (e.g.
electrons per second, counts, etc.) but instead simple wants the data to correspond to a certain signal to noise
value.

This can be difficult to achieve when specifying the `intensity` of the input light profiles.

This script illustrates the `lp_snr` light profiles, which when used to simulate a dataset via a plane, set the
signal to noise of each light profile to an input value. This uses the `exposure_time` and `background_sky_level`
of the `SimulatorImaging` object to choose the `intensity` of each light profile such that the input signal to
noise is used.

For normal light profiles, the `intensity` is defined in units of electrons per second, meaning that the
`exposure_time` and `background_sky_level` are used to convert this to counts when adding noise. When the `lp_snr`
profiles are used, the `exposure_time` and `background_sky_level` are instead used to set its S/N, meaning their input
values do not set the S/N.

However, the ratio of `exposure_time` and `background_sky_level` does set how much noise is due to Poisson count
statistics in the CCD imaging detector relative to the background sky. If one doubles the `exposure_time`, the
Poisson count component will contribute more compared to the background sky component. For detailed scientific
analysis, one should therefore make sure their values are chosen to produce images with realistic noise properties.

The use of the `light_snr` profiles changes the meaning of `exposure_time` and `background_sky_level`.

This script simulates `Imaging` of a galaxy where:

 - The first galaxy's bulge is an `Sersic` with a S/N of 20.
 - The second galaxy's bulge is an `Sersic` with a S/N of 10.

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

 - The image will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_type = "imaging"
dataset_label = "misc"
dataset_name = "manual_signal_to_noise_ratio"

"""
The path where the dataset will be output, which in this case is:
`/autogalaxy_workspace/dataset/imaging/misc/manual_signal_to_noise_ratio`
"""
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

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
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Plane__

Setup the galaxy's light (elliptical Sersic + Exponential), mass (SIE+Shear) and galaxy light
(elliptical Sersic) for this simulation.

For modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a galaxy you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoGalaxy** `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp_snr.Sersic(
        signal_to_noise_ratio=20.0,
        centre=(0.0, -1.0),
        ell_comps=(0.25, 0.1),
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp_snr.Sersic(
        signal_to_noise_ratio=10.0,
        centre=(0.0, 1.0),
        ell_comps=(0.0, 0.1),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

"""
Use these galaxies to setup a plane, which generates the image for the simulated `Imaging` dataset.
"""
plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

"""
Lets look at the plane`s image, this is the image we'll be simulating.
"""
plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
plane_plotter.figures_2d(image=True)

"""
We can now pass this simulator a plane, which creates the image plotted above and simulates it as an
imaging dataset.
"""
dataset = simulator.via_plane_from(plane=plane, grid=grid)

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

Output a subplot of the simulated dataset, the image and the plane's quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid, mat_plot_2d=mat_plot)
plane_plotter.subplot()

"""
__Plane Output__

Save the `Plane` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `plane = ag.from_json()`.
"""
ag.output_to_json(
    obj=plane,
    file_path=path.join(dataset_path, "plane.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/misc/manual_signal_to_noise_ratio`.
"""
