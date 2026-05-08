"""
Simulator: Mutli Interferometer
===============================

This script simulates `Interferometer` data of a galaxy where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.

This dataset is paired with the script `multi/simulators/simple.py` and therefore
provides interferometer observations of the same galaxy.

It is used to demonstrate the combination of imaging and interferometer datasets.

__Contents__

- **Simulate:** Creating the grid and uv-wavelengths for interferometer simulation.
- **Galaxies:** Setting up the galaxy with bulge and disk light profiles.
- **Output:** Saving the simulated interferometer dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Galaxies json:** Saving the galaxy model as a JSON file for future reference.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

""" 
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. 

 - The image will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "simple"

"""
The path where the dataset will be output.
"""
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Simulate__

For simulating interferometer data of a galaxy, we recommend using a Grid2D object with a `sub_size` of 1. This
simplifies the generation of the galaxy image in real space before it is transformed to Fourier space.
"""
grid = ag.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.05)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.

By default we use baselines from the Square Mile Array (SMA), which produces low resolution interferometer data that
can be fitted extremely efficiently. The `autogalaxy_workspace` includes ALMA uv_wavelengths files for simulating
much high resolution datasets (which can be performed by replacing "sma.fits" below with "alma.fits").
"""
uv_wavelengths_path = Path("dataset", "interferometer", "uv_wavelengths")
uv_wavelengths = ag.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
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
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) and disk (elliptical exponential) for this simulation.
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
Use these galaxies to generate the image for the simulated interferometer dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Lets look at the galaxies images, which are the images we'll be simulating.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
We can now pass this simulator galaxies, which creates the image plotted above and simulates it as an
interferometer dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Lets plot the simulated interferometer dataset before we output it to fits.
"""
aplt.plot_array(array=dataset.dirty_image, title="Dirty Image")
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies's quantities to the dataset path as .png files.
"""
aplt.subplot_interferometer_dirty_images(
    dataset=dataset, output_path=dataset_path, output_format="png"
)
aplt.plot_array(
    array=dataset.dirty_image,
    title="Data",
    output_path=dataset_path,
    output_format="png",
)
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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/no_galaxy_light/simple`.
"""
