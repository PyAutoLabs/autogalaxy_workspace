"""
Simulator: Manual Signal to Noise Ratio
=======================================

Build controlled imaging simulations to test galaxy morphology and separation of overlapping light components as data quality changes.
The `lp_snr` profiles let you request a signal-to-noise ratio instead of manually tuning each profile's intensity.
This is useful for comparing model recovery across a range of brightnesses at fixed observing conditions.

__What the Requested S/N Means__

The target is the expected signal-to-noise ratio at the brightest pixel of each component, evaluated on
the simulation grid with the supplied PSF. It is not an integrated S/N for the galaxy. Each component is
normalized separately, so overlapping components and the final random noise realization need not reproduce
those targets in the combined image.

The normalization solves S/N = S / sqrt(S + B), where S is the component's expected counts in its brightest
pixel and B is the sky counts there. `exposure_time` still has its usual meaning, and `background_sky_level`
is still a count rate: together they determine the intensity required to reach the requested S/N.
Changing either while keeping the target S/N fixed changes the simulated object's brightness as well.

Choose exposure, sky, PSF and sampling to represent the observations you want to study, then inspect the
resulting image and noise map. An S/N target alone does not specify all of the image's noise properties.

This script simulates `Imaging` of a galaxy where:

 - The first galaxy's bulge is an `Sersic` with a S/N of 20.
 - The second galaxy's bulge is an `Sersic` with a S/N of 10.

__Start Here Notebook__

If any code in this script is unclear, refer to the `imaging/simulator.ipynb` notebook.

__Contents__

- **Dataset Paths:** Defining the output path for the simulated dataset.
- **Grid:** Setting up the 2D grid with adaptive over-sampling for simulation.
- **Galaxies:** Defining two galaxies using signal-to-noise ratio based Sersic light profiles.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Plane Output:** Saving the Galaxies object as a JSON file for future reference.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_label = "misc"
dataset_name = "manual_signal_to_noise_ratio"
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Grid__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
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
"""
psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
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

Define two galaxies, each with a Sersic light profile and its own target peak-pixel S/N.
This is a galaxy-light simulation; no lens mass or external shear is included.
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
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

"""
Lets look at the galaxies image, this is the image we'll be simulating.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
We can now pass this simulator galaxies, which creates the image plotted above and simulates it as an
imaging dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
"""
aplt.subplot_imaging_dataset(
    dataset=dataset, output_path=dataset_path, output_format="png"
)
aplt.plot_array(
    array=dataset.data, title="Data", output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies(
    galaxies=galaxies, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Plane Output__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/dataset/imaging/misc/manual_signal_to_noise_ratio`.
"""
