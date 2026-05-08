"""
Simulator: Sersic
=================

This script simulates `Imaging` of a galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.

__Contents__

- **Dataset Paths:** Set the output path for the simulated dataset.
- **Grid:** Create a 2D grid with adaptive over-sampling for simulation.
- **Galaxies:** Define the galaxy Sersic light profile used for simulation.
- **Output:** Save the simulated dataset to FITS files.
- **Visualize:** Output subplot and image PNGs of the simulated dataset.
- **Mask Extra Galaxies:** Save an empty `mask_extra_galaxies.fits` so noise-scaling tutorials can load it.
- **Plane Output:** Save the Galaxies object as a JSON file.

__Start Here__

If any code in this script is unclear, refer to `imaging/simulator.py` which is a more detailed
walk-through of the same simulation steps for a bulge + disk galaxy.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "simple__sersic"

dataset_path = Path("dataset", dataset_type, dataset_name)

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

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator galaxies, which creates the image which is simulated as an imaging dataset.
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
__Mask Extra Galaxies__

This dataset has no extra galaxies, but pixelization tutorials that load it (e.g.
`imaging/features/pixelization/modeling.py`, `imaging/features/pixelization/fit.py`) demonstrate the
noise-scaling API by applying a `mask_extra_galaxies` mask to the dataset. Output an empty (all-False,
no-pixels-masked) mask so those tutorials can call `apply_noise_scaling(mask=...)` without crashing on a
missing FITS file. The mask shape tracks `dataset.shape_native`, so `PYAUTO_SMALL_DATASETS=1` is honoured
automatically.
"""
mask_extra_galaxies = ag.Mask2D.all_false(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/simple__sersic`.
"""
