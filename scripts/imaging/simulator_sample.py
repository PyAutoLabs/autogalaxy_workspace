"""
Simulator: Sample Power-Law
===========================

This script simulates a sample of `Imaging` datasets of galaxies where:

 - The galaxy's bulge is an `Sersic`.

To simulate the sample of galaxies, each galaxy is set up as a `Model` such that its parameters are drawn from
distributions defined via priors.

This script uses the signal-to-noise based light profiles described in the
script `simulators/imaging/misc/manual_signal_to_noise_ratio.ipynb`, to make it straight forward to ensure every galaxy
is visible in each image.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.

__Contents__

- **Dataset Paths:** Defining the output path for the simulated sample.
- **Grid:** Setting up the 2D grid with adaptive over-sampling for simulation.
- **Sample Model Distributions:** Defining prior distributions for the galaxy light profile parameters.
- **Sample Instances:** Generating random galaxy instances and simulating imaging datasets in a loop.
- **Output:** Saving each simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Plane Output:** Saving each Galaxies object as a JSON file.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The path where the dataset sample will be output.
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple__sersic"
dataset_path = Path("dataset", dataset_type, dataset_label, dataset_sample_name)

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

grid = ag.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

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
__Sample Truth Distributions__

To simulate a sample, we draw random instances of galaxies. Each parameter is sampled
directly from a numpy ``Generator`` and used to construct a concrete light-profile
instance — there is no ``af.Model`` involved here because we are generating *truths*
for synthetic data, not fitting a model.

The bulge uses ``ag.lp_snr.Sersic`` so each galaxy hits a target signal-to-noise ratio
in the data — SNR is a property of the data, not a fitted parameter.
"""
rng = np.random.default_rng()


def _clipped_ell_comp() -> float:
    return float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0))


def _random_galaxy() -> ag.Galaxy:
    bulge = ag.lp_snr.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
        effective_radius=float(rng.uniform(1.0, 5.0)),
        sersic_index=float(np.clip(rng.normal(4.0, 1.0), 0.8, 5.0)),
        signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
    )
    return ag.Galaxy(redshift=0.5, bulge=bulge)


"""
__Sample Instances__

Within a for loop, we will now generate instances of each simulated galaxy. This loop will
run for `total_datasets` iterations, which sets the number of galaxies that are simulated.

Each iteration of the for loop creates galaxies to simulate the imaging dataset.
"""
total_datasets = 3

for sample_index in range(total_datasets):
    dataset_sample_path = Path(dataset_path, f"dataset_{sample_index}")

    galaxy = _random_galaxy()

    """
    __Galaxies__

    Use the sample's galaxies to generate the image for the 
    simulated `Imaging` dataset.

    The steps below are expanded on in other `imaging/simulator` scripts, so check them out if anything below is unclear.
    """
    galaxies = ag.Galaxies(galaxies=[galaxy])

    aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

    aplt.subplot_imaging_dataset(dataset=dataset)

    """
    __Output__

    Output the simulated dataset to the dataset path as .fits files.

    This uses the updated `dataset_path_sample` which outputs this sample lens to a unique folder.
    """
    aplt.fits_imaging(
        dataset=dataset,
        data_path=Path(dataset_sample_path, "data.fits"),
        psf_path=Path(dataset_sample_path, "psf.fits"),
        noise_map_path=Path(dataset_sample_path, "noise_map.fits"),
        overwrite=True,
    )

    """
    __Visualize__

    Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
    """
    aplt.subplot_imaging_dataset(
        dataset=dataset, output_path=dataset_sample_path, output_format="png"
    )
    aplt.plot_array(
        array=dataset.data,
        title="Data",
        output_path=dataset_sample_path,
        output_format="png",
    )
    aplt.subplot_galaxies(
        galaxies=galaxies,
        grid=grid,
        output_path=dataset_sample_path,
        output_format="png",
    )

    """
    __Plane Output__

    Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
    are safely stored and available to check how the dataset was simulated in the future. 

    This can be loaded via the method `galaxies = ag.from_json()`.
    """
    ag.output_to_json(
        obj=galaxies,
        file_path=Path(dataset_sample_path, "galaxies.json"),
    )

    """
    The dataset can be viewed in the 
    folder `autogalaxy_workspace/imaging/sample/light_sersic_{sample_index]`.
    """
