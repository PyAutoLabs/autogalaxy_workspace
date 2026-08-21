"""
Simulator: Operated Light Profiles (Interferometer)
===================================================

This script simulates `Interferometer` data of a galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy has a compact component of point-source emission at its centre which is modeled as an operated
   `Gaussian`.

For interferometer data there is no Point Spread Function: the visibilities are the Fourier transform of the
galaxy's sky emission, and the synthesized beam only enters when a dirty image is formed. The operated
`Gaussian` therefore represents compact nuclear emission (e.g. an AGN or a compact knot of star formation)
whose image-plane shape is specified directly, and it is Fourier transformed to the visibility plane like
every other light profile.

The output dataset is consumed by the companion `modeling.py` in the same folder.

__Advanced__

This is an advanced simulator script, meaning that detailed explanations of certain code are omitted. Refer to
simulators not in the `advanced` folder for more detailed comments.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/simulator.ipynb` notebook.

__Contents__

- **Dataset Paths:** Defining the output path for the simulated dataset.
- **Grid:** Real-space grid the galaxy images are evaluated on.
- **uv-wavelengths:** Load the uv baselines used to NUFFT the image to the visibility plane.
- **Simulator:** `SimulatorInterferometer` (no PSF; uv-plane noise instead of image-plane Poisson noise).
- **Galaxies:** Defining the galaxy with a Sersic bulge and an operated Gaussian point source.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and dirty-image visualizations as PNG files.
- **Galaxies json:** Saving the Galaxies object as a JSON file for future reference.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "interferometer"
dataset_name = "operated"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a (y,x) grid. Over-sampling is an imaging-only technique and is not used for
interferometer data.
"""
grid = ag.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
__uv-wavelengths__

To perform the Fourier transform we need the wavelengths of the baselines.
"""
uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = ag.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
__Simulator__

Create the simulator for the interferometer data, which defines the exposure time, visibility-plane
noise sigma, and transformer.
"""
simulator = ag.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=ag.TransformerDFT,
)

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.

This includes an operated `Gaussian` component which represents the compact point-source emission at the
galaxy's centre. Its image-plane shape is specified directly by the profile and is Fourier transformed to
the visibility plane like every other light profile (there is no PSF convolution for interferometer data).
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
    psf=ag.lp_operated.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=100.0, sigma=0.1
    ),
)

"""
Use these galaxies to generate the image which is simulated as an `Interferometer` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator galaxies, which creates the real-space image and NUFFTs it to visibilities.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Plot the simulated `Interferometer` dataset before outputting it to fits.
"""
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

Output a subplot of the simulated dataset and the galaxies' images to the dataset path as .png files.
"""
aplt.subplot_interferometer_dirty_images(
    dataset=dataset, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies(
    galaxies=galaxies, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Galaxies json__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/dataset/interferometer/operated`.
"""
