"""
Simulator: Clumpy Galaxy (Interferometer)
=========================================

This script simulates `Interferometer` data of a galaxy whose real-space light has two distinct components:

 - A smooth central **bulge**, represented by an elliptical `Sersic` profile.
 - Several off-centre **asymmetric star-forming clumps**, represented by `Gaussian` light profiles with varied
   centres, sizes and elongations.

The dataset is the canonical example used by the `interferometer/features/pixelization/` package. The bulge is a
textbook target for a parametric Sersic fit, whereas the clumpy star formation is exactly the kind of irregular
structure that a pixelization is designed to reconstruct.

The galaxy is identical to the one used by the imaging clumpy simulator, so reconstructions in the two pipelines
are directly comparable.

__Contents__

- **Dataset Paths:** Output path for the simulated dataset.
- **Grid:** Real-space grid on which the galaxy image is evaluated before Fourier transform.
- **UV Wavelengths:** Baseline UV wavelengths used to define the visibility coverage.
- **Simulator:** Creating the `SimulatorInterferometer` object.
- **Galaxies:** Defining the galaxy with Sersic bulge and Gaussian clumps.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Plane Output:** Saving the Galaxies object as a JSON file.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "interferometer"
dataset_name = "clumpy"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2D grid of (y,x) coordinates that the galaxy images are evaluated on. For interferometer data, this
image is evaluated in real-space and then transformed to Fourier space — over-sampling is therefore not used.
"""
grid = ag.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.05)

"""
__UV Wavelengths__

To perform the Fourier transform we need the wavelengths of the baselines, loaded here from the workspace's
default SMA `uv_wavelengths` file.
"""
uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = ag.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
__Simulator__

Create the interferometer simulator, which defines the exposure time, noise level and Fourier transform method
used in the simulation.
"""
simulator = ag.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=ag.TransformerNUFFT,
)

"""
__Clump Centres__

The off-centre clumps are placed at the following (y, x) coordinates. They match the imaging clumpy simulator
exactly so the two datasets correspond to the same true galaxy.
"""
clump_centres = [
    (0.4, 0.3),
    (-0.5, 0.6),
    (0.5, -0.4),
    (-0.3, -0.5),
    (0.2, 0.8),
]

"""
__Galaxies__

Set up a single galaxy that consists of a central elliptical `Sersic` **bulge** and five elliptical `Gaussian`
**clumps** placed at the centres defined above.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.5,
    effective_radius=0.6,
    sersic_index=2.5,
)

clump_0 = ag.lp.Gaussian(
    centre=clump_centres[0],
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=30.0),
    intensity=2.0,
    sigma=0.15,
)
clump_1 = ag.lp.Gaussian(
    centre=clump_centres[1],
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=120.0),
    intensity=1.6,
    sigma=0.12,
)
clump_2 = ag.lp.Gaussian(
    centre=clump_centres[2],
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.4, angle=75.0),
    intensity=1.3,
    sigma=0.18,
)
clump_3 = ag.lp.Gaussian(
    centre=clump_centres[3],
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=150.0),
    intensity=1.8,
    sigma=0.10,
)
clump_4 = ag.lp.Gaussian(
    centre=clump_centres[4],
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=10.0),
    intensity=1.2,
    sigma=0.13,
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=bulge,
    clump_0=clump_0,
    clump_1=clump_1,
    clump_2=clump_2,
    clump_3=clump_3,
    clump_4=clump_4,
)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Lets look at the real-space galaxy image, which is what gets Fourier transformed to produce the visibilities.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the galaxies to the simulator, which Fourier transforms the real-space image to produce a simulated
interferometer dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Plot the simulated interferometer dataset before outputting it to fits.
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

Output subplots and PNG images of the simulated dataset.
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
__Plane Output__

Save the `Galaxies` in the dataset folder as a .json file so the true bulge + clumps galaxy is available for later
inspection.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/dataset/interferometer/clumpy`.
"""
