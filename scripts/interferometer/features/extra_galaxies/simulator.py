"""
Simulator: Extra Galaxies (Interferometer)
===========================================

Certain galaxies have extra galaxies nearby their emission, which may overlap the emission of the galaxy
we are interested in. For interferometer data — unlike strong-lensing imaging — we typically *want* to
detect the extra galaxies and fit their light, because autogalaxy is a morphology-fitting tool for
multi-galaxy fields rather than a gravitational-lensing tool.

This script simulates an interferometer dataset containing a main galaxy with a Sersic bulge + Exponential
disk, plus two extra galaxies with `ExponentialSph` light profiles. The output dataset is consumed by the
companion `modeling.py` in the same folder.

This uses the **PyAutoGalaxy** extra galaxies API for the simulator side; the modeling-side use is
illustrated in `autogalaxy_workspace/*/interferometer/features/extra_galaxies/modeling.py`.

__Model__

This script simulates `Interferometer` data of a galaxy where:

 - The main galaxy's bulge is an `Sersic`.
 - The main galaxy's disk is an `Exponential`.
 - There are two extra galaxies whose `ExponentialSph` light blends in the field of view.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.

__Contents__

- **Dataset Paths:** Output path for the simulated dataset.
- **Grid:** Real-space grid the galaxy images are evaluated on.
- **uv-wavelengths:** Load the uv baselines used to NUFFT the image to the visibility plane.
- **Simulator:** `SimulatorInterferometer` (no PSF; uv-plane noise instead of image-plane Poisson noise).
- **Galaxies:** Main galaxy + 2 extra galaxies (light only — autogalaxy is non-lensing).
- **Output:** Saves visibility .fits + dirty-image .png visualizations.
- **Galaxies json:** Save the `Galaxies` collection for future reference.
- **Extra Galaxies Centres:** Save the (y, x) centres of the two extras to `extra_galaxies_centres.json`
  so the modeling script can fix them.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive
name.
"""
dataset_type = "interferometer"
dataset_name = "extra_galaxies"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a (y,x) grid. Over-sampling is an imaging-only technique and is not used for
interferometer data.

This simulated galaxy has additional galaxies offset from the main galaxy centre of (0.0", 0.0").
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

Setup the main galaxy (Sersic bulge + Exponential disk) for this simulation.
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
__Extra Galaxies__

Includes two extra galaxies, which must be modeled to ensure their light does not confuse the fit of the
main galaxy.

Note that their redshift is the same as the main galaxy, which is not necessarily the case in real
observations.
"""
extra_galaxy_0_centre = (1.0, 3.5)

extra_galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.ExponentialSph(
        centre=extra_galaxy_0_centre, intensity=2.0, effective_radius=0.5
    ),
)

extra_galaxy_1_centre = (-2.0, -3.5)

extra_galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.ExponentialSph(
        centre=extra_galaxy_1_centre, intensity=2.0, effective_radius=0.8
    ),
)

"""
Use these galaxies to generate the image which is simulated as an `Interferometer` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy, extra_galaxy_0, extra_galaxy_1])

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

Save the `Galaxies` collection in the dataset folder as a .json file, ensuring the true light profiles
and galaxies are safely stored and available to check how the dataset was simulated in the future.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
__Extra Galaxies Centres__

Save the (y,x) centres of the two extra galaxies as a `Grid2DIrregular` JSON file. The modeling tutorial
`features/extra_galaxies/modeling.py` loads this file to fix the extra-galaxy light-profile centres.
"""
extra_galaxies_centres = ag.Grid2DIrregular(
    values=[extra_galaxy_0_centre, extra_galaxy_1_centre]
)
ag.output_to_json(
    obj=extra_galaxies_centres,
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/dataset/interferometer/extra_galaxies`.
"""
