"""
Simulator: Extra Galaxies
=========================

Certain galaxies have extra galaxies nearby their emission, which may overlap the emission of the galaxy we are
interested in.

We therefore will mask the emission of these extra galaxies or include them in the model as light profiles which
fit and subtract the emission.

This uses the **PyAutoGalaxy** extra galaxies API, which is illustrated in
the  script `autogalaxy_workspace/*/modeling/imaging/features/extra_galaxies.py`.

This script simulates an imaging dataset which includes extra galaxies near the main galaxy.
This is used to illustrate the extra galaxies API in the script above.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - There are two extra galaxies whose light obscures that of the main galaxy.

__Other Scripts__

This dataset is used in the following scripts:

 `autogalaxy_workspace/*/data_preparation/imaging/examples/optional/scaled_dataset.ipynb`

To illustrate how to subtract and remove the light of extra galaxies in real imaging data, so that it does
not impact the model.

 `autogalaxy_workspace/*/data_preparation/imaging/examples/optional/extra_galaxies_centres.ipynb`

To illustrate how mark extra galaxy centres on a dataset so they can be used in the model.

 `autogalaxy_workspace/*/modeling/imaging/features/extra_galaxies.ipynb`

To illustrate how compose and fit a model which includes the extra galaxies as light profiles.

__Advanced__

This is an advanced simulator script, meaning that detailed explanations of certain code are omitted. Refer to
simulators not in the `advanced` folder for more detailed comments.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.

__Contents__

- **Dataset Paths:** Defining the output path for the simulated dataset.
- **Grid:** Setting up the 2D grid with adaptive over-sampling for all galaxy centres.
- **Galaxies:** Defining the main galaxy with Sersic bulge and Exponential disk light profiles.
- **Extra Galaxies:** Defining two extra galaxies with spherical Exponential light profiles.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Mask Extra Galaxies:** Building and saving `mask_extra_galaxies.fits` so consumers can load it directly.
- **Plane Output:** Saving the Galaxies object as a JSON file for future reference.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "extra_galaxies"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.

This simulated galaxy has additional galaxies and light profiles which are offset from the main galaxy centre 
of (0.0", 0.0"). The adaptive over sampling grid has all centres input to account for this.
"""
grid = ag.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.1,
)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), (1.0, 3.5), (-2.0, -3.5)],
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

Includes two extra galaxies, which must be modeled or masked to ensure they do not impact the fit.

Note that their redshift is the same as the main galaxy, which is not necessarily the case in real observations. 

If they are at a different redshift, the tools for masking or modeling extra galaxies are equipped to handle this.
"""
extra_galaxy_0_centre = (1.0, 3.5)

extra_galaxy_0 = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=extra_galaxy_0_centre, intensity=2.0, effective_radius=0.5
    ),
)

extra_galaxy_1_centre = (-2.0, -3.5)

extra_galaxy_1 = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=extra_galaxy_1_centre, intensity=2.0, effective_radius=0.8
    ),
)


"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy, extra_galaxy_0, extra_galaxy_1])
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

Build and output a `mask_extra_galaxies.fits` covering the two extra galaxy regions, so that downstream tutorials
which use this dataset (e.g. `imaging/start_here.py`, `imaging/features/extra_galaxies/modeling.py`,
`imaging/features/pixelization/modeling.py`) can load the mask directly without a separate data-preparation step.

Each circle is sized to ~3x the galaxy's `effective_radius`, which comfortably covers the light extent for the
`ExponentialSph` profiles used above. The geometry is derived from the same centres + radii defined for the
extra galaxies in this script, so it stays in sync with any future tweak to those values.

`Mask2D.circular` honours the `PYAUTO_SMALL_DATASETS=1` env var (caps to 15x15 at 0.6"/px), so the mask
automatically shrinks alongside the small-dataset image and never raises an out-of-bounds error.
"""
extra_galaxies_mask = np.zeros(dataset.shape_native, dtype=bool)

for centre, radius in [
    (extra_galaxy_0_centre, 3.0 * 0.5),
    (extra_galaxy_1_centre, 3.0 * 0.8),
]:
    circle = ag.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        centre=centre,
        radius=radius,
        invert=True,  # True inside the circle (i.e. masked region)
    )
    extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask_extra_galaxies = ag.Mask2D(
    mask=extra_galaxies_mask,
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
__Extra Galaxies Centres__

Save the (y,x) centres of the two extra galaxies as a `Grid2DIrregular` JSON file. The modeling tutorial
`features/extra_galaxies/modeling.py` loads this file to fix or initialise the extra-galaxy light-profile
centres. The data-preparation tutorial `data_preparation/examples/optional/extra_galaxies_centres.py`
demonstrates how a user would mark these centres themselves on real data.
"""
extra_galaxies_centres = ag.Grid2DIrregular(
    values=[extra_galaxy_0_centre, extra_galaxy_1_centre]
)
ag.output_to_json(
    obj=extra_galaxies_centres,
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/extra_galaxies`.
"""
