"""
Simulator: Clumpy Galaxy
========================

This script simulates `Imaging` of a galaxy whose light has two distinct components:

 - A smooth central **bulge**, represented by an elliptical `Sersic` profile.
 - Several off-centre **asymmetric star-forming clumps**, represented by `Gaussian` light profiles with varied centres,
   sizes and elongations. The clumps are placed irregularly across the galaxy and cannot be reproduced by any
   axisymmetric (or even multi-Sersic) decomposition.

The dataset is the canonical example used by the `imaging/features/pixelization/` package. The bulge is a textbook
target for a parametric Sersic fit, whereas the clumpy star formation is exactly the kind of irregular structure
that a pixelization is designed to reconstruct. Together they motivate the standard pixelization use-case:
**parametric profile for the smooth component, pixelization for the irregular component**.

__Contents__

- **Dataset Paths:** Defining the output path for the simulated dataset.
- **Grid:** Setting up the 2D grid of coordinates for image evaluation.
- **Over Sampling:** Applying adaptive over-sampling around the bulge and every clump.
- **Galaxies:** Defining the galaxy with a Sersic bulge and several Gaussian clumps.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Mask Extra Galaxies:** Saving an empty `mask_extra_galaxies.fits` so the noise-scaling section of the
  pixelization tutorials can load it without crashing.
- **Plane Output:** Saving the Galaxies object as a JSON file for future reference.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autogalaxy_workspace/dataset/imaging/clumpy/data.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/imaging/clumpy/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/imaging/clumpy/psf.fits`.
"""
dataset_type = "imaging"
dataset_name = "clumpy"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2D grid of (y,x) coordinates that the galaxy image is evaluated on:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
__Clump Centres__

The off-centre clumps are placed at the following (y, x) coordinates. They are intentionally asymmetric — no two
clumps share a centre, position angle, or elongation, which is what makes the resulting galaxy hard to fit with
parametric light profiles.

The same centres are passed to the adaptive over-sampling helper below so that the bulge and every clump are
evaluated with the same numerical accuracy.
"""
clump_centres = [
    (0.4, 0.3),
    (-0.5, 0.6),
    (0.5, -0.4),
    (-0.3, -0.5),
    (0.2, 0.8),
]

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

An adaptive oversampling scheme is implemented, evaluating each bright region (the bulge centre and every clump
centre) at a resolution of 32x32, transitioning to 8x8 in intermediate areas, and 2x2 in the outskirts.

Once you are more experienced, you should read up on over-sampling in more detail via
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_centres = [(0.0, 0.0)] + clump_centres

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=over_sample_centres,
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

Set up a single galaxy that consists of:

 - A central elliptical `Sersic` **bulge** with a high Sersic index (n=3), placed at the centre of the image. This is
   the smooth component a parametric profile fits well.

 - Five elliptical `Gaussian` **clumps** at the centres defined above, with varied intensities, sizes (sigma) and
   position angles. These represent asymmetric, irregular star-forming regions that no parametric decomposition can
   capture cleanly — a pixelization will be used to reconstruct them in the modeling tutorial.
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

"""
Use this galaxy to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the galaxy to the simulator, which creates the image that is simulated as an imaging dataset.
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

Output a subplot of the simulated dataset, the image and the galaxy quantities to the dataset path as .png files.
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

This dataset has no extra galaxies, but the pixelization tutorials that load it (e.g.
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
The dataset can be viewed in the folder `autogalaxy_workspace/dataset/imaging/clumpy`.
"""
