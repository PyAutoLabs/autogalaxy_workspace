"""
Simulator: Point Source
=======================

This script simulates an `Imaging` dataset containing a galaxy with:

 - An extended `Sersic` host galaxy.
 - An unresolved point source represented by `ag.lp.PointSource`.

A point source is a delta function in the image plane. Its `intensity` is the
total flux of the source, not a surface-brightness value evaluated independently
at every grid coordinate.

Accurate point-source imaging requires an over-sampled PSF. The source is placed
on the nearest sub-pixel, convolved with the PSF at that finer resolution and
then binned back to the detector pixels while conserving total flux. Setting
`convolve_over_sample_size=1` remains a valid pixel-centred approximation, but
does not retain the source's sub-pixel position.

For model fitting, `ag.lp_linear.PointSource` provides the same spatial model
with its total flux solved by the linear inversion. The source centre remains a
non-linear parameter.

__Contents__

- **Dataset Paths:** Define where the simulated dataset is written.
- **Grid:** Create a detector grid with sampling compatible with the PSF.
- **PSF:** Define a Gaussian PSF sampled at twice the detector resolution.
- **Galaxies:** Combine an extended host and unresolved point source.
- **Simulation:** Simulate noisy imaging using fine-grid PSF convolution.
- **Output:** Write the dataset, plots and input galaxies to disk.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The simulated data are written to `dataset/imaging/point_source`.
"""
dataset_path = Path("dataset", "imaging", "point_source")

"""
__Grid__

The detector pixels are 0.1 arcseconds across. A uniform over-sample size of 2
gives four sub-pixels per detector pixel and is compatible with the PSF
convolution factor defined below.
"""
over_sample_size = 2

grid = ag.Grid2D.uniform(
    shape_native=(101, 101),
    pixel_scales=0.1,
    over_sample_size=over_sample_size,
)

"""
__PSF__

The PSF kernel is sampled on pixels that are twice as fine as the detector
pixels. Therefore its `pixel_scales` are `0.1 / 2 = 0.05` arcseconds and its
`convolve_over_sample_size` is 2.

During simulation PyAutoGalaxy evaluates the point source on this fine grid,
convolves it with the fine PSF and bins the result back to the 0.1 arcsecond
detector grid.
"""
psf = ag.Convolver.from_gaussian(
    shape_native=(21, 21),
    sigma=0.12,
    pixel_scales=grid.pixel_scales[0] / over_sample_size,
    normalize=True,
    convolve_over_sample_size=over_sample_size,
)

simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

"""
__Galaxies__

The point source is offset by a quarter of a detector pixel in each direction.
At over-sample size 2 this selects one of the central pixel's four sub-pixels,
so the blurred image retains the offset instead of forcing the source onto the
detector-pixel centre.

The point source `intensity=25.0` is its total flux before noise and background
sky are added. It is conserved when the fine-grid image is binned back to the
detector resolution.
"""
point_source_centre = (0.025, -0.025)

galaxy = ag.Galaxy(
    redshift=0.5,
    host=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=2.0,
    ),
    point_source=ag.lp.PointSource(
        centre=point_source_centre,
        intensity=25.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
__Simulation__

Simulate the dataset. The simulator automatically uses the over-sampled PSF
path because `convolve_over_sample_size=2`.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Write the image, PSF and noise map as FITS files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
Write PNG visualizations of the simulated dataset and its input galaxy.
"""
aplt.subplot_imaging_dataset(
    dataset=dataset,
    output_path=dataset_path,
    output_format="png",
)
aplt.subplot_galaxies(
    galaxies=galaxies,
    grid=grid,
    output_path=dataset_path,
    output_format="png",
)

"""
Save the exact input galaxy as JSON for reproducibility.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=dataset_path / "galaxies.json",
)

print(f"Point-source dataset written to {dataset_path}")
