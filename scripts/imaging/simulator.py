"""
Simulator: Sersic + Exp
=======================

This script simulates `Imaging` of a galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - A faint extra galaxy is included offset from the main galaxy, whose emission must be removed via noise scaling
   (a `mask_extra_galaxies.fits` covering it is written below).

__Contents__

- **Dataset Paths:** Defining the output path for the simulated dataset.
- **Grid:** Setting up the 2D grid of coordinates for image evaluation.
- **Over Sampling:** Applying adaptive over-sampling for accurate image simulation.
- **PSF Convolution:** Define the Point Spread Function (PSF) that blurs the simulated image.
- **Galaxies:** Defining the galaxy with Sersic bulge and Exponential disk light profiles.
- **Output:** Saving the simulated dataset to FITS files.
- **Visualize:** Outputting subplot and image visualizations as PNG files.
- **Plane Output:** Saving the Galaxies object as a JSON file for future reference.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autogalaxy_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "imaging"
dataset_name = "simple"

"""
The path where the dataset will be output.

In this example, this is: `/autogalaxy_workspace/dataset/imaging/simple`
"""
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the galaxy images are evaluated and therefore simulated on, via
the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
__Extra Galaxy Centre__

This `simple` dataset deliberately includes a faint extra galaxy offset from the main galaxy, so that the modeling
examples can demonstrate the `__Extra Galaxies Noise Scaling__` step end-to-end. Its centre is defined here so it
can be reused for over-sampling, the galaxy itself and the `mask_extra_galaxies.fits` written further down.

It is placed inside the 3.0" modeling mask, towards the edge where the main galaxy's emission is faint.
"""
extra_galaxy_centre = (2.2, 1.6)

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

An adaptive oversampling scheme is implemented, evaluating the central regions at (0.0", 0.0") of the light profile at a 
resolution of 32x32, transitioning to 8x8 in intermediate areas, and 2x2 in the outskirts. This ensures precise and 
accurate image simulation while focusing computational resources on the bright regions that demand higher oversampling.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), extra_galaxy_centre],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF Convolution__

All CCD imaging data (e.g. Hubble Space Telescope, Euclid) are blurred by the telescope optics when they are imaged.

The Point Spread Function (PSF) describes the blurring of the image by the telescope optics, in the form of a
two dimensional convolution kernel. The modeling scripts use this PSF when fitting the data, to account for
this blurring of the image.

In this example, use a simple 2D Gaussian PSF, which is convolved with the image of the galaxies when simulating
the dataset.

PSF convolution runs at the image resolution (sub size 1), which is the fastest option and accurate for well-sampled
PSFs. Supplying a PSF at a multiple of the image resolution and raising this value improves blurring fidelity for
undersampled PSFs (e.g. HST / Euclid VIS) at extra compute cost — see `guides/advanced/over_sampling.py`.
"""
psf_convolve_over_sample_size = 1

psf = ag.Convolver.from_gaussian(
    convolve_over_sample_size=psf_convolve_over_sample_size,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) and disk (elliptical exponential) for this simulation.

For modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a galaxy you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
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
A single faint extra galaxy offset from the main galaxy, representing a nearby contaminating object whose emission
is not associated with the target. Its emission is removed in the modeling examples via the
`__Extra Galaxies Noise Scaling__` step using the `mask_extra_galaxies.fits` written below.
"""
extra_galaxy = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=extra_galaxy_centre, intensity=1.0, effective_radius=0.3
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy, extra_galaxy])
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
__Mask Extra Galaxies__

Build and output a `mask_extra_galaxies.fits` covering the extra galaxy, so the modeling examples
(`imaging/modeling.py`, `imaging/fit.py`, `imaging/likelihood_function.py`) can load it directly and apply
noise scaling without a separate data-preparation step.

The circle is sized to ~3x the galaxy's `effective_radius`, derived from the same `extra_galaxy_centre` defined
above so it stays in sync with any future tweak.
"""
mask_extra_galaxies = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    centre=extra_galaxy_centre,
    radius=3.0 * 0.3,
    invert=True,  # `True` inside the circle, i.e. the region whose noise is scaled.
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
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
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/simple`.

__JAX Variant__

For an order-of-magnitude speedup on large or repeated simulations
(parameter sweeps, mock-data studies, batch figure generation), construct
the simulator with `use_jax=True` and wrap your call in `@jax.jit`. The
simulator handles pytree registration internally.

```python
import jax

simulator_jax = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    use_jax=True,
)

@jax.jit
def simulate(galaxies):
    return simulator_jax.via_galaxies_from(galaxies=galaxies, grid=grid)

dataset_jax = simulate(galaxies)   # Imaging with jax.Array data
```

The `dataset_jax.data.array` is a `jax.Array`; `aplt.fits_imaging` and the
plotters call `numpy.asarray()` internally, so saving / plotting works
without manual conversion.

Note: eager `simulator_jax.via_galaxies_from(galaxies, grid)` (no `@jax.jit`)
already runs on JAX and is sufficient for one-off simulations. The
`@jax.jit` wrap is only beneficial when you call the function many times.

See `scripts/guides/api/data_structures.py` for the broader "JIT-it-
yourself" pattern.
"""
