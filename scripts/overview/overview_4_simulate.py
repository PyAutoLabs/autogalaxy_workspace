"""
Overview: Simulate
------------------

**PyAutoGalaxy** provides tool for simulating galaxy data-sets, which can be used to test modeling pipelines
and train neural networks to recognise and analyse images of galaxies.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Grid + Lens__

In this overview we used galaxies and grid to create an image of a galaxy.
"""
grid = ag.Grid2D.uniform(
    shape_native=(80, 80),
    pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

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


galaxies = ag.Galaxies(galaxies=[galaxy])

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Simulator__

Simulating galaxy images uses a `SimulatorImaging` object, which models the process that an instrument like the
Hubble Space Telescope goes through to observe a galaxy. This includes accounting for the exposure time to 
determine the signal-to-noise of the data, blurring the observed light of the galaxy with the telescope optics 
and accounting for the background sky in the exposure which adds Poisson noise.
"""
psf = ag.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=0.05)

simulator = ag.SimulatorImaging(
    exposure_time=300.0, background_sky_level=1.0, psf=psf, add_poisson_noise=True
)

"""
Once we have a simulator, we can use it to create an imaging dataset which consists of an image, noise-map and 
Point Spread Function (PSF) by passing it a galaxies and grid.

This uses the galaxies above to create the image of the galaxy and then add the effects that occur during data
acquisition.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
By plotting a subplot of the `Imaging` dataset, we can see this object includes the observed image of the galaxy
(which has had noise and other instrumental effects added to it) as well as a noise-map and PSF:
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Wrap Up__

The `autogalaxy_workspace` includes many example simulators for simulating galaxies with a range of different 
physical properties, to make imaging datasets for a variety of telescopes (e.g. Hubble, Euclid) as well as 
interferometer datasets.
"""
