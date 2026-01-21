"""
PyAutoGalaxy
============

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

![HST Image](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/hstcombined.png)

**PyAutoGalaxy** has three core aims:

- **Big Data**: Scaling automated Sérsic fitting to extremely large datasets, *accelerated with JAX on GPUs and using
  tools like an SQL database to **build a scalable scientific workflow***.

- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse
  Fitting, Irregular Meshes) that go beyond just simple Sérsic fitting.

- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be
  fitted independently or simultaneously.

This notebook gives an overview of **PyAutoGalaxy**'s API, core features and details of the autogalaxy_workspace.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoGalaxy installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

setup_colab.for_autogalaxy(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
# %matplotlib inline

import autogalaxy as ag
import autogalaxy.plot as aplt

import matplotlib.pyplot as plt
from os import path

"""
Lets illustrate a simple galaxy structure calculations creating an an image of a galaxy using a light profile.

__Grid__

The emission of light from a galaxy is described using the `Grid2D` data structure, which is two-dimensional
Cartesian grids of (y,x) coordinates where the light profile of the galaxy is evaluated on the grid.

We make and plot a uniform Cartesian grid:
"""
grid = ag.Grid2D.uniform(
    shape_native=(150, 150),  # The [pixels x pixels] shape of the grid in 2D.
    pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
__Light Profiles__

Our aim is to create an image of the morphological structures that make up a galaxy.

This uses analytic functions representing a galaxy's light, referred to as `LightProfile` objects. 

The most common light profile in Astronomy is the elliptical Sersic, which we create an instance of below:
"""
sersic_light_profile = ag.lp.Sersic(
    centre=(0.0, 0.0),  # The light profile centre [units of arc-seconds].
    ell_comps=(
        0.2,
        0.1,
    ),  # The light profile elliptical components [can be converted to axis-ratio and position angle].
    intensity=0.005,  # The overall intensity normalisation [units arbitrary and are matched to the data].
    effective_radius=2.0,  # The effective radius containing half the profile's total luminosity [units of arc-seconds].
    sersic_index=4.0,  # Describes the profile's shape [higher value -> more concentrated profile].
)

"""
By passing the light profile the `grid`, we evaluate the light emitted at every (y,x) coordinate and therefore create 
an image of the Sersic light profile.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

plt.imshow(image.native)  # Dont worry about the use of .native for now.

"""
__Plotting__

In-built plotting methods are provided for plotting objects and their properties, like the image of
a light profile we just created.

By using a `LightProfilePlotter` to plot the light profile's image, the figured is improved. 

Its axis units are scaled to arc-seconds, a color-bar is added, its given a descriptive labels, etc.

The plot module is highly customizable and designed to make it straight forward to create clean and informative figures
for fits to large datasets.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.figures_2d(image=True)

"""
__Galaxy__

A `Galaxy` object is a collection of light profiles at a specific redshift.

This object is highly extensible and is what ultimately allows us to fit complex models to galaxy images.

Below, we combine the Sersic light profile above with an Exponential light profile to create a galaxy containing both
a bulge and disk component.
"""
exponential_light_profile = ag.lp.Exponential(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.0), intensity=0.1, effective_radius=0.5
)

galaxy = ag.Galaxy(
    redshift=0.5, bulge=sersic_light_profile, disk=exponential_light_profile
)

"""
The `GalaxyPlotter` object plots the image of the galaxy, which is the sum of its bulge and disk light profiles.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
One example of the plotter's customizability is the ability to plot the individual light profiles of the galaxy
on a subplot.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
__Galaxies__

The `Galaxies` object is a collection of galaxies at the same redshift.

In a moment, we will see it is integral to the model-fitting API. 

For now, lets use it to create an image of a pair of merging galaxies, noting that a more concise API for creating
the galaxy is used below where the `Sersic` is passed directly to the `Galaxy` object.
"""
galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.5, 0.2), intensity=1.0, effective_radius=1.0, sersic_index=2.0
    ),
)

galaxies = ag.Galaxies(
    galaxies=[galaxy, galaxy_1],
)

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Units__

The units used throughout the galaxy structure literature vary, therefore lets quickly describe the units used in
**PyAutoGalaxy**.

Most distance quantities, like an `effective_radius` are quantities in terms of angles, which are defined in units
of arc-seconds. To convert these to physical units (e.g. kiloparsecs), we use the redshift of the galaxy and an 
input cosmology. A run through of all normal unit conversions is given in guides in the workspace outlined below.

The use of angles in arc-seconds has an important property, it means that calculations are independent of
the galaxy's redshifts and the input cosmology. This has a number of benefits, for example it makes it straight
forward to compare the properties of different galaxies even when the redshifts of the galaxies are unknown.

__Extensibility__

All of the objects we've introduced so far are highly extensible, for example a galaxy can be made up of any number of
light profiles and many galaxy objects can be combined into a galaxies object.

To further illustrate this, we create a merging galaxy system with 4 star forming clumps of light, using a 
`SersicSph` profile to make each spherical.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=0.2,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    disk=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.1,
        effective_radius=1.6,
    ),
    extra_galaxy_0=ag.lp.SersicSph(
        centre=(1.0, 1.0), intensity=0.5, effective_radius=0.2
    ),
    extra_galaxy_1=ag.lp.SersicSph(
        centre=(0.5, 0.8), intensity=0.5, effective_radius=0.2
    ),
    extra_galaxy_2=ag.lp.SersicSph(
        centre=(-1.0, -0.7), intensity=0.5, effective_radius=0.2
    ),
    extra_galaxy_3=ag.lp.SersicSph(
        centre=(-1.0, 0.4), intensity=0.5, effective_radius=0.2
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 1.0),
        ell_comps=(0.0, 0.1),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Galaxy Modeling__

Galaxy modeling is the process of fitting a physical model to imaging data in order to infer the structural
and photometric properties of galaxies, such as their light distribution, size, shape, and orientation.

The primary goal of **PyAutoGalaxy** is to make galaxy modeling **simple, scalable to large datasets, and fast**,
with GPU acceleration provided via JAX.

The animation below illustrates the galaxy modeling workflow. Many models are fitted to the data iteratively,
progressively improving the quality of the fit until the model closely reproduces the observed image.

NOTE: Placeholder showing strong lens modeling animation used currently.

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")

**Credit: Amy Etherington**

The next documentation page guides you through galaxy modeling for a variety of data types (e.g. CCD imaging at 
different resolutions) and scientific use-cases (e.g. galaxy morphology studies, bulge–disk decomposition).

__Simulations__

Simulating galaxy images is often essential, for example to:

- Practice galaxy modeling before working with real data.
- Generate large training sets (e.g. for machine learning).
- Test galaxy formation and structural models in a fully controlled environment.

The next documentation page guides you through how to simulate galaxies for different types of data
(e.g. CCD imaging) and different modeling goals (e.g. single-component galaxies, multi-component systems).

__Wrap Up__

This completes the introduction to **PyAutoGalaxy**, including a brief overview of the core API for galaxy light
profiles, galaxy modeling, and data simulation.

__What Data Type?__

If you are interested in modeling galaxies, you now need to decide what type of imaging data you want to work with:

- **CCD Imaging**: For image data from telescopes like Hubble, James Webb, or ground-based observatories,
  go to `imaging/start_here.ipynb`.

- **Interferometer**: For radio / sub-mm interferometer data from instruments like ALMA, where galaxies are
  observed via visibilities in the uv-plane, go to `interferometer/start_here.ipynb`.

- **Multi-Band Imaging**: For galaxies observed in multiple wavebands (e.g. colour gradients, stellar population
  studies), go to `multi_band/start_here.ipynb`.

__Google Colab__

You can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed.

This is a great way to get started quickly without needing to install **PyAutoGalaxy** on your own machine,
so you can check it is the right software for you before going through the installation process:

- [imaging/start_here.ipynb](https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace/blob/release/notebooks/imaging/start_here.ipynb>):
  Galaxy modeling with CCD imaging (e.g. Hubble, James Webb, ground-based telescopes).

- [interferometer/start_here.ipynb](https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace/blob/release/notebooks/interferometer/start_here.ipynb):
  Galaxy modeling with interferometer data (e.g. ALMA), fitting directly in the uv-plane.

- [multi_band/start_here.ipynb](https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace/blob/release/notebooks/multi/start_here.ipynb):
  Multi-band galaxy modeling to study colour gradients and wavelength-dependent structure.
  
__Still Unsure?__

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task. 
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you 
should just read through a few different notebooks and go from there.

__HowToGalaxy Lectures__

For experienced scientists, the run through above will have been a breeze. Concepts surrounding galaxy structure and 
morphology were already familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the 
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToGalaxy** Jupyter Notebook lectures are provide exactly this They are a 3+ chapter guide which thoroughly 
take you through the core concepts of galaxy light profiles, teach you the principles ofthe  statistical techniques 
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

If this sounds like it suits you, checkout the `autogalaxy_workspace/notebooks/howtogalaxy` package now, its it
recommended you go here before anywhere else!

__Features__

Here is a brief overview of the advanced features of **PyAutoGalaxy**. 

Firstly, brief one sentence descriptions of each feature are given, with more detailed descriptions below including 
links to the relevant workspace examples.

**Interferometry**: Modeling of interferometer data (e.g. ALMA, LOFAR) directly in the uv-plane.
**Multi-Wavelength**: Simultaneous analysis of imaging and / or interferometer datasets observed at different wavelengths.
**Ellipse Fitting**: Fitting ellipses to determine a galaxy's ellipticity, position angle and centre.
**Multi Gaussian Expansion (MGE)**: Decomposing a galaxy into hundreds of Gaussians, capturing more complex structures than simple light profiles.
**Shapelets**: Decomposing a galaxy into a set of shapelet orthogonal basis functions, capturing more complex structures than simple light profiles.
**Sky Background**: Including the background sky in the model to ensure robust fits to the outskirts of galaxies.
**Operated Light Profiles**: Assuming a light profile has already been convolved with the PSF, for when the PSF is a significant effect.
**Pixelizations**: Reconstructing a galaxy's on a mesh of pixels, to capture extremely irregular structures like spiral arms.


__Interferometry__

Modeling interferometer data from submillimeter (e.g. ALMA) and radio (e.g. LOFAR) observatories:

![ALMA Image](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/almacombined.png)

Visibilities data is fitted directly in the uv-plane, circumventing issues that arise when fitting a dirty image
such as correlated noise. This uses the non-uniform fast fourier transform algorithm
[PyNUFFT](https://github.com/jyhmiinlin/pynufft) to efficiently map the galaxy model images to the uv-plane.

Checkout the`autogalaxy_workspace/*/interferometer` package to get started.


__Multi-Wavelength__

Modeling imaging datasets observed at different wavelengths (e.g. HST F814W and F150W) simultaneously or simultaneously
analysing imaging and interferometer data:

![g-band](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/g_image.png)

![r-band](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/r_image.png)

The appearance of the galaxy changes as a function of wavelength, therefore multi-wavelength analysis means we can learn
more about the different components in a galaxy (e.g a redder bulge and bluer disk) or when imaging and interferometer
data are combined, we can compare the emission from stars and dust.

Checkout the `autogalaxy_workspace/*/multi` package to get started, however combining datasets is a more advanced
feature and it is recommended you first get to grips with the core API.


__Ellipse Fitting__

Ellipse fitting is a technique which fits many ellipses to a galaxy's emission to determine its ellipticity, position
angle and centre, without assuming a parametric form for its light (e.g. a Sersic profile):

![ellipse](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/ellipse.png)

This provides complementary information to parametric light profile fitting, for example giving insights on whether
the ellipticity and position angle are constant with radius or if the galaxy's emission is lopsided. 

There are also multipole moment extensions to ellipse fitting, which determine higher order deviations from elliptical 
symmetry providing even more information on the galaxy's structure.

The following paper describes the technique in detail: https://arxiv.org/html/2407.12983v1

Checkout `autogalaxy_workspace/notebooks/features/ellipse_fitting.ipynb` to learn how to use ellipse fitting.


__Multi Gaussian Expansion (MGE)__

An MGE decomposes the light of a galaxy into tens or hundreds of two dimensional Gaussians:

![MGE](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/mge.png)

In the image above, 30 Gaussians are shown, where their sizes go from below the pixel scale (in order to resolve
point emission) to beyond the size of the galaxy (to capture its extended emission).

Scientific Applications include capturing departures from elliptical symmetry in the light of galaxies, providing a 
flexible model to deblend the emission of point sources (e.g. quasars) from the emission of their host galaxy and 
deprojecting the light of a galaxy from 2D to 3D.

Checkout `autogalaxy_workspace/notebooks/features/multi_gaussian_expansion.ipynb` to learn how to use an MGE.


__Shapelets__

Shapelets are a set of orthogonal basis functions that can be combined the represent galaxy structures:

Scientific Applications include capturing symmetric structures in a galaxy which are more complex than a Sersic profile,
irregular and asymmetric structures in a galaxy like spiral arms and providing a flexible model to deblend the emission 
of point sources (e.g. quasars) from the emission of their host galaxy.

Checkout `autogalaxy_workspace/notebooks/features/shapelets.ipynb` to learn how to use shapelets.


__Sky Background__

When an image of a galaxy is observed, the background sky contributes light to the image and adds noise:

For detailed studies of the outskirts of galaxies (e.g. stellar halos, faint extended disks), the sky background must be
accounted for in the model to ensure robust and accurate fits.

Checkout `autogalaxy_workspace/notebooks/features/sky_background.ipynb` to learn how to use include the sky
background in your model.


__Operated Light Profiles__

An operated light profile is one where it is assumed to already be convolved with the PSF of the data, with the 
`Moffat` and `Gaussian` profiles common choices:

They are used for certain scientific applications where the PSF convolution is known to be a significant effect and
the knowledge of the PSF allows for detailed modeling abd deblending of the galaxy's light.

Checkout `autogalaxy_workspace/notebooks/features/operated_light_profiles.ipynb` to learn how to use operated profiles.


__Pixelizations__

A pixelization reconstructs a galaxy's light on a mesh of pixels, for example a rectangular mesh, Delaunay 
triangulation or Voronoi grid. 

These models are highly flexible and can capture complex structures in a galaxy's light that parametric models
like a Sersic profile cannot, for example spiral arms or asymmetric merging features.

The image below shows a non parametric of a galaxy observed in the Hubble Ultra Deep Field. Its bulge and disk are
fitted accurately using light profiles, whereas its asymmetric and irregular spiral arm features are accurately
captured using a rectangular mesh:

![HST Image](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/hstcombined.png)

Checkout `autogalaxy_workspace/notebooks/features/pixelizations.ipynb` to learn how to use a pixelization, however
this is a more advanced feature and it is recommended you first get to grips with the core API.


__Other:__

- Automated pipelines / database tools.
- Graphical models.
"""
