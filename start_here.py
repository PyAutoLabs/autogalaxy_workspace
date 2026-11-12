"""
PyAutoGalaxy
=============

**PyAutoGalaxy** models galaxy structure and morphology, from bulge–disc decomposition to comparisons across wavebands. You can fit individual galaxies or large samples, using imaging and interferometer observations.

![Galaxy morphology examples from Hubble imaging](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/paper/hstcombined.png)

__Using This Guide__

This notebook gives an overview of the **PyAutoGalaxy Python API**, including how to import `autogalaxy`, create light profiles and galaxies, and simulate and fit galaxy images. You can read through the examples or run the cells below, either locally or in your web browser using [Google Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/start_here.ipynb).

If you use AI, the [PyAutoGalaxy AI Assistant](https://github.com/PyAutoLabs/autogalaxy_assistant) can give you an interactive walkthrough. Paste this prompt into Claude Code, Codex or another AI coding agent:

> I want to study galaxy structure and morphology with PyAutoGalaxy
> (https://github.com/PyAutoLabs/PyAutoGalaxy) and the
> autogalaxy_assistant (https://github.com/PyAutoLabs/autogalaxy_assistant).
>
> First clone the assistant repository, cd into it and follow its AGENTS.md.
> Begin the "start here" guide for a new user.

See [AI Options](https://github.com/PyAutoLabs/PyAutoScientist/blob/main/AI_OPTIONS.md) for tools and setup guidance.

__Google Colab Setup__

Run this cell to install PyAutoGalaxy and download the files needed for this notebook on Google Colab.
If you’re running locally or in HTML, it leaves your installation unchanged.
"""

from autogalaxy import setup_colab
setup_colab.for_autogalaxy(raise_error_if_not_gpu=False)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
import autogalaxy as ag
import autogalaxy.plot as aplt

import matplotlib.pyplot as plt
from pathlib import Path

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

aplt.plot_grid(grid=grid, title="Uniform Grid")

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

By using the plot module, the image is improved: axis units are scaled to arc-seconds, a color-bar is added,
descriptive labels are included, etc.

The plot module is highly customizable and designed to make it straight forward to create clean and informative figures
for fits to large datasets.
"""
aplt.plot_array(
    array=sersic_light_profile.image_2d_from(grid=grid),
    title="Sersic Light Profile Image",
)

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

aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Galaxy Image")

"""
The individual light profiles of the galaxy can be plotted on a subplot.
"""
aplt.subplot_galaxy_light_profiles(galaxy=galaxy, grid=grid)

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

aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Galaxies Image")

"""
__JAX__

**PyAutoGalaxy** runs on either **NumPy** (the default) or **JAX** (Google's
array library with GPU support and just-in-time compilation). JAX makes
galaxy model-fitting 10-100x faster on large grids — sometimes more on GPU.

You do not have to do anything to use it. JAX installs with `autogalaxy`
by default (`pip install autogalaxy`), and the analysis objects you'll
meet in the `__Galaxy Modeling__` section below use JAX automatically. The first time a model-fit starts you'll see a one-time
log line like `JAX: Applying vmap and jit to likelihood function -- may
take a few seconds.` — that's JAX compiling the likelihood function, after
which every evaluation re-uses the compiled code. If JAX is not installed,
the analysis warns once and falls back to NumPy automatically.

That is all a new user needs to know: install the extra, and model-fitting
is fast. Everything else — disabling JAX, writing `@jax.jit` code yourself
and how JAX changes the arrays inside results — is covered in the
`scripts/guides/using_jax.py` guide.

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

aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Galaxies Image")

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

__Where To Next__

This guide helps you choose a **PyAutoGalaxy example notebook** for your observations. Start with the galaxies you need to model together, then choose your data type. Each example includes Python code you can read, run and adapt.

If you use AI, describe your galaxies and observations to the [PyAutoGalaxy AI Assistant](https://github.com/PyAutoLabs/autogalaxy_assistant). It can help you choose an example and work through it interactively.

__Which galaxies need modelling?__

Choose an example based on how much their light overlaps and how many galaxies you need to fit together.

![Illustrations of an isolated disk galaxy, two galaxies with overlapping light, and a cluster field with a bright central galaxy and many members.](https://raw.githubusercontent.com/PyAutoLabs/autogalaxy_workspace/main/images/new_user_guide/galaxy_systems.png)

*AI-generated conceptual illustration, not observations or fitted models. Panels are not shown at the same angular scale.*

__Single galaxy__

One galaxy is the target. Mask unrelated neighbours whose light does not need to be modelled jointly. Start with your data type below.

[Choose your data type below](#what-dataset-type)

__Blended galaxies__

Two or more galaxies have overlapping light. Fit a separate light model for each galaxy at the same time.

[Notebook ](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/notebooks/multi_galaxy/start_here.ipynb)

__Cluster field__

Model a bright central galaxy and a catalogue of member galaxies. The example uses catalogue photometry to constrain fainter members and reduce the number of free parameters.

[Notebook ](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/cluster/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/notebooks/cluster/start_here.ipynb)

This cluster example models the galaxies’ light. For strong-lensing mass models and background-source images, use [PyAutoLens](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html).

<a id="what-dataset-type"></a>

__What dataset type?__

For a single galaxy, choose the example for your observations. For blended galaxies or cluster fields, begin with the system-specific guide above.

![A pixelated galaxy image, interferometer sampling points in the u–v plane, and illustrative views of the same galaxy in three wavebands.](https://raw.githubusercontent.com/PyAutoLabs/autogalaxy_workspace/main/images/new_user_guide/dataset_types.png)

*AI-generated schematic, not measured data. The interferometer panel shows sampling locations; multi-band images illustrate how the relative prominence of a galaxy’s components can change with wavelength.*

__Imaging__

Images from Hubble, JWST or ground-based telescopes. Model the distribution of starlight using light profiles.

[Notebook ](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/notebooks/imaging/start_here.ipynb)

__Interferometer__

Radio or submillimetre observations, such as ALMA data. Fit the measured visibilities directly.

[Notebook ](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/notebooks/interferometer/start_here.ipynb)

__Multi-band imaging__

Images of the same galaxy in several wavebands. Fit them together to study colour gradients and wavelength-dependent structure.

[Notebook ](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb)·[Google Colab ](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.12.1/notebooks/multi_dataset/start_here.ipynb)

Read each notebook on GitHub, open it locally, or use its Google Colab link to run it in your browser. In Colab, run the setup cell first to install the software and prepare the example.

__Still unsure?__

Tell the [AI Assistant](https://github.com/PyAutoLabs/autogalaxy_assistant) which galaxies overlap in your field and what observations you have. It can help you identify a starting example.

__Learn the science__

[HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy) teaches galaxy structure and Bayesian modelling through worked notebooks. Use it for a slower explanation of the concepts behind these examples.

__Next steps__

Once you have chosen an example, explore the [Features guide](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_3_features.html) to find the modelling tools relevant to your science.

__Features__

Here is a brief overview of the advanced features of **PyAutoGalaxy**. 

Firstly, brief one sentence descriptions of each feature are given, with more detailed descriptions below including 
links to the relevant workspace examples.

**Linear Light Profiles**: Solve brightnesses while sampling galaxy geometry.

**Extra Galaxies**: Model neighbouring light that overlaps the target.

**Point Source**: Simulate unresolved nuclear emission in an extended host.

**Interferometry**: Modeling of interferometer data (e.g. ALMA, LOFAR) directly in the uv-plane.
**Multi-Wavelength**: Simultaneous analysis of imaging and / or interferometer datasets observed at different wavelengths.
**Ellipse Fitting**: Fitting ellipses to determine a galaxy's ellipticity, position angle and centre.
**Multi Gaussian Expansion (MGE)**: Describe central and extended galaxy light with a flexible sum of Gaussian components.
**Shapelets**: Decomposing a galaxy into a set of shapelet orthogonal basis functions, capturing more complex structures than simple light profiles.
**Sky Background**: Fit uncertain sky levels alongside faint galaxy outskirts.
**Operated Light Profiles**: Fit compact imaging emission whose profile already includes its PSF response.
**Pixelizations**: Reconstruct irregular galaxy light on a regularized mesh.


__Interferometry__

Study the radio and submillimeter structure of galaxies by modeling interferometer data from submillimeter (e.g. ALMA) and radio (e.g. LOFAR) observatories:

![ALMA Image](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/paper/almacombined.png)

Visibilities data is fitted directly in the uv-plane, circumventing issues that arise when fitting a dirty image
such as correlated noise. This uses the non-uniform fast fourier transform algorithm
[nufftax](https://github.com/GragasLab/nufftax) to efficiently map the galaxy model images to the uv-plane.

Checkout the`autogalaxy_workspace/*/interferometer` package to get started.


For flexible galaxy-light modeling, see [MGE](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/start_here.ipynb) and [pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/pixelization/start_here.ipynb).
Inspect the sky reconstruction as well as the visibility fit; a dirty image is not the likelihood dataset.


__Multi-Wavelength__

Modeling imaging datasets observed at different wavelengths (e.g. HST F814W and F150W) simultaneously or simultaneously
analysing imaging and interferometer data:

![g-band](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/g_image.png)

![r-band](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/r_image.png)

The appearance of the galaxy changes as a function of wavelength, therefore multi-wavelength analysis means we can learn
more about the different components in a galaxy (e.g a redder bulge and bluer disk) or when imaging and interferometer
data are combined, we can compare the emission from stars and dust.

Checkout the `autogalaxy_workspace/*/multi_dataset` package to get started, however combining datasets is a more advanced
feature and it is recommended you first get to grips with the core API.


![Imaging, visibility sampling and multi-band observations](images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic, not observed data. Visibility sampling positions are distinct from
measured amplitudes, and separate bands need not share the same light distribution.

Use [wavelength dependence](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/start_here.ipynb) for linked bulge and disk radii, [pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/pixelization/start_here.ipynb) for per-band
light reconstructions, or [imaging and interferometer](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/start_here.ipynb) for complementary emission.
The [one-by-one example](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/start_here.ipynb) shows parameter transfer; fixing an earlier estimate does not propagate its full uncertainty.


__Ellipse Fitting__

Ellipse fitting is a technique which fits many ellipses to a galaxy's emission to determine its ellipticity, position
angle and centre, without assuming a parametric form for its light (e.g. a Sersic profile):

![ellipse](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/ellipse.png)

This provides complementary information to parametric light profile fitting, for example giving insights on whether
the ellipticity and position angle are constant with radius or if the galaxy's emission is lopsided. 

There are also multipole moment extensions to ellipse fitting, which determine higher order deviations from elliptical 
symmetry providing even more information on the galaxy's structure.

The following paper describes the technique in detail: https://arxiv.org/html/2407.12983v1

Checkout `autogalaxy_workspace/notebooks/features/ellipse_fitting.ipynb` to learn how to use ellipse fitting.


__Linear Light Profiles__

Fit galaxy structure with fewer sampled parameters by solving component brightnesses at every likelihood evaluation. This is especially useful for bulge–disk decomposition and models containing many overlapping light components: the search explores their geometry while their intensities are solved together.

![Linear Light Profiles](images/features/linear_light_profiles.png)

Conceptual illustration: the paths symbolize different sampled parameter spaces, not measured convergence rates or removal of physical degeneracies.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

__Extra Galaxies__

Measure a target galaxy's light in a crowded field by modeling the neighbouring galaxies whose emission overlaps the fitted region. Separating those contributions helps protect the target's inferred size, shape and flux from contamination.

![Single, blended and cluster-field galaxy-light environments](images/new_user_guide/galaxy_systems.png)

Reused new-user-guide schematic: these are light-modeling environments, not gravitational-lensing mass models.

See [the imaging introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb) and [the multi-galaxy introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_galaxy/features/extra_galaxies/start_here.ipynb) for component choices and fixed-centre companions.

__Point Source__

Explore how an unresolved nucleus contributes to a galaxy image and affects measurements of its extended host. The point-source component supplies a position and flux; the instrumental response determines its observed footprint.

![Point Source](images/features/point_source.png)

Conceptual illustration of added unresolved nuclear emission. The bright footprint is set by the response; it does not resolve the intrinsic nucleus.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

The point-source folder currently supplies a simulator, not a complete inference workflow. It demonstrates
an unresolved nuclear component in an extended host; the PSF determines its observed footprint.

__Multi Gaussian Expansion (MGE)__

Describe a galaxy's central light and extended envelope with a flexible sum of Gaussian profiles. An MGE is useful when one simple profile leaves structured residuals, or when you need a smooth light model from which to measure the combined luminosity and radial structure.

![Multi Gaussian Expansion](images/features/mge_illustration.png)

Conceptual illustration: the displayed Gaussians are components of one light model. The inset is a decomposition, not a spatial magnification of the galaxy core.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

The existing Gaussian-component grid shows the widths used to describe central and extended light:

![Gaussian component teaching figure](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/mge.png)

Subpixel basis widths do not establish resolved subpixel structure. Gaussian components are mathematical
building blocks; measure luminosity and size from their combined light. Deprojection requires additional
geometric assumptions and is not established by a good projected fit alone.


__Shapelets__

Describe departures from a smooth galaxy profile with a flexible set of spatial basis functions. Shapelets are useful for studying asymmetric morphology and structured residuals while retaining a compact parameterization of the basis geometry.

![Shapelets](images/features/shapelets_illustration.png)

Reused conceptual illustration. “Source model” here means the emitting galaxy; there is no gravitational-lensing source plane. The summed basis represents light, not separate physical structures.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/shapelets/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

![Shapelet basis teaching grid](images/features/shapelets_basis.png)

Keep the basis grid alongside the conceptual illustration. Signed basis coefficients need not imply
negative total light; inspect the sum. The basis order, scale and shared geometry determine its flexibility.


__Sky Background__

Measure faint galaxy outskirts while fitting an uncertain sky level alongside the galaxy light. This is useful for retained backgrounds and residual sky-subtraction offsets, where a small background error can otherwise alter the inferred size or outer profile.

![Sky Background](images/features/sky_background.png)

Conceptual illustration: the separated layers are additive light and background components, not physical planes in space.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

The example fits a constant sky level. Fitting its mean does not remove photon noise. The supplied
simulation uses 5.0 electrons per second, at the upper edge of the example prior [0.0, 5.0]; choose
appropriate bounds when adapting the analysis.


__Operated Light Profiles__

Separate compact nuclear emission from an extended host using a profile whose imaging PSF response is already included. This is useful when the component you want to fit describes the observed compact glow, rather than an intrinsic light distribution that still needs to be blurred.

![Operated Light Profile](images/features/operated_light_profile.png)

Conceptual illustration of the imaging response: the compact component already includes its PSF blur. The extended galaxy is still treated with its appropriate response.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

This PSF interpretation applies to imaging. In the [interferometer example](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/start_here.ipynb),
the compact sky profile is Fourier transformed into visibilities; there is no imaging PSF convolution to bypass.


__Pixelizations__

Reconstruct galaxy light that is difficult to describe with a small set of analytic profiles, such as asymmetric or clumpy emission. A pixelization solves brightness on a mesh, allowing the morphology of the emission to be studied while accounting for the observation's response and noise.

![Pixelization](images/features/pixelization_illustration.png)

Conceptual illustration, not a fitted reconstruction. The two views represent the same galaxy brightness; the schematic triangular overlay is not a claim about the mesh used in every example.

Read the [feature introduction](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb) for the scientific use cases,
model assumptions and worked-example routes.

Start with the smooth-bulge-plus-pixelized-light imaging example for irregular galaxy emission. The
linked galaxy_reconstruction.py shows how to inspect and export the inferred light. Regularization and
mesh choices affect what structure the data can support.

The existing Hubble Ultra Deep Field teaching figure illustrates a combined parametric and pixelized model:

![Combined profile and pixelized galaxy-light model](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/paper/hstcombined.png)


__Graphical Models__

The examples above fit each galaxy dataset one-by-one. However, many galaxy properties are shared across a
sample (e.g. the population's distribution of Sersic indices or sizes), and fitting galaxies independently does
not exploit this shared structure.

Graphical models fit multiple datasets simultaneously, explicitly defining which parameters are unique to each
galaxy and which are shared across the sample. Hierarchical extensions assume parameters are drawn from a parent
distribution, whose properties are inferred from the full sample, extracting significantly more information
than one-by-one fitting. This is a powerful tool for galaxy evolution studies of large samples, for example
measuring how the galaxy population's structural properties vary with mass or redshift.

Checkout `autofit_workspace/*/features/graphical_models.py` to learn how to compose and fit a graphical model;
the API applies directly to the galaxy fits shown throughout this workspace.
__Choose a Feature by Dataset__

Each introduction below explains the scientific use cases, teaching concepts and example-specific choices.

**Imaging**

- [Extra Galaxies](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb)
- [Linear Light Profiles](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb)
- [Multi Gaussian Expansion](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb)
- [Operated Light Profile](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/start_here.ipynb)
- [Pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb)
- [Point Source](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/start_here.ipynb)
- [Shapelets](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/shapelets/start_here.ipynb)
- [Sky Background](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/start_here.ipynb)

**Interferometer**

- [Extra Galaxies](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/start_here.ipynb)
- [Linear Light Profiles](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/start_here.ipynb)
- [Multi Gaussian Expansion](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/start_here.ipynb)
- [Operated Light Profile](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/start_here.ipynb)
- [Pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/pixelization/start_here.ipynb)
- [Shapelets](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/shapelets/start_here.ipynb)

**Multiple Datasets**

- [Dataset Offsets](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/start_here.ipynb)
- [Imaging And Interferometer](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/start_here.ipynb)
- [One By One](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/start_here.ipynb)
- [Pixelization](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/pixelization/start_here.ipynb)
- [Same Wavelength](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/start_here.ipynb)
- [Wavelength Dependence](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/start_here.ipynb)

**Blended Galaxies**

- [Extra Galaxies](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_galaxy/features/extra_galaxies/start_here.ipynb)

"""
