"""
PyAutoGalaxy
============

This notebook is the starting point for all new **PyAutoGalaxy** users!

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

![HST Image](https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/hstcombined.png)

**PyAutoGalaxy** has three core aims:

- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse Fitting, Irregular Meshes) that go beyond just simple Sersic fitting (which is supported too!).

- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be fitted independently or simultaneously.

- **Big Data**: Scaling automated analysis to extremely large datasets, using tools like an SQL database to build a scalable scientific workflow.

This notebook gives an overview of **PyAutoGalaxy**'s API, core features and details of the autogalaxy_workspace.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

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
**PyAutoLens**.

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
__Simulating Data__

The galaxy images above are **not** what we would observe if we looked at the sky through a telescope.

In reality, images of galaxies are observed using a telescope and detector, for example a CCD Imaging device attached
to the Hubble Space Telescope.

To make images that look like realistic Astronomy data, we must account for the effects like how the length of the
exposure time change the signal-to-noise, how the optics of the telescope blur the galaxy's light and that
there is a background sky which also contributes light to the image and adds noise.

The `SimulatorImaging` object simulates this process, creating realistic CCD images of galaxies using the `Imaging`
object.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    background_sky_level=1.0,
    psf=ag.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=0.05),
    add_poisson_noise_to_data=True,
)

"""
Once we have a simulator, we can use it to create an imaging dataset which consists of an image, noise-map and 
Point Spread Function (PSF) by passing it a galaxies and grid.

This uses the galaxies above to create the image of the galaxy and then add the effects that occur during data
acquisition.

This data is used below to illustrate model-fitting, so lets simulate a very simple image of a galaxy using
just a single Sersic light profile.
"""
galaxies = ag.Galaxies(
    galaxies=[
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp.Sersic(
                centre=(0.0, 0.0),
                ell_comps=(0.1, 0.2),
                intensity=1.0,
                effective_radius=0.8,
                sersic_index=2.0,
            ),
        )
    ]
)

dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
__Observed Dataset__

We now have an `Imaging` object, which is a realistic representation of the data we observe with a telescope.

We use the `ImagingPlotter` to plot the dataset, showing that it contains the observed image, but also other
import dataset attributes like the noise-map and PSF.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
If you have come to **PyAutoGalaxy** to perform interferometry, the API above is easily adapted to use 
a `SimulatorInterferometer` object to simulate an `Interferometer` dataset instead.

However, you should finish reading this notebook before moving on to the interferometry examples, to get a full
overview of the core **PyAutoGalaxy** API.

__Masking__

We are about to fit the data with a model, but first must define a mask, which defines the regions of the image that 
are used to fit the data and which regions are not.

We create a `Mask2D` object which is a 3.0" circle, whereby all pixels within this 3.0" circle are used in the 
model-fit and all pixels outside are omitted. 

Inspection of the dataset above shows that no signal from the galaxy is observed outside of this radius, so this is a 
sensible mask.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,  # The mask's shape must match the dataset's to be applied to it.
    pixel_scales=dataset.pixel_scales,  # It must also have the same pixel scales.
    radius=3.0,  # The mask's circular radius [units of arc-seconds].
)

"""
Combine the imaging dataset with the mask.
"""
dataset = dataset.apply_mask(mask=mask)

"""
When we plot a masked dataset, the removed regions of the image (e.g. outside the 3.0") are automatically set to zero
and the plot axis automatically zooms in around the mask.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
__Fitting__

We are now at the point a scientist would be after observing a galaxy - we have an image of it, have used to a mask to 
determine where we observe signal from the galaxy, but cannot make any quantitative statements about its morphology.

We therefore must now fit a model to the data. This model is a representation of the galaxy's light, and we seek a way
to determine whether a given model provides a good fit to the data.

A fit is performing using a `FitImaging` object, which takes a dataset and galaxies object as input and determine if 
the galaxies are a good fit to the data.
"""
fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
The fit creates `model_data`, which is the image of the galaxy including effects which change its appearance
during data acquisition.

For example, by plotting the fit's `model_data` and comparing it to the image of the galaxies obtained via
the `GalaxiesPlotter`, we can see the model data has been blurred by the dataset's PSF.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=fit.galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d(model_image=True)

"""
The fit also creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset`s `image`.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.
 
We can plot all 3 of these on a subplot that also includes the data, signal-to-noise map and model data.

In this example, the galaxies used to simulate the data are used to fit it, thus the fit is good and residuals are minimized.
"""
fit_plotter.subplot_fit()

"""
The overall quality of the fit is quantified with the `log_likelihood`.
"""
print(fit.log_likelihood)

"""
If you are familiar with statistical analysis, this quick run-through of the fitting tools will make sense and you
will be familiar with concepts like model data, residuals and a likelihood. 

If you are less familiar with these concepts, I recommend you finish this notebook and then go to the fitting API
guide, which explains the concepts in more detail and provides a more thorough overview of the fitting tools.

The take home point is that **PyAutoGalaxy**'s API has extensive tools for fitting models to data and visualizing the
results, which is what makes it a powerful tool for studying the morphologies of galaxies.

__Modeling__

The fitting tools above are used to fit a model to the data given an input set of galaxies. Above, we used the true
galaxies used to simulate the data to fit the data, but we do not know what this "truth" is in the real world and 
is therefore not something a real scientist can do.

Modeling is the processing of taking a dataset and inferring the model that best fits the data, for example
the galaxy light profile(s) that best fits the light observed in the data or equivalently the combination
of Sersic profile parameters that maximize the likelihood of the fit.

Galaxy modeling uses the probabilistic programming language **PyAutoFit**, an open-source project that allows complex 
model fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you 
are interested in developing your own software to perform advanced model-fitting:

https://github.com/rhayes777/PyAutoFit

We import **PyAutoFit** separately to **PyAutoGalaxy**:
"""
import autofit as af

"""
We now compose the galaxy model using `af.Model` objects. 

These behave analogously to the `Galaxy`, `Galaxies` and `LightProfile` objects above, however when using a `Model` 
their parameter values are not specified and are instead determined by a fitting procedure.

We will fit our galaxy data with a model which has one galaxy where:

 - The galaxy's bulge is a `Sersic` light profile. 
 - The galaxy's disk is a `Exponential` light profile.
 - The redshift of the galaxy is fixed to 0.5.
 
The light profiles below are linear light profiles, input via the `lp_linear` module. These solve for the intensity of
the light profiles via linear algebra, making the modeling more efficient and accurate. They are explained in more
detail in other workspace examples, but are a key reason why modeling with **PyAutoGalaxy** performs well and
can scale to complex models.
"""
galaxy_model = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.Sersic,  # Note the use of `lp_linear` instead of `lp`.
    disk=ag.lp_linear.Exponential,  # This uses linear light profiles explained in the modeling `start_here` example.
)

"""
By printing the `Model`'s we see that each parameters has a prior associated with it, which is used by the
model-fitting procedure to fit the model.
"""
print(galaxy_model)

"""
We input the galaxy model above into a `Collection`, which is the model we will fit. 

Note how we could easily extend this object to compose more complex models containing many galaxies.
"""
model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))

"""
The `info` attribute shows the model information in a more readable format:
"""
print(model.info)

"""
We now choose the 'non-linear search', which is the fitting method used to determine the light profile parameters that 
best-fit the data.

In this example we use [nautilus](https://nautilus-sampler.readthedocs.io/en/stable/), a nested sampling algorithm 
that in our experience has proven very effective at galaxy modeling.
"""
search = af.Nautilus(name="start_here")

"""
To perform the model-fit, we create an `AnalysisImaging` object which contains the `log_likelihood_function` that the
non-linear search calls to fit the galaxy model to the data.

The `AnalysisImaging` object is expanded on in the modeling `start_here` example, but in brief performs many useful
associated with modeling, including outputting results to hard-disk and visualizing the results of the fit.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
Nautilus samples, model parameters, visualization) to your computer's storage device.

However, the galaxy modeling of this system takes a minute or so. Therefore, to save time, we have commented out 
the `fit` function below so you can skip through to the next section of the notebook. Feel free to uncomment the code 
and run the galaxy modeling yourself!

Once a model-fit is running, **PyAutoGalaxy** outputs the results of the search to storage device on-the-fly. This
includes galaxy model parameter estimates with errors non-linear samples and the visualization of the best-fit galaxy
model inferred by the search so far.
"""
# result = search.fit(model=model, analysis=analysis)

"""
The animation below shows a slide-show of the galaxy modeling procedure. Many galaxy models are fitted to the data over
and over, gradually improving the quality of the fit to the data and looking more and more like the observed image.

NOTE, the animation of a non-linear search shown below is for a strong gravitational lens using **PyAutoGalaxy**'s 
child project **PyAutoLens**. Updating the animation to show a galaxy model-fit is on the **PyAutoGalaxy** to-do list!

We can see that initial models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")

**Credit: Amy Etherington**

__Results__

The fit returns a `Result` object, which contains the best-fit galaxies and the full posterior information of the 
non-linear search, including all parameter samples, log likelihood values and tools to compute the errors on the 
galaxy model.

Using results is explained in full in the `guides/results` section of the workspace, but for a quick illustration
the commented out code below shows how easy it is to plot the fit and posterior of the model.
"""
# fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
# fit_plotter.subplot_fit()

# plotter = aplt.NestPlotter(samples=result.samples)
# plotter.corner_cornerpy()

"""
We have now completed the API overview of **PyAutoGalaxy**. This notebook has given a brief introduction to the core
API for creating galaxies, simulating data, fitting data and performing galaxy modeling.

__New User Guide__

Now you have a basic understanding of the **PyAutoGalaxy** API, you should read the new user guide on the readthedocs
to begin navigating the different examples in the workspace and learning how to use **PyAutoGalaxy**:

https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html

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
**pixelizations**: Reconstructing a galaxy's on a mesh of pixels, to capture extremely irregular structures like spiral arms.


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
