"""
Plots: Plotters
===============

This example illustrates the API for plotting using `Plotter` objects, which enable quick visualization of all
key quantities.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**Array2D:** Plot an `Array2D` object, which is a base object representing any 2D quantity (e.g. images, convergence, data).
**Grid2D:** Plot a `Grid2D` object, which is a base object representing a (y,x) grid of coordinates in 2D space.
**Tracer:** Plot a `Tracer` object, representing a tracer of light through the universe, including the mass of lens galaxies.
**Imaging:** Plot an `Imaging` object, representing an imaging dataset, including the data, noise-map and PSF.
**Fit Imaging:** Plot a `FitImaging` object, representing the fit of a model to an imaging dataset (including residuals, chi-squared and model image).
**Light Profile:** Plot a `LightProfile` object, representing the light of a galaxy.
**Mass Profile:** Plot a `MassProfile` object, representing the mass of a galaxy.
**Galaxy:** Plot a `Galaxy` object, which is a collection of light and mass profiles.
**Galaxies:** Plot a `Galaxies` object, which is a collection of galaxies.
**Interferometer:** Plot an `Interferometer` object, representing an interferometer dataset, including the data, noise-map and UV wavelengths.
**Fit Interferometer:** Plot a `FitInterferometer` object, representing the fit of a model to an interferometer dataset (including residuals, chi-squared and model image).
**Point Dataset:** Plot a `PointDataset` object, representing a point source dataset (e.g. lensed quasar, supernova).
**Fit Point Dataset:** Plot a `FitPointDataset` object, representing the fit of a model to a point source dataset (including residuals and chi-squared).

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

bulge = ag.lp.Sersic(
    centre=(0.0, -0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.05),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=2.0,
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, -1.0),
        ell_comps=(0.25, 0.1),
        intensity=0.1,
        effective_radius=0.8,
        sersic_index=2.5,
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

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Array2D__

The `Array2D` object is base object which represents any 2D quantity (e.g. images, convergence, data).

It can be plotted using an `Array2DPlotter` and calling the `figure` method.
"""
array_plotter = aplt.Array2DPlotter(array=dataset.data)
array_plotter.figure_2d()

"""
__Grid2D__

The `Grid2D` object is a base object which represents a (y,x) grid of coordinates in 2D space, 9including image-plane
and source-plane grids.

It can be plotted using a `Grid2DPlotter` and calling the `figure` method.
"""
grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
__Galaxy__

A `Galaxy` is a collection of light and mass profiles, and can be plotted using a `GalaxyPlotter`.

We first pass a galaxy and grid to a `GalaxyPlotter` and call various `figure_*` methods to plot different
attributes in 1D and 2D.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
galaxy_plotter.figures_2d(
    image=True,
)

"""
The `GalaxyPlotter` also has subplot method that plot each individual `Profile` in 2D as well as a 1D plot showing all
`Profiles` together.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
We can plot 1D profiles, which display a properties of the galaxy in 1D as a function of radius.

For the 1D plot of each profile, the 1D grid of (x,) coordinates is centred on the profile and aligned with the 
major-axis. 

Because the `GalaxyPlotter` above has an input `Grid2D` object, the 1D grid of radial coordinates used to plot
these quantities is derived from this 2D grid. The 1D grid corresponds to the longest radial distance from the centre
of the galaxy's light or mass profiles to the edge of the 2D grid.
"""
galaxy_plotter.figures_1d(image=True)

"""
If we want a specific 1D grid of a certain length over a certain range of coordinates, we can manually input a `Grid1D`
object.

Below, we create a `Grid1D` starting from 0 which plots the image and convergence over the radial range 0.0" -> 10.0".
"""
grid_1d = ag.Grid1D.uniform_from_zero(shape_native=(1000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)

galaxy_plotter.figures_1d(image=True)

"""
Using a `Grid1D` which does not start from 0.0" plots the 1D quantity with both negative and positive radial 
coordinates.

This plot isn't particularly useful, but it shows how 1D plots work.
"""
grid_1d = ag.Grid1D.uniform(shape_native=(1000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)

galaxy_plotter.figures_1d(image=True)

"""
We can also plot decomposed 1D profiles, which display the 1D quantity of every individual light and / or mass profiles. 

For the 1D plot of each profile, the 1D grid of (x) coordinates is centred on the profile and aligned with the 
major-axis. This means that if the galaxy consists of multiple profiles with different centres or angles the 1D plots 
are defined in a common way and appear aligned on the figure.

We'll plot this using our masked grid above, which converts the 2D grid to a 1D radial grid used to plot every
profile individually.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)

galaxy_plotter.figures_1d_decomposed(image=True)

"""
Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light and mass models estimated via a 
model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of a model-fit (the 
database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` with errors, which are plotted as vertical lines.

Below, we manually input two `Galaxy` objects with light profiles that clearly show these errors on the figure.
"""
galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=[galaxy_0, galaxy_1], grid=grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(image=True)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(image=True)

"""
A galaxy consists of light profiles, and their centres can be extracted and plotted over the image. 
The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter`
describes how to plot these visuals over images.

__Galaxies__

A `Galaxies` is a collection of galaxies, and can be plotted using a `GalaxiesPlotter`.

We first pass a galaxies and grid to a `GalaxiesPlotter` and call various `figure_*` methods to plot different
attributes in 1D and 2D.

We separate the `image_plane_galaxies` and `source_plane_galaxies` so that we can plot them separately, as they
are often at different redshifts and thus have different properties.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
A subplot of the above quantaties can be plotted.
"""
galaxies_plotter.subplot_galaxies()

"""
A subplot of the image of the galaxies in the plane can also be plotted.
"""
galaxies_plotter.subplot_galaxy_images()

"""
We can also plot specific images of galaxies in the plane.
"""
galaxies_plotter.figures_2d_of_galaxies(image=True, galaxy_index=0)

"""
A galaxy consists of light profiles, and their centres can be extracted and plotted over the image. 
The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter`,
describes how to plot these visuals over images.

__Imaging__

The `Imaging` object is a base object which represents an imaging dataset, including the data, noise-map and PSF.

It can be plotted using an `ImagingPlotter` and calling the `figure_*` methods to plot different attributes.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    psf=True,
)

"""
The `ImagingPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""
__Fit Imaging__

The `FitImaging` object is a base object which represents the fit of a model to an imaging dataset, including the
residuals, chi-squared and model image.

It can be plotted using a `FitImagingPlotter` and calling the `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
It can plot of the model image of an input galaxies.
"""
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, model_image=True)
fit_plotter.figures_2d_of_galaxies(galaxy_index=1, model_image=True)

"""
It can plot the image of galaxies with all other model images subtracted.
"""
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, subtracted_image=True)
fit_plotter.figures_2d_of_galaxies(galaxy_index=1, subtracted_image=True)

"""
The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_galaxies(galaxy_index=1)

"""
By default, the `residual_map` and `normalized_residual_map` use a symmetric colormap. 

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    residual_map=True,
    normalized_residual_map=True,
)

"""
__Light Profile__

Light profiles have dedicated plotters which can plot their attributes in 1D and 2D.

We first pass a light profile and grid to a `LightProfilePlotter` and call various `figure_*` methods to 
plot different attributes in 1D and 2D.
"""
light_profile_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=grid)
light_profile_plotter.figures_1d(image=True)
light_profile_plotter.figures_2d(image=True)

"""
Using a `LightProfilePDFPlotter`, we can make 1D plots that show the errors of a light model estimated via a model-fit. 

Here, the `light_profile_pdf_list` is a list of `LightProfile` objects that are drawn randomly from the PDF of a 
model-fit (the database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` with errors, which are plotted as vertical lines.

Below, we manually input two `LightProfiles` that clearly show these errors on the figure.
"""
light_profile_pdf_plotter = aplt.LightProfilePDFPlotter(
    light_profile_pdf_list=[bulge], grid=grid, sigma=3.0
)
light_profile_pdf_plotter.figures_1d(image=True)

"""
A light profile centre can be extracted and plotted over the image. The `visuals.ipynb` notebook, under the 
section `LightProfileCentreScatter` describes how to plot these visuals over images.

__Interferometer__

The `Interferometer` object is a base object which represents an interferometer dataset, including the data, noise-map,
and UV wavelengths.

First, we load one.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "interferometer" / dataset_name

real_space_mask = ag.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerNUFFT,
)

"""
We now pass the interferometer to an `InterferometerPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    u_wavelengths=True,
    v_wavelengths=True,
    uv_wavelengths=True,
    amplitudes_vs_uv_distances=True,
    phases_vs_uv_distances=True,
)

"""
The dirty images of the interferometer dataset can also be plotted, which use the transformer of the interferometer 
to map the visibilities, noise-map or other quantity to a real-space image.
"""
dataset_plotter.figures_2d(
    dirty_image=True,
    dirty_noise_map=True,
    dirty_signal_to_noise_map=True,
)

"""
The `InterferometerPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Fit Interferometer__

The `FitInterferometer` object is a base object which represents the fit of a model to an interferometer dataset,
including the residuals, chi-squared and model image.

We now create one.
"""
galaxy = ag.Galaxy(
    redshift=1.0,
    bulge=ag.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)

"""
We now pass the FitInterferometer to an `FitInterferometerPlotter` and call various `figure_*` methods 
to plot different attributes.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_data=True,
    residual_map_real=True,
    residual_map_imag=True,
    normalized_residual_map_real=True,
    normalized_residual_map_imag=True,
    chi_squared_map_real=True,
    chi_squared_map_imag=True,
)

"""
The dirty images of the interferometer fit can also be plotted, which use the transformer of the interferometer
to map the visibilities, noise-map, residual-map or other quantitiy to a real-space image.

Bare in mind the fit itself uses the visibilities and not the dirty images, so these images do not provide a direct
visualization of the fit itself. However, they are easier to inspect than the fits plotted above which are in Fourier
space and make it more straight forward to determine if an unphysical lens model is being fitted.
"""
fit_plotter.figures_2d(
    dirty_image=True,
    dirty_noise_map=True,
    dirty_signal_to_noise_map=True,
    dirty_model_image=True,
    dirty_residual_map=True,
    dirty_normalized_residual_map=True,
    dirty_chi_squared_map=True,
)

"""
The `FitInterferometerPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
The plane images can be combined to plot the appearance of the galaxy in real-space.
"""
fit_plotter.subplot_fit_real_space()

"""
By default, the `ditry_residual_map` and `dirty_normalized_residual_map` use a symmetric colormap.

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    dirty_residual_map=True,
    dirty_normalized_residual_map=True,
)

"""
Finish.
"""
