"""
Plots: FitImagingPlotter
========================

This example illustrates how to plot an `FitImaging` object using an `FitImagingPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a galaxy as an `Imaging` object.
"""
dataset_name = "sersic_x2"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Fit__

We now mask the data and fit it with a `Plane` to create a `FitImaging` object.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

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
__Figures__

We now pass the FitImaging to an `FitImagingPlotter` and call various `figure_*` methods to plot different attributes.
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
__Subplots__

The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_galaxies(galaxy_index=1)

"""
__Symmetric Residual Maps__

By default, the `residual_map` and `normalized_residual_map` use a symmetric colormap. 

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    residual_map=True,
    normalized_residual_map=True,
)

"""`
__Include__

`FitImaging` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
    tangential_caustics=True,
    radial_caustics=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_galaxies(galaxy_index=0)
fit_plotter.subplot_of_galaxies(galaxy_index=1)

"""
__Pixelization__

We can also plot a `FitImaging` which uses a `Pixelization`.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(25, 25)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Include__

The `figures_2d_of_galaxies` method now plots the reconstructed galaxy on the Rectangular pixel-grid. It can use the
`Include2D` object to plot the `Mapper`'s specific structures like the pixelization grids.
"""
include = aplt.Include2D(
    mapper_image_plane_mesh_grid=True, mapper_source_plane_data_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, model_image=True)

"""
__Inversion Plotter__

We can even extract an `InversionPlotter` from the `FitImagingPlotter` and use it to plot all of its usual methods,
which will now include the caustic and border.
"""
inversion_plotter = fit_plotter.inversion_plotter
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)
