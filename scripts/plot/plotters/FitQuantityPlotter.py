"""
Plots: FitQuantityPlotter
========================

This example illustrates how to plot a `FitQuantity` object using a `FitQuantityPlotter`.

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
__Grid__

Define the 2D grid the quantity (in this example, the image) is evaluated using.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

"""
__Galaxies__

Create galaxies which we will use to create our `DatasetQuantity`.
"""
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

"""
__Dataset__

Use this `Plane`'s 2D image to create the `DatasetQuantity`.

We assume a noise-map where all values are arbritrarily 0.01.
"""
image = galaxies.image_2d_from(grid=grid)

dataset = ag.DatasetQuantity(
    data=image,
    noise_map=ag.Array2D.full(
        fill_value=0.01,
        shape_native=image.shape_native,
        pixel_scales=image.pixel_scales,
    ),
)

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit, which we define and apply to the 
`DatasetQuantity` object.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Fit__

We now fit the `DatasetQuantity` with a `Plane`'s to create a `FitQuantity` object.

This `Plane` has a slightly different galaxy and therefore image map, creating residuals in the plot.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.05, 0.05),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

galaxies_fit = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitQuantity(
    dataset=dataset, light_mass_obj=galaxies_fit, func_str="image_2d_from"
)

"""
__Figures__

We now pass the FitQuantity to an `FitQuantityPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_dataset_plotter = aplt.FitQuantityPlotter(fit=fit)
fit_dataset_plotter.figures_2d(
    image=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
__Subplots__

The `FitQuantityPlotter` may also plot a subplot of these attributes.
"""
fit_dataset_plotter.subplot_fit()

"""`
__Include__

`FitQuantity` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(origin=True, mask=True, light_profile_centres=True)

fit_plotter = aplt.FitQuantityPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
