"""
Plots: Visuals
==============

This example illustrates the API for adding visuals to plots and customizing their appearance.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

__Setup__

To illustrate plotting, we require standard objects like a grid, galaxies and dataset.
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

galaxy = ag.Galaxy(
    redshift=1.0,
    bulge_0=ag.lp.SersicSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
    bulge_1=ag.lp.SersicSph(
        centre=(0.4, 0.3), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

dataset_path = Path("dataset") / "imaging" / "complex"
data_path = dataset_path / "data.fits"
data = ag.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.1)

"""
__Light Profile Centres__

The centres of all light profiles in the galaxies (or other object, like a galaxy) can be extracted and plotted.
"""
light_profile_centres = galaxies.extract_attribute(
    cls=ag.LightProfile, attr_name="centre"
)

visuals = aplt.Visuals2D(light_profile_centres=light_profile_centres)
image = galaxies.image_2d_from(grid=grid)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=grid, visuals_2d=visuals
)
galaxies_plotter.figures_2d(image=True)

"""
The appearance of the light profile centres are customized using a `LightProfileCentresScatter` object.

To plot the light profile centres this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
light_profile_centres = galaxies.extract_attribute(
    cls=ag.LightProfile, attr_name="centre"
)

visuals = aplt.Visuals2D(light_profile_centres=light_profile_centres)
image = galaxies.image_2d_from(grid=grid)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=grid, visuals_2d=visuals
)
galaxies_plotter.figures_2d(image=True)


"""
By specifying two colors to the `LightProfileCentresScatter` object the light profile centres of each plane
are plotted in different colors.

The plot below uses the `tracer_x2` object which consists of multiple galaxies with multiple light profiles.
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(c=["r", "w"], s=150)

mat_plot = aplt.MatPlot2D(light_profile_centres_scatter=light_profile_centres_scatter)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
galaxies_plotter.figures_2d(image=True)


"""
__Mask__

The mask is plotted over all images by default as black points.

We now show how to manually pass in a mask to plot and customize its appearance.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = ag.Array2D(values=data.native, mask=mask)

visuals = aplt.Visuals2D(mask=mask)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the mask is customized using a `Scatter` object.

To plot the mask this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
mask_scatter = aplt.MaskScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(mask_scatter=mask_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Origin__

We can plot the (y,x) origin on the data to show where the grid is defined from.

By default the origin of (0.0", 0.0") is at the centre of the image.
"""
visuals = aplt.Visuals2D(origin=ag.Grid2DIrregular(values=[(1.0, 1.0)]))

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the (y,x) origin coordinates is customized using a `Scatter` object.

To plot these (y,x) grids of coordinates these objects wrap the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html

The example script `plot/mat_wrap/Scatter.py` gives a more detailed description on how to customize its appearance.
"""
origin_scatter = aplt.OriginScatter(marker="o", s=50)

mat_plot = aplt.MatPlot2D(origin_scatter=origin_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Grid__

We can plot a grid of (y,x) coordinates over an image.

We'll use a uniform grid at a coarser resolution than our dataset.
"""
grid = ag.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)

visuals = aplt.Visuals2D(grid=grid)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We customize the grid's appearance using the `GridScatter` `matplotlib wrapper object which wraps the following method(s): 

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
grid_scatter = aplt.GridScatter(c="r", marker=".", s=1)

mat_plot = aplt.MatPlot2D(grid_scatter=grid_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Border__

A border is the `Grid2D` of (y,x) coordinates at the centre of every pixel at the border of a mask. 

A border is defined as a pixel that is on an exterior edge of a mask (e.g. it does not include the inner pixels of 
an annular mask).

Borders are rarely plotted, but are important when it comes to defining the edge of the source-plane for pixelized
source reconstructions, with examples on this topic sometimes plotting the border.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = ag.Array2D(values=data.native, mask=mask)

visuals = aplt.Visuals2D(border=mask.derive_grid.border)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the border is customized using a `BorderScatter` object.

To plot the border this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
border_scatter = aplt.BorderScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(border_scatter=border_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Array Overlay__

We can overlay a 2D array over an image.
"""
arr = ag.Array2D.no_mask(
    values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=0.5
)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We customize the overlaid array using the `ArrayOverlay` matplotlib wrapper object which wraps the following method(s):

To overlay the array this objects wrap the following matplotlib method:

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
"""
array_overlay = aplt.ArrayOverlay(alpha=0.5)

mat_plot = aplt.MatPlot2D(array_overlay=array_overlay)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Patch Overlay__

The matplotlib patch API can be used to plot shapes over an image.

This is used in certain weak lensing plots to visualize galaxy shapes and ellipticities.

To plot a patch on an image, we use the `matplotlib.patches` module. 

In this example, we will use the `Ellipse` patch.
"""
from matplotlib.patches import Ellipse

patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
patch_1 = Ellipse(xy=(-2.0, -3.0), height=1.0, width=2.0, angle=1.0)

visuals = aplt.Visuals2D(patches=[patch_0, patch_1])

array_plotter = aplt.Array2DPlotter(array=data)  # , visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the patches using the `Patcher` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/collections_api.html
"""
patch_overlay = aplt.PatchOverlay(
    facecolor=["r", "g"], edgecolor="none", linewidth=10, offsets=3.0
)

mat_plot = aplt.MatPlot2D(patch_overlay=patch_overlay)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Vector Field__

A quiver plot showing vectors (e.g. 2D (y,x) directions at (y,x) coordinates) can be plotted using the `matplotlib`
`quiver` function.

This is often used for weak lensing plots, showing the direction and magnitude of weak lensing infrred at each
galaxy's location.
"""
vectors = ag.VectorYX2DIrregular(
    values=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)
visuals = aplt.Visuals2D(vectors=vectors)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the appearance of the vectors using the `VectorYXQuiver` matplotlib wrapper object which wraps 
the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
"""
quiver = aplt.VectorYXQuiver(
    headlength=1,
    pivot="tail",
    color="w",
    linewidth=10,
    units="width",
    angles="uv",
    scale=None,
    width=0.5,
    headwidth=3,
    alpha=0.5,
)

mat_plot = aplt.MatPlot2D(vector_yx_quiver=quiver)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Half Light Radius__

For 1D plots of a light profile (e.g. radius vs intensity) a 1D line of its half light radius can be plotted.
"""
visuals = aplt.Visuals1D(half_light_radius=galaxies.bulge_0.half_light_radius)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=galaxies.bulge_0, grid=grid, visuals_1d=visuals
)
light_profile_plotter.figures_1d(image=True)

"""
The appearance of the half-light radius is customized using a `HalfLightRadiusAXVLine` object.

To plot the half-light radius as a vertical line this wraps the following matplotlib method:

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
half_light_radius_axvline = aplt.HalfLightRadiusAXVLine(
    linestyle="-.", c="r", linewidth=20
)

mat_plot = aplt.MatPlot1D(half_light_radius_axvline=half_light_radius_axvline)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=galaxies.bulge_0, grid=grid, mat_plot_1d=mat_plot, visuals_1d=visuals
)
light_profile_plotter.figures_1d(image=True)


"""
Finish.
"""
