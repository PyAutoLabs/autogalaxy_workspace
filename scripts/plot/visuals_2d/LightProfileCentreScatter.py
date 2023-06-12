"""
Plots: LightProfileCentreScatter
================================

This example illustrates how to customize the light profile centres plotted over data.
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
This means the centre of every `LightProfile` of every `Galaxy` in a plot are plotted on the figure. 
A `Plane` object is a good example of an object with many `LightProfiles`, so lets make one with three.

We will show the plots in the image-plane, however it is the centre's of the source galaxy `LightProfile`'s in the 
source-plane that are plotted.
"""
galaxy = ag.Galaxy(
    redshift=1.0,
    bulge_0=ag.lp.SersicSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
    bulge_1=ag.lp.SersicSph(
        centre=(0.4, 0.3), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

plane = ag.Plane(galaxies=[galaxy])

"""
We also need the `Grid2D` that we can use to make plots of the `Plane`'s properties.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
The light profile centres are an internal property of the `Plane`, so we can plot them via an `Include2D` object.
"""
include = aplt.Include2D(light_profile_centres=True)
plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid, include_2d=include)
plane_plotter.figures_2d(image=True)


"""
The appearance of the light profile centres are customized using a `LightProfileCentresScatter` object.

To plot the light profile centres this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(
    marker="o", c="r", s=150
)
mat_plot = aplt.MatPlot2D(light_profile_centres_scatter=light_profile_centres_scatter)
plane_plotter = aplt.PlanePlotter(
    plane=plane, grid=grid, include_2d=include, mat_plot_2d=mat_plot
)
plane_plotter.figures_2d(image=True)

"""
By specifying two colors to the `LightProfileCentresScatter` object the light profile centres of each plane
are plotted in different colors.
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(c=["r", "w"], s=150)
mat_plot = aplt.MatPlot2D(light_profile_centres_scatter=light_profile_centres_scatter)
plane_plotter = aplt.PlanePlotter(
    plane=plane, grid=grid, include_2d=include, mat_plot_2d=mat_plot
)
plane_plotter.figures_2d(image=True)


"""
To plot the light profile centres manually, we can pass them into a` Visuals2D` object. This is useful for plotting 
the centres on figures where they are not an internal property, like an `Array2D`.
"""
light_profile_centres = plane.extract_attribute(cls=ag.LightProfile, attr_name="centre")

visuals = aplt.Visuals2D(light_profile_centres=light_profile_centres)
image = plane.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
Finish.
"""
