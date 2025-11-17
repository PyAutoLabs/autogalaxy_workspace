"""
Plots: MultiYX1DPlotter
=========================

This example illustrates how to plot multi 1D figure lines on the same plot.

It uses the specific example of plotting a `LightProfile`'s 1D image using multiple `LightProfilePlotter`'s.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag

"""
First, lets create two simple `LightProfile`'s which we'll plot the 1D images of on the same figure.
"""
light_0 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=1.0,
    sersic_index=2.0,
)

light_1 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=2.0,
    sersic_index=2.0,
)

"""
We also need the 2D grid the `LightProfile`'s are evaluated on.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
