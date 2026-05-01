"""
Over Sampling
=============

Throughout the workspace, we have created 2D grids of (y,x) coordinates and input them into light profiles to
compute their image.

This calculates how much of the light profile's emission is observed with every 2D pixel defined on the grid.

However, there is a problem. If we only input the (y,x) coordinates at the centre of every pixel, we are not
evaluating how the entire light profile is observed within that pixel. If the light profile has a very steep gradient
in intensity from one edge of the pixel to the other, only evaluating the intensity at the centre of the pixel will
not give an accurate estimate of the total amount of light that falls within that pixel.

Over-sampling addresses this problem. Instead of evaluating the light profile at the centre of every pixel, we
evaluate it using a sub-grid of coordinates within every pixel and take the average of the intensity values.
Provided the sub-grid is high enough resolution that it "over-samples" the light profile within the pixel enough, this
will give an accurate estimate of the total intensity within the pixel.

__Contents__

**Illustration:** Visualize how over-sampling splits pixels into sub-grids of coordinates.
**Numerics:** Inspect the numerical values of sub-pixel coordinates in the over-sampled grid.
**Images:** Compute Sersic profile images with different over-sampling levels and compare residuals.
**Adaptive Over Sampling:** Use radial bins to apply high over-sampling only where needed.
**Default Over-Sampling:** The default over-sampling scheme used by PyAutoGalaxy.
**Multiple Galaxies:** Apply adaptive over-sampling with multiple galaxy centres.
**Dataset & Modeling:** Apply over-sampling to imaging datasets for model fitting.
**Pixelization:** Apply over-sampling to pixelized source reconstructions.

__Default Over-Sampling__

Examples throughout the workspace use a default over-sampling set up that should ensure accurate results for any
analysis you have done.

- When evaluating the image of a galaxy, an adaptive over sampling grid is used which uses sub grids of size 8 x 8
in the central regions of the image, 4x4 further out and 1x1 beyond that.

This guide will explain why these choices were made for the default over-sampling behaviour.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Illustration__

To illustrate over sampling, lets first create a uniform grid which does not over sample the pixels, using 
the `over_sample_size` input.

The input below uses `over_sample_size=1`, therefore each pixel is split into a sub-grid of 
size  `over_sample_size x over_sample_size` = `1 x 1`. This means the light profile is evaluated once at the centre of each pixel, 
which is equivalent to not over-sampling the grid at all.  
"""
grid_sub_1 = ag.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sample_size=1,
)

"""
We now plot the grid, over laying a uniform grid of pixels to illustrate the area of each pixel within which we
want light profile intensities to be computed.
"""
aplt.plot_grid(grid=grid_sub_1, title="Grid Without Over-Sampling")

"""
We now create and plot a uniform grid which does over-sample the pixels, by inputting `over_sample_size=2`.
"""
grid_sub_2 = ag.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sample_size=2,
)

"""
If we print `grid_sub_2` and its shape, we will find it is actually identical to `grid_sub_1`, despite the change
in `over_sample_size`:
"""
print(grid_sub_1)
print(grid_sub_2)
print(grid_sub_1.shape)
print(grid_sub_2.shape)

"""
This is because the over sampled version of the grid is stored in a separate attribute, called `over_sampled`,
which we print below.

We see that for `grid_sub_1` and `grid_sub_2` the `over_sampled` grids are different, with the over sampled grid for
`grid_sub_2` containing four times as many entries corresponding to each pixel being sub-gridded in a 2 x 2 shape.
"""
print(grid_sub_1.over_sampled)
print(grid_sub_2.over_sampled)
print(grid_sub_1.over_sampled.shape)
print(grid_sub_2.over_sampled.shape)

"""
We now plot the over sampled grid over the image, showing that each pixel is now split into a 2x2 sub-grid of 
coordinates. 

These are used to compute the intensity of the light profile and therefore more accurately estimate the total 
intensity within each pixel if there is a significant gradient in intensity within the pixel.

In the code below, it is the input `plot_over_sampled_grid=True` which ensures we plot the over sampled grid.
"""
aplt.plot_grid(
    grid=grid_sub_2,
    title="Grid With 2x2 Over-Sampling",
    plot_over_sampled_grid=True,
)

"""
__Numerics__

Lets quickly check how the sub-grid is defined and stored numerically.

The first four pixels of this sub-grid correspond to the first four sub-pixels in the first pixel of the grid. 

The top-left pixel image above shows how the sub-pixels are spaced within the pixel. 
"""
print("(y,x) pixel 0 of grid_sub_1:")
print(grid_sub_1.over_sampled[0])
print("(y,x) pixel 0 of grid_sub_2:")
print(grid_sub_2.over_sampled[0])

"""
We now confirm that the first four sub-pixels of the over-sampled grid correspond are contained within the 
first pixel of the grid.
"""
print("(y,x) pixel 0 (of original grid):")
print(grid_sub_2[0])
print("(y,x) sub-pixel 0 (of pixel 0):")
print(grid_sub_2.over_sampled[0])
print("(y,x) sub-pixel 1 (of pixel 0):")
print(grid_sub_2.over_sampled[1])
print("(y,x) sub-pixel 2 (of pixel 0):")
print(grid_sub_2.over_sampled[2])
print("(y,x) sub-pixel 3 (of pixel 0):")
print(grid_sub_2.over_sampled[3])

"""
Numerically, the over-sampled grid contains the sub-pixel coordinates of every pixel in the grid, going from the 
first top-left pixel right and downwards to the bottom-right pixel. 

So the pixel to the right of the first pixel is the next 4 sub-pixels in the over-sampled grid, and so on.

__Images__

We now use over-sampling to compute the image of a Sersic light profile, which has a steep intensity gradient
at its centre which a lack of over-sampling does not accurately capture.

We create the light profile, input the two grids (with `over_sample_size=1` and `over_sample_size=2`) and compute 
the image of the light profile using each grid. We then plot the residuals between the two images in order to show the 
difference between the two images and thus why over-sampling is important.

Over sampling occurs automatically when a grid is input into a function like `image_2d_from`, therefore internally 
the line of code, `image_sub_2 = light.image_2d_from(grid=grid_sub_2)`, is evaluating the light profile using the
2 x 2 oversampled grid and internally binning it up in to fully perform over sampling.
"""
light = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=1.0,
    effective_radius=0.2,
    sersic_index=3.0,
)

image_sub_1 = light.image_2d_from(grid=grid_sub_1)
image_sub_2 = light.image_2d_from(grid=grid_sub_2)

aplt.plot_array(array=image_sub_1, title="Image of Sersic Profile")

residual_map = image_sub_2 - image_sub_1

aplt.plot_array(array=residual_map, title="Over-Sampling Residuals")


"""
In the central 4 pixels of the image, the residuals are large due to the steep intensity gradient of the Sersic
profile at its centre. 

The gradient in these pixels is so steep that evaluating the intensity at the centre of the pixel, without over 
sampling, does not accurately capture the total intensity within the pixel.

At the edges of the image, the residuals are very small, as the intensity gradient of the Sersic profile is very 
shallow and it is an accurate approximation to evaluate the intensity at the centre of the pixel.

The absolute value of the central residuals are 0.74, however it is difficult to assess whether this is a large or
small value. We can quantify this by dividing by the evaluated value of the Sersic image in each pixel in order
to compute the fractional residuals.
"""
fractional_residual_map = residual_map / image_sub_2

aplt.plot_array(array=fractional_residual_map, title="Fractional Residuals")

"""
The fractional residuals in the centre exceed 0.1, or 10%, which is a significant error in the image and
demonstrates why over-sampling is important.

Lets confirm sub-griding can converge to central residuals that are very small.

The fractional residuals with high levels of over-sampling are below 0.01, or 1%, which is sufficiently accurate
for most scientific purposes (albeit you should think carefully about the level of over-sampling you need for
your specific science case).
"""
grid_sub_16 = ag.Grid2D.uniform(
    shape_native=(40, 40), pixel_scales=0.1, over_sample_size=16
)
grid_sub_32 = ag.Grid2D.uniform(
    shape_native=(40, 40), pixel_scales=0.1, over_sample_size=32
)

image_sub_16 = light.image_2d_from(grid=grid_sub_16)
image_sub_32 = light.image_2d_from(grid=grid_sub_32)

residual_map = image_sub_32 - image_sub_16

aplt.plot_array(array=residual_map, title="Over-Sampling Residuals (32 vs 16)")

fractional_residual_map = residual_map / image_sub_32

aplt.plot_array(array=fractional_residual_map, title="Fractional Residuals (32 vs 16)")

"""
__Adaptive Over Sampling__

We have shown that over-sampling is important for accurate image evaluation. However, there is a major drawback to
over-sampling, which is that it is computationally expensive. 

For example, for the 32x32 over-sampled grid above, 1024 sub-pixels are used in every pixel, which must all be 
evaluated using the Sersic light profile. The calculation of the image is therefore at least 1000 times slower than if
we had not used over-sampling.

Speeding up the calculation is crucial for model-fitting where the image is evaluated many times to fit the
model to the data.

Fortunately, there is a solution to this problem. We saw above that the residuals rapidly decrease away
from the centre of the light profile. Therefore, we only need to over-sample the central regions of the image,
where the intensity gradient is steep. We can use lower levels of over-sampling away from the centre, which
will be fast to evaluate.

Up to now, the `over_sample_size` input has been an integer, however it can also be an `ndarray` of values corresponding
to each pixel. We create an `ndarray` of values which are high in the centre, but reduce to 2 at the outskirts,
therefore providing high levels of over sampling where we need it whilst using lower values which are computationally
fast to evaluate at the outskirts.

Specifically, we define a 24 x 24 sub-grid within the central 0.3" of pixels, uses a 8 x 8 grid between
0.3" and 0.6" and a 2 x 2 grid beyond that. 
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid_sub_1, sub_size_list=[24, 8, 2], radial_list=[0.3, 0.6]
)

grid_adaptive = ag.Grid2D.no_mask(
    values=grid_sub_1.native,
    pixel_scales=grid_sub_1.pixel_scales,
    over_sample_size=over_sample_size,
)

aplt.plot_grid(
    grid=grid_adaptive,
    title="Adaptive Over-Sampling",
    plot_over_sampled_grid=True,
)

print(over_sample_size)


"""
Modeling uses masked grids, therefore the code below shows how we would create this adaptive over sample grid via 
a circular mask, which can be used for modeling.

Throughout the modeling examples in the workspace, we use this adaptive grid to ensure that the image of the
galaxy is evaluated accurately and efficiently.
"""
mask = ag.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=5.0)

grid = ag.Grid2D.from_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid, sub_size_list=[24, 8, 2], radial_list=[0.3, 0.6]
)

grid_adaptive = ag.Grid2D(values=grid, mask=mask, over_sample_size=over_sample_size)

aplt.plot_grid(
    grid=grid_adaptive,
    title="Adaptive Over-Sampling",
    plot_over_sampled_grid=True,
)

"""
We can compare this adaptive grid to the grid with over sampling of 32 x 32 to confine it produces low amounts
of residuals.
"""
image_adaptive = light.image_2d_from(grid=grid_adaptive)
image_sub_32 = light.image_2d_from(grid=grid_sub_32)

residual_map = image_adaptive - image_sub_32

fractional_residual_map = residual_map / image_sub_32

aplt.plot_array(array=fractional_residual_map, title="Adaptive Fractional Residuals")

"""
__Default Over-Sampling__

The default over-sampling scheme used by the source code is 4 x 4 uniform over sampling over the whole image. 

A uniform scheme is used, instead of the adaptive scheme above, because the adaptive scheme requires input knowledge of 
where the centre of the galaxy is (e.g. above the centre is at (0.0", 0.0").

Uniform over sampling is precise enough for many calculations, especially when you are simply performing quick 
calculations to investigate a problem. However, for detailed calculations you must ensure that high enough
levels of over sampling are used.

For modeling, all example scripts begin by switching to an adaptive over sampling scheme, as modeling assumes
the centre of the galaxy is at (0.0", 0.0").

__Multiple Galaxies__

The analysis may contain multiple galaxies, each of which must be over-sampled accurately. 

There are two approaches you can take to over sampling multi-galaxy systems:

1) Use a high level of uniform over sampling over the full image.

2) Use an adaptive over sampling scheme with multiple centres of high over sampling levels, with the API shown below
  for two galaxies with centres (1.0, 0.0) and (-1.0, 0.0).
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid_sub_1,
    sub_size_list=[24, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(1.0, 0.0), (-1.0, 0.0)],
)

"""
__Dataset & Modeling__

Throughout this guide, grid objects have been used to compute the image of light profiles and illustrate over sampling.

If you are performing calculations with imaging data or want to fit a model to the data with a specific
over-sampling level, the `apply_over_sampling` method is used to update the over sampling scheme of the dataset.

The grid this is applied to is called `lp`, to indicate that it is the grid used to evaluate the emission of light
profiles for which this over sampling scheme is applied.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator_sersic.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

over_sample_size = 4  # This could be an adaptive over sample grid.

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
)

"""
__Pixelization__

Source galaxies can be reconstructed using pixelizations, which discretize the source's light onto a mesh,
for example a Voronoi mesh.

Over sampling is used by pixelizations in an analogous way to light profiles. By default, a 4 x 4 sub-grid is used,
whereby every image pixel is ray-traced on its 4 x 4 sub grid to the source mesh and fractional mappings are computed.

A different grid and over sampling scheme is applied to light profiles and pixelizations, which is why
there are separate inputs called `lp` and `pix`.

This is explained in more detail in the pixelization examples.

Here is an example of how to change the over sampling applied to a pixelization for a lens model fit:
"""
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=4)

"""
Finish.
"""
