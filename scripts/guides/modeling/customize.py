"""
Modeling: Customize
===================

This script gives a run through of all the different ways the analysis can be customized for modeling, with
reasons explaining why each customization is useful.

__Contents__

**Dataset**: Load a dataset which is used to illustrate the customizations.
**Mask:** Apply a custom mask to the dataset, which can be used to remove regions of the image that contain no emission.
**Over Sampling:** Change the over sampling used to compute the surface brightness of every image-pixel.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

All customizations in this script are applied to the strong dataset `simple`, which is a
simple strong with no light emission.

We therefore load and plot the strong dataset `simple` via .fits files.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Mask__

All example default scritps use a circular mask to model a lens, which contains the and source emission.
However, the mask can be customized to better suit the and source emission, for example by using an annular
mask to only contain the emission of the Einstein ring itelf.

Advantages: Galaxies with complex and difficult-to-subtract foreground galaxies can leave residuals that
bias the galaxy model, which this custom mask can remove from the model-fit. The custom mask can also provide
faster run times, as the removal of large large regions of the image (which contain no signal) no longer need to be
processed and fitted.

Disadvantages: Pixels containing no galaxy emission may still constrain the model, if a model incorrectly
predicts that flux will appear in these image pixels. By using a custom mask, the model-fit will not be penalized for
incorrectly predicting flux in these image-pixels (As the mask has removed them from the fit).

We first show an example using an annular masks, which removes the central pixels and thus only fit the outer regions.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.5,
    outer_radius=2.5,
)

dataset = dataset.apply_mask(mask=mask)  # <----- The custom mask is used here!

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
We can also load the mask from a .fits file, which could have been produced in a way which is even more customized
to the source emission than the annular masks above.

To create the .fits file of a mask, we use a GUI tool which is described in the following script:

 `autolens_workspace/*/data_preparation/imaging/gui/mask.py`
 
We reload the data to restore it to its original shape, as the previous cell applied a mask to it which changed its
shape to prepare for the fast Fourier transforms.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_gui.fits"),
    hdu=0,
    pixel_scales=dataset.pixel_scales,
)

dataset = dataset.apply_mask(mask=mask)  # <----- The custom mask is used here!

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

f the light profile has a very steep gradient in intensity from one edge of the pixel to the other, like a Sersic
profile does towards its centre, over sampling is necessary to evaluate to total emission observed in that pixel
correctly.

This example demonstrates how to change the over sampling used to compute the surface brightness of every image-pixel,
whereby a higher sub-grid resolution better oversamples the image of the light profile so as to provide a more accurate
model of its image.

**Benefit**: Higher level of over sampling provide a more accurate estimate of the surface brightness in every image-pixel.
**Downside**: Higher levels of over sampling require longer calculations and higher memory usage.

Prequisites: You should read `autogalaxy_workspace/*/guides/advanced/over_sampling.ipynb` before running this script, which
introduces the concept of over sampling in PyAutoand explains why the and source galaxy are evaluated
on different grids.



The over sampling used to fit the data is customized using the `apply_over_sampling` method, which you may have
seen in example `modeling` scripts.

To apply uniform over sampling of degree 4x4, we simply input the integer 4.

The grid this is applied to is called `lp`, to indicate that it is the grid used to evaluate the emission of light
profiles for which this over sampling scheme is applied.
"""
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

"""
Above, the `over_sample_size` input has been an integer, however it can also be an `ndarray` of values corresponding
to each pixel. 

We create an `ndarray` of values which are high in the centre, but reduce to 2 at the outskirts, therefore 
providing high levels of over sampling where we need it whilst using lower values which are computationally fast to 
evaluate at the outskirts.

Specifically, we define a 24 x 24 sub-grid within the central 0.3" of pixels, uses a 8 x 8 grid between
0.3" and 0.6" and a 2 x 2 grid beyond that. 

This will provide high levels of over sampling for the galaxy, whose emission peaks at the centre of the
image near (0.0", 0.0"), but will not produce high levels of over sampling for the lensed source.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[24, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
Finish.
"""
