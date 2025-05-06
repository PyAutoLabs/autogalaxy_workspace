"""
Customize: Custom Mask
======================

This example demonstrates how to use a custom mask (tailored to the galaxyed source galaxy's light distribution)
in a model-fit.

__Advantages__

Galaxies with complex and difficult-to-subtract foreground galaxies can leave residuals that
bias the galaxy model, which this custom mask can remove from the model-fit. The custom mask can also provide
faster run times, as the removal of large large regions of the image (which contain no signal) no longer need to be
processed and fitted.

__Disadvantages__

Pixels containing no galaxy emission may still constrain the model, if a model incorrectly
predicts that flux will appear in these image pixels. By using a custom mask, the model-fit will not be penalized for
incorrectly predicting flux in these image-pixels (As the mask has removed them from the fit).

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load and plot the galaxy dataset `simple__sersic` via .fits files, which we will fit with the model.
"""
dataset_name = "simple__sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

We will load the mask from a .fits file, like we did `Imaging` above. 

To create the .fits file of a mask, we use a GUI tool which is described in the following script:

 `autogalaxy_workspace/*/preprocess/imaging/gui/mask.py`
"""
mask_custom = ag.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_gui.fits"),
    hdu=0,
    pixel_scales=dataset.pixel_scales,
)

dataset = dataset.apply_mask(mask=mask_custom)  # <----- The custom mask is used here!

"""
When we plot the `Imaging` dataset with the mask it extracts only the regions of the image in the mask remove 
contaminating bright sources away from the galaxy and zoom in around the mask to emphasize the galaxy.
"""
visuals = aplt.Visuals2D(mask=mask_custom)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Analysis__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("imaging", "customize"),
    name="custom_mask",
    unique_tag=dataset_name,
)

analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Because the `AnalysisImaging` was passed a `Imaging` with the custom mask, this mask is used by the model-fit.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitImaging` object we can confirm the custom mask was used.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
