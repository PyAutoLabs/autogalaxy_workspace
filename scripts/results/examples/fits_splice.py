"""
Results: Fits Splice
====================

In this tutorial, we use the aggregator to load .fits files output by a model-fit, extract hdu images and create
new .fits files, for example all to a single folder on your hard-disk.

For example, a common use case is extracting an image from `model_galaxy_images.fits` of many fits and putting them
into a single .fits file on your hard-disk. If you have modeled 100+ datasets, you can then inspect all model images
in DS9 in .fits format n a single folder, which is more efficient than clicking throughout the `output` open each
.fits file one-by-one.

The most common use of .fits splciing is where multiple observations of the same galaxy are analysed, for example
at different wavelengths, where each fit outputs a different .fits files. The model images of each fit to each
wavelength can then be packaged up into a single .fits file.

This enables the results of many model-fits to be concisely visualized and inspected, which can also be easily passed
on to other collaborators.

Internally, splicing uses standard Astorpy functions to open, edit and save .fit files.

__Interferometer__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are:

 - `ImagingAgg` -> `InterferometerAgg`.
 - `FitImagingAgg` -> `FitInterferometerAgg`.
 - `ImagingPlotter` -> `InterferometerPlotter`.
 - `FitImagingPlotter` -> `FitInterferometerPlotter`.

Quantities specific to an interfometer, for example its uv-wavelengths real space mask, are accessed using the same API
(e.g. `values("dataset.uv_wavelengths")` and `.values{"dataset.real_space_mask")).

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
from PIL import Image

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
for i in range(2):
    dataset_name = f"simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = ag.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    bulge = af.Model(ag.lp_linear.Sersic)
    disk = af.Model(ag.lp_linear.Exponential)
    bulge.centre = disk.centre

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    search = af.Nautilus(
        path_prefix=path.join("results_folder_csv"),
        name="results",
        unique_tag=f"simple_{i}",
        n_live=100,
        number_of_cores=1,
    )

    class AnalysisLatent(ag.AnalysisImaging):
        def compute_latent_variables(self, instance):
            return {"example_latent": instance.galaxies.galaxy.bulge.sersic_index * 2.0}

    analysis = AnalysisLatent(dataset=dataset)

    result = search.fit(model=model, analysis=analysis)

"""
__Aggregator__

First, set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder_csv"),
)

"""
Extract the `AggregateFits` object, which has specific functions for loading .fits files and outputting results in 
.fits format.
"""
agg_fits = af.AggregateFits(aggregator=agg)

"""
__Extract Images__

We now extract 2 images from the `fit.fits` file and combine them together into a single .fits file.

We will extract the `model_image` and `residual_map` images, which are images you are used to
plotting and inspecting in the `output` folder of a model-fit and can load and inspect in DS9 from the file
`fit.fits`.

By inspecting `fit.fits` you will see it contains four images which each have a an `ext_name`: `model_image`,
`residual_map`, `normalized_residual_map`, `chi_squared_map`.

We do this by simply passing the `agg_fits.extract_image` method the name of the fits file we load from `fits.fit`
and the `ext_name` of what we extract.

This runs on all results the `Aggregator` object has loaded from the `output` folder, meaning that for this example
where two model-fits are loaded, the `image` object contains two images.
"""
image = agg_fits.extract_image(
    name="fits.fit",
    ext_name_list=["model_image", "residual_map"]
)


"""
__Output Single Fits__

The `image` object which has been extracted is an `astropy` `Fits` object, which we use to save the .fits to the 
hard-disk.

The .fits has 4 hdus, the `model_image` and `residual_map` for the two datasets fitted.
"""
image.save("png_splice_single_subplot.png")

"""
__Output to Folder__

An alternative way to output the .fits files is to output them as single .fits files for each model-fit in a single 
folder, which is done using the `output_to_folder` method.

It can sometimes be easier and quicker to inspect the results of many model-fits when they are output to individual
files in a folder, as using an IDE you can click load and flick through the images. This contrasts a single .png
file you scroll through, which may be slower to load and inspect.
"""
# agg_fits.output_to_folder(
#     name="png_splice_single_subplot",
#     path="output_folder",
# )

"""
__Naming Convention__

By default, each subplot uses the input `name` with an integer index prefix to name the subplot, which is how
the output folder .png files above are named.

It is more informative to name the .fits files after the dataset name. A list of the `name`'s of the datasets can be
input to the `output_to_folder` method, which will name the subplots after the dataset names. 

To use the names we use the `Aggregator` to loop over the `search` objects and extract their `unique_id`'s, which 
when we fitted the model above used the dataset names. This API can also be used to extract the `name` or `path_prefix`
of the search and build an informative list for the names of the subplots.
"""
# name_list = [search.unique_tag for search in agg.values("search")]
#
# agg_fits.output_to_folder(
#      name_list=name_list,
#      path="output_folder",
#  )

"""
__Add Extra Fits__

We can also add an extra .fits image to the extracted .fits file, for example an RGB image of the dataset.

We create an image of shape (1, 2) and add the RGB image to the left of the subplot, so that the new subplot has
shape (1, 3).

When we add a single .png, we cannot extract or splice it, it simply gets added to the subplot.
"""
# image_rgb = Image.open(path.join(dataset_path, "rgb.png"))
#
# image = agg_fits.extract_image(
#     ag.agg.subplot_dataset.data,
#     ag.agg.subplot_dataset.psf_log_10,
#     subplot_shape=(1, 2),
# )

# image = ag.add_image_to_left(image, additional_img)

# image.save("png_splice_with_rgb.png")

"""
__Custom Fits Files in Analysis__

Describe how a user can extend the `Analysis` class to compute custom images that are output to the .png files,
which they can then extract and splice together.
"""

"""
__Path Navigation__

Example combinig `fit.fits` from `source_lp[1]` and `mass_total[0]`.
"""
